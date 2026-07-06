"""
app.py
------
Main Streamlit app. Run with:  streamlit run app.py

What it does, every time a package is shown to the webcam:
1. Reads the live webcam feed (OpenCV)
2. Scans for a barcode (pyzbar) -> identifies the SKU
3. Detects the package outline -> estimates width/height in cm
4. Runs damage detection (trained model if available, else classical CV rules),
   smoothed over multiple frames so single noisy frames don't flip the verdict
5. Decides a storage zone based on size + condition
6. Logs ONCE per physical box shown (not once per frame/timer) via pandas
7. Shows a live dashboard + lets you download the full log
"""

import streamlit as st
import cv2
import time
import pandas as pd

from barcode_scanner import scan_barcode
from damage_detector import (
    find_package_contour,
    estimate_size_cm,
    categorize_size,
    detect_damage,
    assign_zone,
    load_trained_model,
)
from inventory import load_log, append_entry, make_entry, get_stats, clear_log

st.set_page_config(page_title="Warehouse Inbound Scanner", layout="wide")

# ---------- SESSION STATE SETUP ----------
if "log_df" not in st.session_state:
    st.session_state.log_df = load_log()

# Tracks the barcode currently "in front of" the camera, so we log it once
# and ignore it while it's still there (even if rotated).
if "active_barcode" not in st.session_state:
    st.session_state.active_barcode = None

# Counts consecutive frames with NO barcode seen, to detect when a box has
# actually been removed (vs. just briefly turned/occluded).
if "absent_count" not in st.session_state:
    st.session_state.absent_count = 0

# Rolling buffer of recent damage scores, so one noisy frame near the
# decision boundary can't flip the verdict by itself.
if "score_buffer" not in st.session_state:
    st.session_state.score_buffer = []

ABSENT_FRAMES_THRESHOLD = 15   # ~0.5 sec of no barcode = box was actually removed
SCORE_BUFFER_SIZE = 10          # average damage score over the last N frames

model = load_trained_model()

# ---------- HEADER ----------
st.title("📦 Automated Warehouse Inbound Scanner")
st.caption(
    "Live webcam scanning: barcode ID + damage detection + size estimation + auto-logging"
)

if model is not None:
    st.success("Using TRAINED damage classifier (models/damage_classifier.pkl)")
else:
    st.info(
        "No trained model found yet — using classical CV heuristic for damage detection. "
        "Run `python train_damage_model.py` after collecting sample photos to switch to a trained model."
    )

# ---------- STATS ROW ----------
stats = get_stats(st.session_state.log_df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Scanned", stats["total"])
col2.metric("Damaged", stats["damaged"])
col3.metric("Damage Rate", f"{stats['damage_rate']}%")
col4.metric("Unique Sizes Seen", len(stats["by_size"]))

# ---------- DOWNLOAD BUTTON ----------
csv_data = st.session_state.log_df.to_csv(index=False).encode("utf-8")
dl_col, clear_col = st.columns([3, 1])
with dl_col:
    st.download_button(
        label="⬇️ Download Inventory Log (CSV)",
        data=csv_data,
        file_name="inventory_log.csv",
        mime="text/csv",
    )
with clear_col:
    if st.button("🗑️ Clear Log (start fresh)"):
        st.session_state.log_df = clear_log()
        st.session_state.active_barcode = None
        st.session_state.absent_count = 0
        st.session_state.score_buffer = []
        st.rerun()

st.divider()

# ---------- WEBCAM CONTROL ----------
run = st.checkbox("▶️ Start Webcam Scanning")
debug_placeholder = st.empty()
FRAME_WINDOW = st.image([])
status_placeholder = st.empty()

if run:
    camera = cv2.VideoCapture(0)  # change index (1, 2...) if wrong camera opens

    if not camera.isOpened():
        st.error("Could not access webcam. Check camera permissions / index.")
    else:
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            # NOTE: we do NOT mirror/flip the frame - flipping breaks barcode
            # decoding. The feed will look "unmirrored" but scanning works correctly.

            # 1. Barcode scan
            barcode_text, frame = scan_barcode(frame)

            # 2. Package contour + size estimate
            contour, bbox = find_package_contour(frame)
            width_cm, height_cm = estimate_size_cm(bbox)
            size_category = categorize_size(width_cm, height_cm)

            condition, damage_score = "Unknown", None

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                crop = frame[y : y + h, x : x + w]

                if crop.size > 0:
                    # Get this frame's raw score, then smooth over recent frames
                    _, raw_score = detect_damage(crop, model=model)
                    st.session_state.score_buffer.append(raw_score)
                    st.session_state.score_buffer = st.session_state.score_buffer[-SCORE_BUFFER_SIZE:]
                    damage_score = sum(st.session_state.score_buffer) / len(st.session_state.score_buffer)
                    condition = "Damaged" if damage_score > 0.5 else "OK"

                    label_color = (0, 0, 255) if condition == "Damaged" else (0, 200, 0)
                    label_y = y - 10 if y - 10 > 15 else y + h + 20  # keep label on-screen
                    cv2.putText(
                        frame,
                        f"{condition} ({size_category}) {damage_score:.2f}",
                        (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        label_color,
                        2,
                    )

            FRAME_WINDOW.image(frame, channels="BGR")

            # --- DEBUG LINE: shows exactly what's being detected, every frame ---
            debug_placeholder.code(
                f"Barcode detected: {barcode_text}\n"
                f"Active barcode (already logged): {st.session_state.active_barcode}\n"
                f"Absent frame count: {st.session_state.absent_count}\n"
                f"Package box (bbox) found: {bbox}\n"
                f"Size category: {size_category}\n"
                f"Condition: {condition} (smoothed score: {damage_score})"
            )

            # 3. Log ONCE per physical box shown - not once per timer tick
            if barcode_text:
                st.session_state.absent_count = 0

                if barcode_text != st.session_state.active_barcode:
                    # A genuinely new box: either a different barcode, or the
                    # same barcode reappearing after actually being removed.
                    zone = assign_zone(size_category, condition)
                    entry = make_entry(
                        barcode=barcode_text,
                        width_cm=width_cm,
                        height_cm=height_cm,
                        size_category=size_category,
                        condition=condition,
                        damage_score=damage_score,
                        zone=zone,
                    )
                    st.session_state.log_df = append_entry(st.session_state.log_df, entry)
                    st.session_state.active_barcode = barcode_text

                    status_placeholder.success(
                        f"✅ Logged: {barcode_text} | {size_category} | {condition} | → {zone}"
                    )
            else:
                st.session_state.absent_count += 1
                if st.session_state.absent_count > ABSENT_FRAMES_THRESHOLD:
                    # Box has genuinely been taken away - ready to log a new one next time
                    st.session_state.active_barcode = None
                    st.session_state.score_buffer = []

            time.sleep(0.03)  # ~30fps cap, keeps loop from pegging CPU

        camera.release()

st.divider()

# ---------- LIVE LOG TABLE ----------
st.subheader("📋 Inventory Log")
st.dataframe(st.session_state.log_df, use_container_width=True, height=350)

if not st.session_state.log_df.empty:
    st.subheader("📊 Size Distribution")
    st.bar_chart(st.session_state.log_df["size_category"].value_counts())