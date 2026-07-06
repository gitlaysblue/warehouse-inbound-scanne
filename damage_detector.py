"""
damage_detector.py
-------------------
Two jobs live here:

1. SIZE ESTIMATION (classical CV, no training needed)
   - Find the package's outline (contour) in the frame
   - Convert its pixel width/height into real-world cm using a calibration
     constant (pixels-per-cm), which you set once for your camera setup.

2. DAMAGE DETECTION (two modes)
   - "classical" mode: rule-based heuristics (edge irregularity + dark/wet
     patch detection). No training needed, works out of the box.
   - "trained" mode: loads a scikit-learn model (trained by train_damage_model.py
     on YOUR real photos) and uses it to classify the package as ok/damaged.
     Falls back to classical mode automatically if no trained model exists yet.
"""

import cv2
import numpy as np
import os
import joblib

MODEL_PATH = "models/damage_classifier.pkl"

# --- CALIBRATION ---
# Set this by placing an object of KNOWN width (e.g. an A4 sheet = 21cm)
# at the distance you'll actually use, measuring its pixel width in the
# frame, and computing PIXELS_PER_CM = pixel_width / real_width_cm
PIXELS_PER_CM = 15.0  # <-- adjust this after calibrating your webcam (see README)


def find_package_contour(frame):
    """
    Detect the largest rectangular-ish object in the frame (assumed to be
    the package). Returns (contour, bounding_box) or (None, None) if nothing
    found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Assume the package is the largest contour in view
    largest = max(contours, key=cv2.contourArea)

    # Ignore tiny noise contours
    if cv2.contourArea(largest) < 3000:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)
    return largest, (x, y, w, h)


def estimate_size_cm(bbox):
    """Convert a pixel bounding box into real-world width/height in cm."""
    if bbox is None:
        return None, None
    _, _, w, h = bbox
    width_cm = w / PIXELS_PER_CM
    height_cm = h / PIXELS_PER_CM
    return width_cm, height_cm


def categorize_size(width_cm, height_cm):
    if width_cm is None:
        return "Unknown"
    area = width_cm * height_cm
    if area < 15 * 15:
        return "Small"
    elif area < 30 * 30:
        return "Medium"
    else:
        return "Large"


def extract_features(image_crop):
    """
    Turn a cropped package image into a fixed-length numeric feature vector.
    Used both for the classical heuristic AND as input to the trained model,
    so the two stay consistent.

    Features:
    - edge irregularity ratio (jagged vs straight boundary)
    - dark/wet patch ratio (proxy for stains, tears, crushed regions)
    - color histogram (captures general damage-related discoloration)
    """
    resized = cv2.resize(image_crop, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 1. Edge irregularity: ratio of edge pixels to total contour perimeter area
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    # 2. Dark/wet patch ratio: proportion of unusually dark pixels
    dark_ratio = np.sum(gray < 60) / gray.size

    # 3. Color histogram (coarse, 8 bins per channel) -> captures discoloration
    hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    features = np.concatenate([[edge_ratio, dark_ratio], hist])
    return features


def classical_damage_score(image_crop):
    """
    Rule-based fallback: no training required.
    Returns a damage_score between 0 (pristine) and 1 (heavily damaged),
    plus a boolean verdict.
    """
    edge_ratio, dark_ratio = extract_features(image_crop)[:2]

    # These thresholds are heuristic starting points - tune them by testing
    # against your own sample packages.
    score = min(1.0, (edge_ratio * 3.0) + (dark_ratio * 2.0))
    is_damaged = score > 0.35
    return score, is_damaged


def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def detect_damage(image_crop, model=None):
    """
    Main entry point. Uses the trained model if available, otherwise
    falls back to the classical heuristic.
    Returns (condition_str, damage_score)
    """
    if model is not None:
        features = extract_features(image_crop).reshape(1, -1)
        prediction = model.predict(features)[0]
        # predict_proba gives a confidence-like score for "damaged" class
        proba = model.predict_proba(features)[0]
        damage_score = proba[list(model.classes_).index("damaged")] if "damaged" in model.classes_ else proba[1]
        condition = "Damaged" if prediction == "damaged" else "OK"
        return condition, damage_score

    score, is_damaged = classical_damage_score(image_crop)
    condition = "Damaged" if is_damaged else "OK"
    return condition, score


def assign_zone(size_category, condition):
    if condition == "Damaged":
        return "Return-to-Vendor Queue"
    return {
        "Small": "Zone A (Bin Storage)",
        "Medium": "Zone B (Shelf Storage)",
        "Large": "Zone C (Pallet Storage)",
        "Unknown": "Manual Review",
    }.get(size_category, "Manual Review")
