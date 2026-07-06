# Automated Warehouse Inbound Scanner

Real-time webcam-based system that simulates automated inbound inspection in a
warehouse: scans a barcode, estimates package size, detects damage, decides a
storage zone, and logs everything to an inventory file you can download.

## What it does
1. **Live webcam feed** (OpenCV) — no image uploads, works off your actual camera.
2. **Barcode/QR scanning** (`pyzbar`) — identifies the SKU.
3. **Size estimation** (classical CV contour detection) — estimates width/height in cm.
4. **Damage detection** — a trained ML classifier if you've trained one on your own
   photos (`train_damage_model.py`), otherwise falls back automatically to a
   rule-based classical CV heuristic. Either way it works out of the box.
5. **Zone assignment** — simple decision logic: damaged → return queue,
   otherwise routed by size (small/medium/large → Zone A/B/C).
6. **Inventory logging** (`pandas`) — every scan is appended to
   `logs/inventory_log.csv` and persists across restarts.
7. **Live dashboard** (Streamlit) — stats, table, size-distribution chart, and a
   **Download CSV** button.

## Setup

```bash
pip install -r requirements.txt

# On Linux, pyzbar needs the zbar system library:
sudo apt-get install libzbar0        # Debian/Ubuntu
# Mac:
brew install zbar
```

Run the app:
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints, tick **"Start Webcam Scanning"**, and
hold a barcoded package up to your webcam.

## Calibrating size estimation (important, takes 2 minutes)

`damage_detector.py` converts pixel measurements to real cm using a constant,
`PIXELS_PER_CM`. To calibrate it for your own webcam/setup:

1. Print or find an object of known width (e.g. an A4 sheet is 21cm wide).
2. Place it at the distance you'll actually scan packages from.
3. Run the app, note the bounding box width in pixels (you can temporarily
   print `bbox` in `app.py` or just eyeball it against the frame).
4. `PIXELS_PER_CM = pixel_width / 21`
5. Update the constant at the top of `damage_detector.py`.

If you skip this, size categories will still work relatively (bigger boxes get
categorized as bigger), but the actual cm numbers won't be accurate.

## Training the damage classifier on real data

Out of the box, damage detection uses a classical CV heuristic (edge
irregularity + dark-patch detection) — no training required, works immediately.

To upgrade to a genuinely trained ML model:

1. Collect photos with your webcam or phone:
   - Pristine/normal packages → `data/ok/`
   - Damaged packages (dented, torn, crushed, wet) → `data/damaged/`
   - Aim for 40-50+ images per class minimum; more and more varied is better.
     Vary lighting, angle, and box type so the model generalizes.
2. Train:
   ```bash
   python train_damage_model.py
   ```
   This prints accuracy on a held-out test set and saves
   `models/damage_classifier.pkl`.
3. Restart the Streamlit app — it automatically detects and uses the trained
   model instead of the heuristic (you'll see a green confirmation banner).

You can keep adding more labeled photos and re-running `train_damage_model.py`
any time to improve accuracy — this is your "training on real data" story for
interviews: data collection → labeling → train/test split → accuracy metric →
iteration.

## Project structure
```
warehouse_scanner/
├── app.py                  # Streamlit app (webcam loop, UI, logging trigger)
├── barcode_scanner.py       # pyzbar barcode reading
├── damage_detector.py       # contour-based size estimation + damage detection
├── train_damage_model.py    # trains RandomForest classifier on your photos
├── inventory.py              # pandas-based logging + stats
├── data/
│   ├── ok/                   # your labeled "pristine" photos go here
│   └── damaged/               # your labeled "damaged" photos go here
├── models/
│   └── damage_classifier.pkl  # created after you run train_damage_model.py
├── logs/
│   └── inventory_log.csv      # persistent scan log
└── requirements.txt
```

## Talking points for your CV / interview
- "Built a real-time computer vision pipeline (OpenCV) that scans barcodes,
  estimates package dimensions, and classifies damage from live webcam
  footage."
- "Trained a RandomForest classifier on a self-collected, labeled image
  dataset to detect package damage, achieving [X]% accuracy on a held-out
  test set — with a classical CV fallback ensuring the system degrades
  gracefully with no data."
- "Automated the inbound decision layer: damaged packages are auto-routed to
  a return queue, while good packages are assigned a storage zone based on
  estimated size — logged via a pandas-backed inventory system with
  exportable records."
