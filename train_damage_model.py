"""
train_damage_model.py
----------------------
Trains a real machine learning classifier to detect package damage,
using photos YOU collect and label yourself. This is the "train on real
data" part of the project.

HOW TO USE:
1. Take photos of packages with your webcam/phone.
   - Pristine boxes -> save into data/ok/
   - Damaged boxes (dented, torn, wet, crushed) -> save into data/damaged/
   - Aim for at least 40-50 images per class to start; more is better.
2. Run: python train_damage_model.py
3. It will print accuracy on a held-out test set and save the trained
   model to models/damage_classifier.pkl
4. app.py will automatically pick up and use this trained model once it exists.
"""

import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from damage_detector import extract_features

DATA_DIR = "data"
MODEL_PATH = "models/damage_classifier.pkl"


def load_dataset():
    X, y = [], []
    for label, folder in [("ok", "ok"), ("damaged", "damaged")]:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"Skipping unreadable file: {path}")
                continue
            features = extract_features(img)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    if len(X) == 0:
        print("No images found. Add photos to data/ok/ and data/damaged/ first.")
        return

    print(f"Loaded {len(X)} images total ({np.sum(y=='ok')} ok, {np.sum(y=='damaged')} damaged)")

    if len(X) < 20:
        print("WARNING: Very small dataset. Model will likely overfit / perform poorly.")
        print("Aim for at least 40-50 images per class before trusting this model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc*100:.1f}%")
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print("app.py will now automatically use this trained model instead of the classical heuristic.")


if __name__ == "__main__":
    main()
