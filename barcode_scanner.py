"""
barcode_scanner.py
-------------------
Reads barcodes/QR codes from a video frame using pyzbar.
No training needed here - barcode decoding is a deterministic algorithm.
"""

from pyzbar.pyzbar import decode
import cv2


def scan_barcode(frame):
    """
    Look for a barcode in the given frame.
    Returns (barcode_text, annotated_frame).
    barcode_text is None if nothing was found.
    """
    decoded_objects = decode(frame)
    barcode_text = None

    for obj in decoded_objects:
        barcode_text = obj.data.decode("utf-8")

        # Draw a box around the detected barcode + write its value on screen
        points = obj.polygon
        if len(points) == 4:
            pts = [(p.x, p.y) for p in points]
            for i in range(4):
                cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)

        x, y, w, h = obj.rect
        cv2.putText(
            frame,
            barcode_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return barcode_text, frame
