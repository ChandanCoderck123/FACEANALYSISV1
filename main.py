# main.py: main controller of the whole project

# Asking the user if they want to upload or capture three images (Center, Left, Right)
# For each image:
    # Prepares and resizes it
    # If it's the Center view, it guesses age and gender
    # Detects facial points (like eyes, nose, lips)
    # Highlights those points on the image
    # Cuts out parts of the face (called ROIs) like cheeks, lips, etc
    # Analyzes each part for things like oiliness, dryness, wrinkles, etc
# Finally, saves everything in a JSON report and image folder

import os
import json
import cv2
import numpy as np

# Local module imports
from capture import capture_image                        # For capturing image using webcam
from preprocessing import preprocess_image               # For reading and resizing the image
from detection import detect_face_landmarks              # For detecting facial landmarks using MediaPipe
from roi_extraction import extract_rois                  # For extracting facial ROIs from landmarks
from roi_analysis import (                               # Import all region-specific analysis functions
    analyze_forehead_roi, analyze_cheek_roi, analyze_nose_roi,
    analyze_lips_roi, analyze_eye_roi
)
from visualization import draw_landmarks                 # For drawing and saving landmarks on image
from age_gender import estimate_age_gender               # For estimating age and gender using models

# Constants
UPLOAD_FOLDER, OUTPUT_FOLDER = 'uploads', 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
REPORT_FILE = os.path.join(OUTPUT_FOLDER, 'report.json')

# JSON serializer to handle NumPy data types (e.g., np.float32)
def convert(o):
    if isinstance(o, (np.generic,)):
        return o.item()  # Convert NumPy scalar to native Python scalar
    raise TypeError

def analyze_images(images):
    """
    images: dict with keys 'Center', 'Left', 'Right', values are image file paths or None
    Returns a report dictionary with all analyses.
    """
    report = {}

    for view, fp in images.items():
        if not fp:
            continue  # Skip if no image available

        img = preprocess_image(fp)  # Resize and convert image to RGB
        vr = {}  # Dictionary for this view's results

        if view == "Center":
            vr["Age/Gender"] = estimate_age_gender(img)  # Estimate age and gender for front-facing image

        lms = detect_face_landmarks(img)  # Get list of (x, y) facial landmarks
        if not lms:
            print(f"No face in {view}")  # Notify if no face was detected
            continue

        draw_landmarks(img, lms, view)  # Annotate and save landmarks on face

        rois = extract_rois(img, lms)  # Extract ROIs (e.g., forehead, lips) based on landmarks

        # Define valid ROIs for each view
        if view == "Left":
            valid_regions = {"forehead", "lips", "nose", "left_eye", "left_cheek"}
        elif view == "Right":
            valid_regions = {"forehead", "lips", "nose", "right_eye", "right_cheek"}
        else:  # Center
            valid_regions = set(rois.keys())  # Analyze everything

        for r, roi in rois.items():
            if r not in valid_regions:
                continue  # Skip non-relevant regions

            out = os.path.join(OUTPUT_FOLDER, f"{view.lower()}_{r}.jpg")
            cv2.imwrite(out, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            normalized_r = r.replace("left_", "").replace("right_", "")
            fn = f"analyze_{normalized_r}_roi"

            if fn in globals():
                vr[r] = globals()[fn](roi)
            else:
                print(f"No analysis function for region: {r}")
                vr[r] = {"error": f"No analysis function defined for region: {r}"}

        report[view] = vr  # Add results for this view to the report

    return report

if __name__ == "__main__":
    images = {v: None for v in ["Center", "Left", "Right"]}

    for view in images:
        choice = input(f"{view} - (1) Upload (2) Capture (3) Skip: ")
        if choice == '1':
            fn = input("Filename inside uploads/: ").strip()
            path = os.path.join(UPLOAD_FOLDER, fn)
            if os.path.exists(path):
                images[view] = path
            else:
                print(f"File {path} not found. Skipping {view}.")
        elif choice == '2':
            images[view] = capture_image(view)
        else:
            print(f"Skipping {view}.")

    report = analyze_images(images)

    # Save the final report to a JSON file
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=convert)

    print(f"Saved report at {REPORT_FILE}")
