# visualization.py: To draw colored dots on the face to show detected points

# For each region (eyes, cheeks, nose), draws colored dots where those landmarks are
# Saves the annotated image in the outputs/ folder for us to see

import cv2  # OpenCV for drawing shapes and saving images
import os   # For creating folders and handling file paths
from roi_extraction import ROI_LANDMARKS  # Import landmark mappings for each face region

# Define the folder where output images with drawn landmarks will be saved
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Create the folder if it doesn’t already exist

# Define a color for each facial region (in BGR format, used by OpenCV)
ZONE_COLORS = {
    'forehead': (255, 0, 0),       # Blue
    'left_eye': (0, 255, 0),       # Green
    'right_eye': (0, 255, 0),      # Green
    'nose': (0, 165, 255),         # Orange
    'lips': (255, 0, 255),         # Magenta
    'left_cheek': (128, 0, 128),   # Purple
    'right_cheek': (128, 0, 128)   # Purple
}

# Function to draw facial landmarks by region onto an image
def draw_landmarks(image, landmarks, view_name):
    img = image.copy()  # Work on a copy of the image so original remains unchanged

    # Iterate over each defined zone and its associated color
    for zone, color in ZONE_COLORS.items():
        # Get the landmark indices for the current zone
        for idx in ROI_LANDMARKS.get(zone, []):
            # Ensure the index is within the landmark list bounds
            if idx < len(landmarks):
                # Draw a small filled circle at the landmark position
                cv2.circle(img, landmarks[idx], 2, color, -1)

    # Create the full path for saving the annotated image
    out = os.path.join(OUTPUT_FOLDER, f"{view_name.lower()}_landmarks.jpg")

    # Save the image (convert RGB to BGR for OpenCV compatibility)
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Notify the user where the output is saved
    print(f"Saved annotated image: {out}")
