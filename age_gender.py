# Guessing age and gender:

# Uses two models:
    # One for age (returns an approximate age)
    # One for gender (tells Male or Female)
# Also sorts age into a group like “21–23”
# If anything fails, it returns “Unknown”

# Import required libraries
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations and array handling
import logging  # Standard Python logging library for tracking errors/info
from typing import Dict, Union  # For type hinting dictionaries with multiple value types
from deepface import DeepFace  # DeepFace for age estimation
from insightface.app import FaceAnalysis  # InsightFace for gender detection

# Setup logging system
logger = logging.getLogger(__name__)  # Create a logger specific to this module
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO for visibility

# Initialize InsightFace's face analysis model once (to avoid repeated heavy loading)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Load pre-trained InsightFace model with CPU backend
face_app.prepare(ctx_id=0, det_size=(640, 640))  # Prepare the model with context ID and detection size

# Function to return age bucket as a string in 3-year intervals (e.g., 18–20, 21–23, ...)
def get_age_range(age: int) -> str:
    if age < 18:
        return "Below 18"  # Handle underage separately
    for start in range(18, 80, 3):  # Iterate over age buckets from 18 to 80 in steps of 3
        end = start + 2
        if start <= age <= end:  # Check if the age falls within the current bucket
            return f"{start}-{end}"  # Return the matched age bucket
    return "80+"  # Handle senior age category separately

# Main function to estimate age and gender from an RGB image
def estimate_age_gender(image_rgb: np.ndarray) -> Dict[str, Union[str, float]]:
    """
    Estimate age (by DeepFace) and gender (by InsightFace) using an RGB image array.
    """
    try:
        # Age Estimation using DeepFace 
        df_result = DeepFace.analyze(
            img_path=image_rgb,  # Provide image as an RGB NumPy array
            actions=["age"],  # Right now only interested in age estimation
            enforce_detection=False  # Skiping exception if face not detected (for safety)
        )
        age = round(df_result[0]["age"], 1)  # Get estimated age (rounded to 1 decimal place)
        age_range = get_age_range(int(age))  # Convert age into an age range bucket

        # Gender Detection using InsightFace 
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert image to BGR (required by InsightFace)
        faces = face_app.get(image_bgr)  # Detect faces using InsightFace
        if not faces:
            logger.warning("No face detected for gender detection.")  # Log a warning if no faces found
            gender = "Unknown"  # Default to unknown
        else:
            # If multiple faces are found, select the one with the largest bounding box (most likely main face)
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            gender = "Male" if face.gender == 1 else "Female"  # Decode gender from model output

        # Return structured results as a dictionary
        return {
            "Age": age,  # Numeric age
            "Age Range": age_range,  # Bucketed age range
            "Gender": gender  # Gender string
        }

    except Exception:
        # Catch-all block to log and handle any runtime errors in processing
        logger.error("Error in age/gender estimation", exc_info=True)
        return {
            "Age": "Unknown",
            "Age Range": "Unknown",
            "Gender": "Unknown"
        }
