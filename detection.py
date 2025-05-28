# finding the important facial points from an image:
    # Finds landmarks like where the eyes, nose, or lips are on the face
    # Returns a list of points on the image where those parts are located

# Import the MediaPipe library's face mesh module
import mediapipe as mp

# Access the FaceMesh module from mediapipe's solutions
mp_face_mesh = mp.solutions.face_mesh

# Create a reusable FaceMesh object (to avoid reinitializing it on every call)
# - static_image_mode=True: assumes the input is a static image (not a video stream)
# - max_num_faces=1: limits detection to a single face for performance and simplicity
_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Define the function to detect face landmarks from an RGB image
def detect_face_landmarks(image_rgb):
    """
    Input: RGB image.
    Output: List of (x, y) pixel coordinates of face landmarks or None if no face found.
    """
    # Run the face mesh detector on the input image
    results = _face_mesh.process(image_rgb)

    # If no face landmarks are detected, return None
    if not results.multi_face_landmarks:
        return None

    # Get image dimensions
    h, w, _ = image_rgb.shape

    # Convert normalized landmark coordinates (0 to 1) to pixel coordinates
    return [
        (int(lm.x * w), int(lm.y * h))  # Scale x and y to image width and height
        for lm in results.multi_face_landmarks[0].landmark  # Use landmarks of the first detected face
    ]
