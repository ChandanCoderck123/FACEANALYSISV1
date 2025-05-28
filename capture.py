# Take a photo from your webcam:
    # Open webcam
    # Press Space to capture the photo
    # Press ESC to cancel
    # Saves the photo into the uploads/ folder

# Import required libraries
import cv2  # OpenCV for handling camera input and image capture
import os  # For interacting with the operating system (creating folders, file paths)
import logging  # Standard logging module for debug and error messages

# Create a logger for this module
logger = logging.getLogger(__name__)

# Folder where captured images will be stored
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Creating the folder 

# Function to capture an image from webcam and save it with a name based on the view (e.g., 'front', 'left')
def capture_image(view_name: str) -> str:
    # Try initializing the webcam using the Windows MSMF backend
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
    # If that fails, try the default backend (platform-dependent fallback)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Try again without specifying backend

    # If still unsuccessful, log an error and return None
    if not cap.isOpened():
        logger.error(f"Camera cannot be opened for {view_name}")
        return None

    # Log the instructions to the user for how to capture or cancel
    logger.info(f"Press SPACE to capture {view_name}, ESC to cancel.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # If reading the frame failed, log and exit the loop
        if not ret:
            logger.error("Failed to grab frame.")
            break

        # Show the current frame in a window
        cv2.imshow(f"Capture {view_name}", frame)

        # Wait for a key press (1 ms). Capture either ESC (27) or SPACE (32)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC key to cancel
            cap.release()  # Release the webcam resource
            cv2.destroyAllWindows()  # Close all OpenCV windows
            return None
        
        elif key == 32:  # SPACE key to capture the image
            # Build the file path to save the image using the view_name (e.g., 'front_captured.jpg')
            path = os.path.join(UPLOAD_FOLDER, f"{view_name.lower()}_captured.jpg")
            
            # Save the current frame as an image file
            cv2.imwrite(path, frame)
            
            # Release the camera and close windows
            cap.release()
            cv2.destroyAllWindows()
            
            # Log the path where the image was saved
            logger.info(f"Saved capture: {path}")
            
            return path  # Return the file path of the saved image
