# roi_extraction.py

import cv2
import numpy as np

# Predefined landmark indices for each facial region
ROI_LANDMARKS = {
    'forehead': [10, 338, 297, 332, 284],  # Central and upper forehead points
    'left_eye': [33, 133],                 # Approximate bounding points for left eye
    'right_eye': [362, 263],               # Approximate bounding points for right eye
    'nose': [1, 6, 197, 195, 5],           # Bridge and sides of the nose
    'lips': [61, 291, 78, 308],            # Corners and center of the lips
    'left_cheek': [50, 205, 187],          # Left cheekbone area
    'right_cheek': [280, 425, 411]         # Right cheekbone area
}

def extract_rois(image: np.ndarray, landmarks: list) -> dict:
    """
    Extract skin regions exactly by:
      1. Computing adaptive padding from inter-ocular distance.
      2. Building a convex-hull mask around each region's landmarks.
      3. Cleaning up the mask with morphological operations.
      4. Cropping the masked region with adaptive padding.

    Returns a dict of region_name -> ROI image patch.
    """
    h, w, _ = image.shape

    # 1. Compute adaptive padding based on inter-ocular distance (IOD)
    #    - Average the left-eye and right-eye landmark points
    left_pts  = [landmarks[i] for i in ROI_LANDMARKS['left_eye']  if i < len(landmarks)]
    right_pts = [landmarks[i] for i in ROI_LANDMARKS['right_eye'] if i < len(landmarks)]
    if left_pts and right_pts:
        left_ctr  = np.mean(left_pts,  axis=0)
        right_ctr = np.mean(right_pts, axis=0)
        iod = np.linalg.norm(right_ctr - left_ctr)        # inter-ocular distance
        pad = int(0.05 * iod)                             # 5% of IOD as padding
    else:
        pad = 10  # fallback fixed padding if eyes not detected

    rois = {}

    for name, idxs in ROI_LANDMARKS.items():
        # 2. Gather the landmark points for this region
        pts = [landmarks[i] for i in idxs if i < len(landmarks)]

        # Special handling to extend forehead region upward
        if name == 'forehead' and pts:
            cx, cy = landmarks[10]
            vertical_span = max(y for _, y in pts) - min(y for _, y in pts)
            offset = int(0.3 * vertical_span)
            pts.append((cx, max(cy - offset, 0)))

        if not pts:
            continue

        pts_arr = np.array(pts, dtype=np.int32)

        # 3. Convex-Hull Masking
        hull = cv2.convexHull(pts_arr)                     # convex hull of region
        mask = np.zeros((h, w), dtype=np.uint8)            # single-channel mask
        cv2.fillConvexPoly(mask, hull, 255)                # fill hull area

        # 4. Morphological Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # remove small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes

        # 5. Crop with adaptive padding
        x, y, w_box, h_box = cv2.boundingRect(hull)        # bounding rect of hull
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w_box + pad, w)
        y2 = min(y + h_box + pad, h)

        # Apply the mask to the original image and then crop
        masked = cv2.bitwise_and(image, image, mask=mask)
        roi = masked[y1:y2, x1:x2]

        rois[name] = roi

    return rois
