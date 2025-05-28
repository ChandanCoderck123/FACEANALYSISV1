# roi_analysis.py: To analyze each face part for skin issues

# For each region (forehead, cheeks, nose, lips, eyes), it checks for:
    # Oiliness
    # Dryness
    # Pigmentation
    # Redness
    # Acne
    # Wrinkles
    # Dark circles, etc
# Returns a simple report saying whether each problem is detected or not


import cv2  # OpenCV might be used elsewhere in the file (though not in this snippet)
import numpy as np  # NumPy is likely used in the full version for pixel analysis or math operations

# Helper function to construct and return analysis results in a standardized format
def build_result(value, threshold, comparison='>', label_positive='Yes', label_negative='No'):
    # Perform comparison based on the operator type
    # If comparison is '>', check if value is greater than the threshold
    # Otherwise, check if it's less than the threshold
    ok = (value > threshold) if comparison == '>' else (value < threshold)

    # Return result as a dictionary with:
    # - the actual value (rounded to 2 decimals)
    # - the threshold used
    # - the type of comparison
    # - whether the condition was met or not, labeled as 'Yes' or 'No' (customizable)
    return {
        "value": round(float(value), 2),   # Rounded value for readability
        "threshold": threshold,            # The comparison threshold
        "comparison": comparison,          # Comparison type ('>' or '<')
        "detected": label_positive if ok else label_negative  # Final detection result
    }

# Analysis functions
def analyze_forehead_roi(roi):
    result = {}
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    v_channel = hsv[:, :, 2]
    avg_brightness = np.mean(v_channel)
    result["Oiliness"] = build_result(avg_brightness, 170)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if avg_brightness < 100:
        result["Dryness"] = build_result(lap_var, 120, '<')  # more sensitive
    else:
        result["Dryness"] = build_result(lap_var, 100, '<')  # less sensitive but still allow detection

    std_dev_l = np.std(lab[:, :, 0])
    result["Pigmentation"] = build_result(std_dev_l, 12)

    # Redness detection using LAB B channel (higher = redder)
    b_channel = lab[:, :, 2]
    result["Redness"] = build_result(np.mean(b_channel), 145)

    result["Wrinkles"] = build_result(lap_var, 350)

    return result

def analyze_cheek_roi(roi):
    result = {}
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    v_channel = hsv[:, :, 2]
    avg_brightness = np.mean(v_channel)
    result["Oiliness"] = build_result(avg_brightness, 170)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if avg_brightness < 100:
        result["Dryness"] = build_result(lap_var, 120, '<')  # more sensitive
    else:
        result["Dryness"] = build_result(lap_var, 100, '<')  # less sensitive but still allow detection

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    result["Acne"] = build_result(edge_density, 0.12)

    std_dev_l = np.std(lab[:, :, 0])
    result["Pigmentation"] = build_result(std_dev_l, 12)

    b_channel = lab[:, :, 2]
    result["Redness"] = build_result(np.mean(b_channel), 145)

    return result

def analyze_nose_roi(roi):
    result = {}
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    avg_brightness = np.mean(hsv[:, :, 2])
    result["Shiny Nose"] = build_result(avg_brightness, 170)

    black_pixel_ratio = np.sum(gray < 50) / (gray.shape[0] * gray.shape[1])
    result["Blackheads"] = build_result(black_pixel_ratio, 0.1)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result["Clogged Pores"] = build_result(lap_var, 180)

    return result

def analyze_lips_roi(roi):
    result = {}
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result["Dry Lips"] = build_result(lap_var, 120, '<')

    a_channel = lab[:, :, 1]
    mean_a = np.mean(a_channel)
    result["Discoloration"] = build_result(abs(mean_a - 150), 15)

    return result

# def analyze_chin_roi(roi):
#     result = {}
#     hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
#     lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
#     gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

#     avg_brightness = np.mean(hsv[:, :, 2])
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

#     if avg_brightness > 170:
#         result["Oiliness"] = build_result(avg_brightness, 170)
#         result["Dryness"] = build_result(0, 0, '>', 'No', 'No')
#     elif avg_brightness < 100 and lap_var < 120:
#         result["Oiliness"] = build_result(0, 0, '>', 'No', 'No')
#         result["Dryness"] = build_result(lap_var, 120, '<')
#     else:
#         result["Oiliness"] = build_result(0, 0, '>', 'No', 'No')
#         result["Dryness"] = build_result(0, 0, '>', 'No', 'No')

#     edges = cv2.Canny(gray, 100, 200)
#     edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
#     result["Acne / Cysts"] = build_result(edge_density, 0.15)

#     b_channel = lab[:, :, 2]
#     result["Redness"] = build_result(np.mean(b_channel), 145)

#     return result

def analyze_eye_roi(roi):
    result = {}
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    avg_v = np.mean(hsv[:, :, 2])
    result["Dark Circles"] = build_result(avg_v, 70, '<')

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result["Wrinkles (Crow's Feet)"] = build_result(lap_var, 300)

    edge_strength = np.sum(cv2.Canny(gray, 50, 150)) / (gray.shape[0] * gray.shape[1])
    result["Puffy Eyes"] = build_result(edge_strength, 5, '<')

    texture_std = np.std(gray)
    result["Open Pores"] = build_result(texture_std, 40)

    return result