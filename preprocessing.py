# preprocessing.py

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations

def preprocess_image(filepath: str):
    """
    Load an image from disk, convert to RGB, resize to 512x512,
    then normalize lighting & color:
      1. Automatic white-balance (SimpleWB)
      2. CLAHE on the V channel of HSV
      3. Specular highlight removal via inpainting of very bright pixels
    """
    # 1. Load & basic format conversion
    image_bgr = cv2.imread(filepath)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read {filepath}")
    # Convert from BGR (OpenCV default) to RGB for our downstream models
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 2. Resize to model’s expected input size (512×512)
    image = cv2.resize(image, (512, 512))

    # 3. Automatic white-balance with fallback
    tmp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        # try the contrib-based SimpleWB
        wb = cv2.xphoto.createSimpleWB()
        balanced_bgr = wb.balanceWhite(tmp)
    except (AttributeError, NameError):
        # fallback to a gray-world white-balance
        b, g, r = cv2.split(tmp.astype(np.float32))
        avgB, avgG, avgR = np.mean(b), np.mean(g), np.mean(r)
        grayAvg = (avgB + avgG + avgR) / 3
        b *= (grayAvg / avgB)
        g *= (grayAvg / avgG)
        r *= (grayAvg / avgR)
        balanced_bgr = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)

    # back to RGB for the rest of the pipeline
    image = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2RGB)


    # 4. Contrast Enhancement via CLAHE on the V (value) channel of HSV
    #    - Flattens out shadows & hot spots for more consistent brightness.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE only on the Value channel
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 5. Specular‐highlight removal:
    #    - Find very bright pixels (V > 240) that represent glare/oil shine
    #    - Inpaint them (median‐based fill) so they don’t bias oiliness metrics
    hsv2 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv2[:, :, 2]
    # Create a binary mask where highlights are “white” (255) and everything else “black” (0)
    _, mask = cv2.threshold(v_channel, 240, 255, cv2.THRESH_BINARY)
    # Inpaint expects a single‐channel mask and a BGR image
    bgr_for_inpaint = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr_for_inpaint, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    # Convert back to RGB for consistency
    image = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    return image
