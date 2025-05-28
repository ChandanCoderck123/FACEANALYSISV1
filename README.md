Overall Workflow (Controlled by main.py): python --version
Python 3.13.3

Step 1: Image Capture or Upload
    File: capture.py
    Role: If the user chooses to take a picture, main.py calls capture_image() to open the webcam and save the image in uploads/

Step 2: Image Preprocessing
    File: preprocessing.py
    Role: main.py sends each image file path to preprocess_image(), which loads and resizes the image, and converts it to RGB format

Step 3: Age & Gender Estimation
    File: age_gender.py
    Role: For the Center view only, main.py passes the RGB image to estimate_age_gender() to get age and gender predictions

Step 4: Landmark Detection
    File: detection.py
    Role: main.py sends the RGB image to detect_face_landmarks() to get a list of key facial points like eyes, nose, and lips

Step 5: Landmark Visualization
    File: visualization.py
    Role: main.py calls draw_landmarks() with the image and detected facial points to draw colored dots and save a copy in outputs/

Step 6: ROI Extraction
    File: roi_extraction.py
    Role: Using the image and landmark points, main.py calls extract_rois() to crop regions like:
    forehead
    cheeks
    lips
    nose
    eyes
These small cropped images (ROIs) are saved and sent for analysis

Step 7: ROI Analysis
    File: roi_analysis.py
    Role: For each ROI, main.py dynamically calls functions like:
    analyze_forehead_roi(roi), analyze_nose_roi(roi) etc to check for issues like oiliness, dryness, acne, etc

Step 8: Final Report
    File: main.py
    Role: Collects all results (age/gender + skin issues per region) and saves them in a structured JSON file (outputs/report.json)




### Visual Summary:

main.py
├── capture.py           → capture_image()
├── preprocessing.py     → preprocess_image()
├── age_gender.py        → estimate_age_gender()
├── detection.py         → detect_face_landmarks()
├── visualization.py     → draw_landmarks()
├── roi_extraction.py    → extract_rois()
├── roi_analysis.py      → analyze_<region>_roi()
└── outputs/
    ├── *.jpg            → saved ROI crops and landmark images
    └── report.json      → final result
