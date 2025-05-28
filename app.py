from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
from main import analyze_images  # Import your core image analysis function

app = Flask(__name__)
CORS(app) 

# Configure folders and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    """
    API endpoint to upload a single face image ('center'),
    analyze it, save report.json, and return JSON response.
    """
    images = {}

    # Handle only the Center image
    file = request.files.get('center')
    if file and allowed_file(file.filename):
        filename = secure_filename(f"center_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        images['Center'] = filepath
    else:
        return jsonify({"error": "No valid image file uploaded or incorrect file format"}), 400

    # Run analysis pipeline on uploaded image
    try:
        report = analyze_images(images)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Save the report JSON file to disk
    report_path = os.path.join(app.config['OUTPUT_FOLDER'], 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Return the analysis report JSON inline as response
    return jsonify(report)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5006)
