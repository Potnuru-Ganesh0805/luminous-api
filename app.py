import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the HOG descriptor for human detection
try:
    print("Initializing HOG/SVM human detector...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("HOG/SVM detector initialized successfully.")
except Exception as e:
    print(f"Error initializing detector: {e}")
    hog = None

# The HTML and JavaScript frontend is embedded in a Python string
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightweight Human Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .container {
            max-width: 900px;
            margin: auto;
            padding: 2rem;
        }
        #imagePreview {
            max-width: 100%;
            max-height: 400px;
            margin-top: 1rem;
            border: 1px solid #ddd;
            border-radius: 0.5rem;
            display: none;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="container bg-white shadow-xl rounded-2xl p-6 text-center">
        <h1 class="text-3xl font-bold mb-6 text-gray-800">Lightweight Human Presence Detector</h1>
        <p class="mb-4 text-gray-600">Upload an image below to detect people using a basic HOG/SVM algorithm.</p>
        
        <div class="flex flex-col items-center space-y-4">
            <div class="w-full flex justify-center">
                <label for="file-upload" class="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg">
                    Choose Image
                </label>
                <input id="file-upload" type="file" accept="image/*" class="hidden">
            </div>
            <img id="imagePreview" class="rounded-lg shadow-md" style="display: none;">
            <button id="detectButton" class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg" style="display: none;">
                Detect Person
            </button>
        </div>
        
        <div id="results" class="mt-8 p-4 bg-gray-50 rounded-lg text-left overflow-x-auto text-sm text-gray-700 hidden">
            <h2 class="text-xl font-semibold mb-2 text-gray-800">Detection Results</h2>
            <pre><code id="json-code"></code></pre>
        </div>
        
        <div id="message" class="mt-4 text-red-600 font-medium"></div>
    </div>

    <script>
        const fileInput = document.getElementById('file-upload');
        const imagePreview = document.getElementById('imagePreview');
        const detectButton = document.getElementById('detectButton');
        const resultsDiv = document.getElementById('results');
        const jsonCode = document.getElementById('json-code');
        const messageDiv = document.getElementById('message');

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                    detectButton.style.display = 'block';
                    resultsDiv.classList.add('hidden');
                    messageDiv.textContent = '';
                };
                reader.readAsDataURL(file);
            }
        });

        detectButton.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) {
                messageDiv.textContent = 'Please select an image first.';
                return;
            }

            messageDiv.textContent = 'Processing... Please wait.';
            detectButton.disabled = true;

            const formData = new FormData();
            formData.append('image', file);

            try {
                // The URL for your API endpoint
                const response = await fetch('/api/detect_person', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (response.ok) {
                    jsonCode.textContent = JSON.stringify(data, null, 2);
                    resultsDiv.classList.remove('hidden');
                    messageDiv.textContent = 'Detection complete.';
                } else {
                    messageDiv.textContent = data.error || 'An error occurred during detection.';
                    resultsDiv.classList.add('hidden');
                }
            } catch (error) {
                console.error('Error:', error);
                messageDiv.textContent = 'An error occurred. Please try again.';
                resultsDiv.classList.add('hidden');
            } finally {
                detectButton.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """
    Serves the main HTML page for the web frontend.
    """
    return render_template_string(HTML_CONTENT)

@app.route('/api/detect_person', methods=['POST'])
def detect_person_api():
    """
    API endpoint to detect the presence of a person in an image.
    """
    if hog is None:
        return jsonify({"error": "Detector not loaded. Service is unavailable."}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Read the image file and convert to a format OpenCV can use
        image_bytes = image_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert PIL image to a NumPy array in BGR format for OpenCV
        cv_image = np.array(pil_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        # Perform the human detection
        (boxes, weights) = hog.detectMultiScale(cv_image, winStride=(8, 8), padding=(16, 16), scale=1.05)
        
        person_detected = False
        if len(boxes) > 0:
            person_detected = True

        # Check if boxes is a NumPy array before calling tolist()
        if not isinstance(boxes, np.ndarray):
            # If it's not an array, it's likely a tuple from an OpenCV bug
            bounding_boxes = list(boxes)
        else:
            bounding_boxes = boxes.tolist()

        return jsonify({
            "person_detected": person_detected,
            "number_of_people": len(boxes),
            "bounding_boxes": bounding_boxes
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({"error": "An error occurred during detection."}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
