import os
import io
import uuid
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the YOLOv8 model once globally
# It is recommended to download your model weights and place them in the same directory.
try:
    print("Loading YOLOv8 model...")
    model = YOLO("model5.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# HTML and JavaScript for the frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 Object Detection</title>
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
        <h1 class="text-3xl font-bold mb-6 text-gray-800">YOLOv8 Object Detection API</h1>
        <p class="mb-4 text-gray-600">Upload an image below to detect objects. The results will be displayed as a JSON response.</p>
        
        <div class="flex flex-col items-center space-y-4">
            <div class="w-full flex justify-center">
                <label for="file-upload" class="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg">
                    Choose Image
                </label>
                <input id="file-upload" type="file" accept="image/*" class="hidden">
            </div>
            <img id="imagePreview" class="rounded-lg shadow-md" style="display: none;">
            <button id="detectButton" class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg" style="display: none;">
                Detect Objects
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
                // The URL is the same server, just a different endpoint
                const response = await fetch('/api/predict', {
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

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint to perform YOLOv8 object detection on an uploaded image.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Service is unavailable."}), 503

    # Check if the 'image' file is present in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    # Check if the file is empty
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Read the image from the file stream and convert it to a format YOLO can use
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Perform the prediction
        results = model(image)

        # Process the results into a JSON-serializable format
        processed_results = []
        for result in results:
            for box in result.boxes:
                # Extract and serialize the data
                processed_results.append({
                    'class_id': int(box.cls),
                    'class_name': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bounding_box': box.xyxy[0].tolist()
                })
        
        return jsonify(processed_results)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == '__main__':
    # When deploying on Render, the 'PORT' environment variable is automatically set.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
