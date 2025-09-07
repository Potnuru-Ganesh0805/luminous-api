import os
import io
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from google.cloud import vision_v1
from google.oauth2.service_account import Credentials

# Initialize Flask app
print("Initializing Flask app...")
app = Flask(__name__)
CORS(app)
print("Flask app initialized.")

# Initialize the Google Cloud Vision client using credentials from a string
try:
    print("Initializing Google Cloud Vision client...")
    
    # Read the JSON key from the environment variable
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_KEY environment variable not set.")
    
    # Convert the JSON string to a credentials object
    credentials = Credentials.from_service_account_info(json.loads(creds_json))
    
    # Initialize the client with the loaded credentials
    vision_client = vision_v1.ImageAnnotatorClient(credentials=credentials)
    print("Google Cloud Vision client initialized successfully.")
    
except Exception as e:
    print(f"ERROR: Exception during Vision client initialization: {e}")
    vision_client = None

# HTML and JavaScript for the frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightweight Human Presence Detector</title>
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
        <p class="mb-4 text-gray-600">Upload an image below to detect human presence using a cloud-based API.</p>
        
        <div class="flex flex-col items-center space-y-4">
            <div class="w-full flex justify-center">
                <label for="file-upload" class="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg">
                    Choose Image
                </label>
                <input id="file-upload" type="file" accept="image/*" class="hidden">
            </div>
            <img id="imagePreview" class="rounded-lg shadow-md" style="display: none;">
            <button id="detectButton" class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg" style="display: none;">
                Detect Presence
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
    API endpoint to perform human presence detection using Google Cloud Vision.
    """
    print("Received a request to /api/predict")
    if vision_client is None:
        print("ERROR: Vision client is not loaded. Returning 503.")
        return jsonify({"error": "Vision API service is unavailable."}), 503

    if 'image' not in request.files:
        print("ERROR: No image file provided in request.")
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        print("ERROR: Empty file provided.")
        return jsonify({"error": "No selected file."}), 400

    try:
        print("Reading image file from request...")
        content = image_file.read()
        image = vision_v1.Image(content=content)
        print("Image read successfully.")

        print("Performing prediction with Google Cloud Vision...")
        response = vision_client.object_localization(image=image)
        print("Prediction complete.")

        # Process the results into a JSON-serializable format
        processed_results = []
        person_count = 0
        for obj in response.localized_object_annotations:
            if obj.name.lower() == 'person':
                person_count += 1
                processed_results.append({
                    'class_name': obj.name,
                    'confidence': obj.score,
                    'bounding_box': [
                        obj.bounding_poly.normalized_vertices[0].x,
                        obj.bounding_poly.normalized_vertices[0].y,
                        obj.bounding_poly.normalized_vertices[2].x,
                        obj.bounding_poly.normalized_vertices[2].y,
                    ]
                })

        final_response = {
            'status': 'success',
            'person_detected': person_count > 0,
            'person_count': person_count,
            'detections': processed_results
        }
        print("Results processed into JSON.")
        
        return jsonify(final_response)

    except Exception as e:
        print(f"ERROR: Vision API call failed: {e}")
        return jsonify({"error": f"An error occurred during API call: {e}"}), 500

if __name__ == '__main__':
    print(f"Getting port from environment variable. PORT is: {os.environ.get('PORT', '5000')}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
