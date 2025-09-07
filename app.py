import os
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Initialize Flask app
print("Initializing Flask app...")
app = Flask(__name__)
CORS(app)
print("Flask app initialized.")

# HTML and JavaScript for the frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>In-Browser Human Presence Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .container { max-width: 900px; margin: auto; padding: 2rem; }
        video, canvas { border: 1px solid #ddd; border-radius: 0.5rem; }
    </style>
    <!-- Import TensorFlow.js and the face-detection model -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/face-detection@1.0.1/dist/face-detection.js"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="container bg-white shadow-xl rounded-2xl p-6 text-center">
        <h1 class="text-3xl font-bold mb-6 text-gray-800">In-Browser Human Presence Detector</h1>
        <p class="mb-4 text-gray-600">This application runs entirely in your browser. No data is sent to the server.</p>
        
        <div class="flex flex-col items-center space-y-4">
            <button id="startWebcamButton" class="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg">
                Start Webcam
            </button>
            <video id="webcamVideo" class="w-full max-w-xl" autoplay playsinline muted style="display: none;"></video>
            <canvas id="webcamCanvas" class="w-full max-w-xl" style="display: none;"></canvas>
            <div id="results" class="mt-8 p-4 bg-gray-50 rounded-lg text-left overflow-x-auto text-sm text-gray-700 hidden">
                <h2 class="text-xl font-semibold mb-2 text-gray-800">Detection Results</h2>
                <pre><code id="json-code"></code></pre>
            </div>
            <div id="message" class="mt-4 text-red-600 font-medium"></div>
        </div>
    </div>

    <script>
        const webcamButton = document.getElementById('startWebcamButton');
        const webcamVideo = document.getElementById('webcamVideo');
        const webcamCanvas = document.getElementById('webcamCanvas');
        const messageDiv = document.getElementById('message');
        const resultsDiv = document.getElementById('results');
        const jsonCode = document.getElementById('json-code');
        const ctx = webcamCanvas.getContext('2d');

        let model = null;

        // Load the face detection model
        async function loadModel() {
            messageDiv.textContent = 'Loading TensorFlow.js and Face-Detection model...';
            try {
                const detector = await faceDetection.createDetector(faceDetection.SupportedModels.BlazeFace);
                model = detector;
                messageDiv.textContent = 'Model loaded successfully. Ready to start webcam.';
                webcamButton.disabled = false;
            } catch (error) {
                console.error('Failed to load model:', error);
                messageDiv.textContent = 'Failed to load model. Please check the console for details.';
            }
        }

        // Main detection loop
        async function detectFaces() {
            if (!model || webcamVideo.paused || webcamVideo.ended) {
                return;
            }

            const video = webcamVideo;
            const canvas = webcamCanvas;

            // Run detection
            const predictions = await model.estimateFaces(video);

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const detections = [];
            predictions.forEach(prediction => {
                const start = prediction.box.xMin;
                const end = prediction.box.yMin;
                const width = prediction.box.width;
                const height = prediction.box.height;

                // Draw bounding box
                ctx.beginPath();
                ctx.rect(start, end, width, height);
                ctx.lineWidth = 4;
                ctx.strokeStyle = '#34D399';
                ctx.stroke();

                detections.push({
                    box: { x: start, y: end, width, height },
                    confidence: prediction.score
                });
            });

            // Update the results display
            const final_response = {
                person_count: detections.length,
                person_detected: detections.length > 0,
                detections: detections
            };
            jsonCode.textContent = JSON.stringify(final_response, null, 2);
            resultsDiv.classList.remove('hidden');
        }

        // Start the webcam and the detection loop
        async function startWebcam() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcamVideo.srcObject = stream;
                webcamVideo.onloadedmetadata = () => {
                    webcamVideo.play();
                    webcamVideo.style.display = 'block';
                    webcamCanvas.width = webcamVideo.videoWidth;
                    webcamCanvas.height = webcamVideo.videoHeight;
                    webcamCanvas.style.display = 'block';
                    messageDiv.textContent = 'Webcam started. Detecting...';
                    setInterval(detectFaces, 100);
                };
            } catch (err) {
                console.error("Error accessing the webcam: ", err);
                messageDiv.textContent = 'Error accessing the webcam. Please ensure it is enabled.';
            }
        }

        webcamButton.addEventListener('click', startWebcam);

        loadModel();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """
    Serves the main HTML page for the web frontend.
    """
    print("Serving the main HTML page.")
    return render_template_string(HTML_CONTENT)

if __name__ == '__main__':
    print("Starting Flask application...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
