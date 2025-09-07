import os
from flask import Flask, render_template_string

# Initialize Flask app
app = Flask(__name__)

# The HTML and JavaScript frontend is embedded in a Python string
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightweight Human Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .container {
            max-width: 900px;
            margin: auto;
            padding: 2rem;
        }
        #image-container {
            position: relative;
            display: inline-block;
            margin-top: 1rem;
        }
        #imagePreview, #canvas-overlay {
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #ddd;
            border-radius: 0.5rem;
        }
        #canvas-overlay {
            position: absolute;
            top: 0;
            left: 0;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="container bg-white shadow-xl rounded-2xl p-6 text-center">
        <h1 class="text-3xl font-bold mb-6 text-gray-800">Lightweight Human Presence Detector</h1>
        <p class="mb-4 text-gray-600">Upload an image below to detect people directly in your browser.</p>
        
        <div class="flex flex-col items-center space-y-4">
            <div class="w-full flex justify-center">
                <label for="file-upload" class="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 shadow-lg">
                    Choose Image
                </label>
                <input id="file-upload" type="file" accept="image/*" class="hidden">
            </div>
            <div id="image-container" style="display: none;">
                <img id="imagePreview" class="rounded-lg shadow-md">
                <canvas id="canvas-overlay"></canvas>
            </div>
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
        const imageContainer = document.getElementById('image-container');
        const canvasOverlay = document.getElementById('canvas-overlay');
        const resultsDiv = document.getElementById('results');
        const jsonCode = document.getElementById('json-code');
        const messageDiv = document.getElementById('message');

        let model;
        (async () => {
            messageDiv.textContent = 'Loading TensorFlow.js model...';
            try {
                model = await cocoSsd.load();
                messageDiv.textContent = 'Model loaded. Select an image to begin.';
            } catch (error) {
                console.error("Failed to load the model:", error);
                messageDiv.textContent = 'Error: Failed to load the detection model.';
            }
        })();

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = async (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                imageContainer.style.display = 'block';
                resultsDiv.classList.add('hidden');
                messageDiv.textContent = 'Detecting objects...';

                imagePreview.onload = async () => {
                    const predictions = await model.detect(imagePreview);
                    
                    const ctx = canvasOverlay.getContext('2d');
                    canvasOverlay.width = imagePreview.offsetWidth;
                    canvasOverlay.height = imagePreview.offsetHeight;
                    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

                    const personDetections = predictions.filter(p => p.class === 'person');

                    if (personDetections.length > 0) {
                        messageDiv.textContent = `Detection complete. Found ${personDetections.length} people.`;
                        ctx.strokeStyle = 'red';
                        ctx.lineWidth = 2;
                        
                        const detectedPeople = [];
                        personDetections.forEach(p => {
                            const [x, y, width, height] = p.bbox;
                            
                            ctx.strokeRect(x, y, width, height);

                            detectedPeople.push({
                                "bounding_box": [
                                    Math.round(x),
                                    Math.round(y),
                                    Math.round(x + width),
                                    Math.round(y + height)
                                ],
                                "score": p.score,
                                "class": p.class
                            });
                        });
                        
                        jsonCode.textContent = JSON.stringify({
                            "person_detected": true,
                            "number_of_people": detectedPeople.length,
                            "detections": detectedPeople
                        }, null, 2);
                        resultsDiv.classList.remove('hidden');

                    } else {
                        messageDiv.textContent = 'No people detected.';
                        jsonCode.textContent = JSON.stringify({
                            "person_detected": false,
                            "number_of_people": 0,
                            "detections": []
                        }, null, 2);
                        resultsDiv.classList.remove('hidden');
                    }
                };
            };
            reader.readAsDataURL(file);
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
