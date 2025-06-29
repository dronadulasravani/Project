from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained model
model = tf.keras.models.load_model(r"D:\capston project\epcd-main\my-app\api\best_model.h5")

# Define the upload folder within the static directory
#UPLOAD_FOLDER = os.path.join('static', 'uploads')
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle file uploads and predictions
@app.route('/predict', methods=['POST'])
def upload_file():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Secure and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0

        # Make prediction
        prediction = model.predict(x)
        result = 'Pneumonia' if np.argmax(prediction[0]) == 1 else 'Normal'

        # Return prediction only
        return jsonify({'prediction': result})

    else:
        return jsonify({'error': 'File type not allowed. Please upload a PNG, JPG, or JPEG image.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)
