from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import requests
import os

# link cloud storage model
url = "https://storage.googleapis.com/express-storage-1/makara.h5"
local_filename = "makara.h5"

response = requests.get(url)
with open(local_filename, "wb") as file:
    file.write(response.content)

model = load_model("makara.h5")

app = Flask(__name__)

def preprocess_image(file):
    # Load image and resize
    img = Image.open(file).resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'})

        # Check if the file is allowed
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'status': 'error', 'message': 'Invalid file type'})

        # Preprocess the image
        img_array = preprocess_image(file)

        # Make prediction
        classes = model.predict(img_array, batch_size=8)

        # Process the prediction result and get the class label
        class_labels = ["Bika Ambon", "Kerak Telor", "Molen", "Nasi Goreng", "Papeda Maluku", "Sate Padang", "Seblak"]
        class_index = np.argmax(classes)
        class_label = class_labels[class_index]

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting"
            },
            "data": {
                "name": class_label,
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image."
            },
            "data": None,
        }), 400


if __name__ == '__main__':
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)))
