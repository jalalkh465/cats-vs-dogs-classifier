from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("D:\\Github Uploads\\flask-app\\cats_vs_dogs_resnet.h5")

# Define image size
IMAGE_SIZE = (224, 224)

# Route for rendering the HTML page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        
        # Convert file to BytesIO
        img = image.load_img(BytesIO(file.read()), target_size=IMAGE_SIZE)
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        
        prediction = model.predict(img_array)[0][0]
        class_name = "Dog" if prediction > 0.5 else "Cat"
        
        return render_template('index.html', prediction=class_name, confidence=float(prediction))
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
