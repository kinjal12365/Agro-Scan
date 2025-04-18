from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import ollama

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained plant disease model
model = load_model('models/plant.h5')

# Class labels for predictions
class_names = [
    'Pepper bell Bacterial_spot', 'Pepper bell healthy',
    'Potato Early blight', 'Potato Late blight', 'Potato healthy',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
    'Tomato YellowLeaf Curl Virus', 'Tomato Tomato mosaicvirus',
    'Tomato healthy'
]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No image uploaded", 400

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = load_img(filepath, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_class, image_path=filepath)

@app.route('/know_more', methods=['POST'])
def know_more():
    disease = request.form.get('disease', '')
    if not disease:
        return "No disease name provided", 400

    prompt = f"Give a brief treatment and solution for the plant disease called '{disease}' in simple terms."

    try:
        response = ollama.chat(
            model='deepseek-r1:1.5b',
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response['message']['content']
    except Exception as e:
        summary = "Sorry, something went wrong while fetching the solution."

    return render_template('know_more.html', disease=disease, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
