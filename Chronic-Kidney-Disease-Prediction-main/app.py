from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Function to handle predictions for structured data models
def predict_structured_data(values):
    try:
        values = np.asarray(values).reshape(1, -1)  # Reshape input to match model expectations
        if len(values[0]) == 8:
            model = pickle.load(open('models/diabetes.pkl', 'rb'))
        elif len(values[0]) == 26:
            model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        elif len(values[0]) == 13:
            model = pickle.load(open('models/heart.pkl', 'rb'))
        elif len(values[0]) == 18:
            model = pickle.load(open('models/kidney.pkl', 'rb'))
        elif len(values[0]) == 10:
            model = pickle.load(open('models/liver.pkl', 'rb'))
        else:
            return "Invalid number of inputs"
        
        # Return the prediction result
        return model.predict(values)[0]
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Function to handle predictions for image data models (Malaria and Pneumonia)
def predict_image(img, model_name, input_shape, grayscale=False):
    try:
        if grayscale:
            img = img.convert('L')  # Convert to grayscale for pneumonia
            img = img.resize(input_shape)  # Resize to model input size
            img = np.asarray(img).reshape(1, input_shape[0], input_shape[1], 1)
        else:
            img = img.resize(input_shape)  # Resize to model input size
            img = np.asarray(img).reshape(1, input_shape[0], input_shape[1], 3)

        img = img / 255.0  # Normalize the image data
        model = load_model(f"models/{model_name}.h5")
        pred = np.argmax(model.predict(img)[0])
        return pred
    except Exception as e:
        return f"Prediction error: {str(e)}"

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/<disease>", methods=['GET', 'POST'])
def disease_page(disease):
    template_name = f"{disease}.html"
    return render_template(template_name)

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict_structured_data(to_predict_list)
            return render_template('predict.html', pred=pred)
    except Exception as e:
        flash("Please enter valid data")
        return render_template("home.html", message=str(e))

@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        if 'image' not in request.files:
            flash("Please upload an image")
            return redirect(request.url)

        img = Image.open(request.files['image'])
        pred = predict_image(img, "malaria", (36, 36))  # Malaria model expects 36x36 RGB image
        return render_template('malaria_predict.html', pred=pred)
    except Exception as e:
        flash(f"Error in Malaria prediction: {str(e)}")
        return render_template('malaria.html')

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' not in request.files:
            flash("Please upload an image")
            return redirect(request.url)

        img = Image.open(request.files['image'])
        pred = predict_image(img, "pneumonia", (36, 36), grayscale=True)  # Pneumonia expects grayscale
        return render_template('pneumonia_predict.html', pred=pred)
    except Exception as e:
        flash(f"Error in Pneumonia prediction: {str(e)}")
        return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True)
