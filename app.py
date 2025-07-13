from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("C:/Users/ComaTozze/Desktop/AI/Lung_cancer/lung_cancer_cnn_model/lung_cancer_cnn_model.h5")

# Function to preprocess the image
def prepare_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.reshape(img, (1, 128, 128, 3))
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", prediction="No file uploaded.")
    
    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", prediction="No file selected.")
    
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    img = prepare_image(filepath)
    if img is None:
        return render_template("result.html", prediction="Error processing image.")

    prediction = model.predict(img)
    pred_class = np.argmax(prediction)
    result = "Lung Cancer Detected" if pred_class == 1 else "No Lung Cancer Detected"

    return render_template("result.html", prediction=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
