import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown  # add this import

# Add this function and call at the very beginning:
MODEL_PATH = "osteoporosis_model.h5"
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?id=14ZMETk2adNeTztXkHeiZoRWGLFi-5Y5e"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model file. This may take a minute...")
        gdown.download(GOOGLE_DRIVE_LINK, MODEL_PATH, quiet=False)
    else:
        st.info("Model file found locally.")

download_model()

# Set page title and layout
st.set_page_config(page_title="Osteoporosis X-ray Classification", layout="centered")
st.title("🦴 Osteoporosis Multiclass Classification")

# Class labels
class_names = ['Normal', 'Osteopenia', 'Osteoporosis']

# Load model once and cache it
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Prediction function
def predict_image(image):
    image = image.resize((224, 224))  # Resize as per your model input
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0  # Normalize if model trained on normalized images
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    return class_names[predicted_index], confidence

# Allowed file extensions (lowercase)
allowed_extensions = ['.jpg', '.jpeg', '.png']

# File uploader without type restriction
uploaded_file = st.file_uploader("Upload an MRI scan image...", type=None)

if uploaded_file is not None:
    # Get file extension in lowercase
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        st.error(f"❌ Unsupported file extension '{ext}'. Please upload a JPG, JPEG, or PNG image.")
    else:
        # Open and display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run prediction
        st.write("⏳ Classifying...")
        label, confidence = predict_image(image)

        st.success(f"✅ Prediction: **{label.upper()}**")
        st.info(f"🔍 Confidence Score: **{confidence}%**")
