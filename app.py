import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.title("Detector de Mascarillas 😷")

# Carga el modelo
model = tf.keras.models.load_model("model.h5")

# Función para procesar la imagen
def preprocess_image(image):
    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = img.reshape(1, 150, 150, 3)
    return img

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Procesar imagen
    img_prepared = preprocess_image(image)

    # Hacer predicción
    prediction = model.predict(img_prepared)[0][0]

    if prediction >= 0.5:
        st.success("✅ Con Mascarilla")
    else:
        st.error("❌ Sin Mascarilla")
