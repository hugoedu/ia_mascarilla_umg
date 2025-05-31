
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_mascarilla.h5")

modelo = cargar_modelo()

# Título de la aplicación
st.title("Detector de Uso de Mascarilla")
st.write("Sube una imagen y el sistema te dirá si la persona está usando mascarilla.")

# Subida de imagen
imagen_subida = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento
    imagen = imagen.resize((150, 150))
    imagen_array = np.array(imagen)
    if imagen_array.shape[-1] == 4:
        imagen_array = cv2.cvtColor(imagen_array, cv2.COLOR_RGBA2RGB)
    imagen_array = imagen_array / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Predicción
    prediccion = modelo.predict(imagen_array)[0][0]
    if prediccion > 0.5:
        st.error("La persona NO está usando mascarilla.")
    else:
        st.success("La persona SÍ está usando mascarilla.")
