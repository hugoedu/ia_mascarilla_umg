import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Configuración de la página
st.set_page_config(page_title="Detector de Mascarillas", layout="wide")

# Cargar el modelo con manejo de errores
@st.cache_resource
def cargar_modelo():
    try:
        return tf.keras.models.load_model("modelo_mascarilla.h5")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo = cargar_modelo()

# Interfaz
st.title("Detector de Uso de Mascarilla")
st.write("Sube una imagen para verificar si la persona usa mascarilla.")

# Subida de imagen
imagen_subida = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida and modelo:
    try:
        imagen = Image.open(imagen_subida)
        st.image(imagen, caption="Imagen cargada", use_column_width=True)

        # Preprocesamiento
        imagen = imagen.resize((150, 150))
        imagen_array = np.array(imagen)
        
        if len(imagen_array.shape) == 4:  # Si tiene canal alpha (RGBA)
            imagen_array = cv2.cvtColor(imagen_array, cv2.COLOR_RGBA2RGB)
        elif len(imagen_array.shape) == 2:  # Si es escala de grises
            imagen_array = cv2.cvtColor(imagen_array, cv2.COLOR_GRAY2RGB)
            
        imagen_array = imagen_array / 255.0
        imagen_array = np.expand_dims(imagen_array, axis=0)

        # Predicción
        prediccion = modelo.predict(imagen_array)[0][0]
        if prediccion > 0.5:
            st.error("❌ La persona NO está usando mascarilla.")
        else:
            st.success("✅ La persona SÍ está usando mascarilla.")
            
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")