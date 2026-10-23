import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

HERE = Path(__file__).parent
st.title("Plant Disease Detection")

@st.cache_resource
def load():
    model = keras.models.load_model(HERE / "plant_model.keras")
    names = json.load(open(HERE / "class_names.json"))
    return model, names

model, class_names = load()
file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img)
    x = np.array(img.resize((224, 224)), dtype="float32")[None]
    probs = model.predict(x)[0]
    for i in probs.argsort()[::-1][:3]:
        st.write(f"**{class_names[i].replace('_', ' ')}**: {probs[i] * 100:.1f}%")