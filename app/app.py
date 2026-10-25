import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

HERE = Path(__file__).parent
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

DISEASES = {
    "bacterial spot": {
        "type": "Bacterial",
        "symptoms": ["Small, dark, water-soaked spots on leaves that turn brown or black, often with a yellow halo",
                     "Spots may merge and leaves can drop; fruit gets rough, scabby spots"],
        "treatment": ["There is no cure, so slow the spread: remove and destroy badly infected leaves",
                      "Copper-based sprays can help reduce spread, mainly when used early",
                      "Do not handle plants when they are wet"],
        "prevention": ["Use disease-free seed and seedlings",
                       "Rotate crops (avoid tomato or pepper in the same spot for 2 to 3 years)",
                       "Water at the base, not over the leaves, and disinfect tools"],
    },
    "early blight": {
        "type": "Fungal",
        "symptoms": ["Brown spots with concentric rings (like a target), usually on older lower leaves first",
                     "Yellowing around the spots; heavily affected leaves dry up and fall"],
        "treatment": ["Remove and destroy infected lower leaves",
                      "Spray a fungicide labelled for early blight, starting at the first signs",
                      "Mulch the soil so spores do not splash up onto leaves"],
        "prevention": ["Rotate crops and clear plant debris after harvest",
                       "Give plants space and airflow, and avoid wetting leaves",
                       "Feed and water properly, because stressed plants get it more easily"],
    },
    "late blight": {
        "type": "Fungal-like (water mold)",
        "symptoms": ["Large, greasy, water-soaked dark patches on leaves and stems",
                     "White fuzzy growth under leaves in humid weather",
                     "Spreads very fast in cool, wet conditions and can destroy a crop in days"],
        "treatment": ["Act at once: remove infected plants or parts and bag them (do not compost)",
                      "Protective fungicides work best when applied before the disease spreads",
                      "Check neighbouring plants every day"],
        "prevention": ["Plant certified disease-free seed potatoes and resistant varieties",
                       "Remove leftover potatoes and volunteer plants",
                       "Avoid overhead watering and keep plants well spaced"],
    },
    "leaf mold": {
        "type": "Fungal",
        "symptoms": ["Pale green or yellow spots on the top of the leaf",
                     "Olive-green to brown velvety mold on the underside",
                     "Common in humid greenhouses"],
        "treatment": ["Improve ventilation and lower humidity",
                      "Remove affected leaves",
                      "Use a fungicide labelled for leaf mold if it spreads"],
        "prevention": ["Space and prune plants for airflow",
                       "Water the soil, not the leaves",
                       "Choose resistant varieties and clean up debris after the season"],
    },
    "septoria": {
        "type": "Fungal",
        "symptoms": ["Many small round spots with dark edges and grey or tan centres",
                     "Tiny black dots in the middle of the spots",
                     "Starts on lower leaves and moves up"],
        "treatment": ["Remove infected lower leaves",
                      "Use a fungicide labelled for Septoria leaf spot, starting early",
                      "Mulch to stop soil splashing onto leaves"],
        "prevention": ["Rotate crops and remove old plant debris",
                       "Avoid overhead watering",
                       "Keep leaves dry and plants well spaced"],
    },
    "spider mite": {
        "type": "Pest (not a disease)",
        "symptoms": ["Tiny yellow or white speckles on leaves, which can turn bronze",
                     "Fine webbing under leaves in hot, dry weather"],
        "treatment": ["Spray leaves, especially the undersides, with a strong jet of water",
                      "Use insecticidal soap or neem oil, repeating as the label says",
                      "Remove heavily infested leaves"],
        "prevention": ["Keep plants well watered and avoid dust and drought stress",
                       "Protect natural predators by avoiding broad-spectrum insecticides",
                       "Check the undersides of leaves regularly"],
    },
    "target spot": {
        "type": "Fungal",
        "symptoms": ["Brown spots with rings and a yellow halo, looking a lot like early blight",
                     "Can spread to stems and fruit in warm, humid weather"],
        "treatment": ["Remove infected leaves",
                      "Use a fungicide labelled for target spot if needed",
                      "Improve airflow by pruning"],
        "prevention": ["Space plants and avoid overhead watering",
                       "Rotate crops and remove crop debris"],
    },
    "yellow leaf": {
        "type": "Viral (spread by whiteflies)",
        "symptoms": ["Leaves curl upward and turn yellow at the edges",
                     "Stunted plants with few flowers and fruit"],
        "treatment": ["There is no cure: remove and destroy infected plants early",
                      "Control whiteflies with yellow sticky traps, insecticidal soap or neem oil",
                      "Check nearby plants for whiteflies"],
        "prevention": ["Use virus-resistant varieties and healthy seedlings",
                       "Cover seedlings with fine insect-proof net",
                       "Remove weeds that host whiteflies"],
    },
    "mosaic virus": {
        "type": "Viral",
        "symptoms": ["Mottled light and dark green patches on leaves",
                     "Curled, narrow or distorted leaves and stunted growth"],
        "treatment": ["There is no cure: remove and destroy infected plants",
                      "Wash hands and disinfect tools after handling them",
                      "Do not handle tobacco before touching plants"],
        "prevention": ["Use certified virus-free seed and resistant varieties",
                       "Disinfect tools and remove weeds",
                       "Rotate crops and clear plant debris"],
    },
    "healthy": {
        "type": "Healthy",
        "symptoms": ["The leaf looks healthy: even green colour with no spots, mold or curling"],
        "treatment": ["No treatment needed"],
        "prevention": ["Water at the base of the plant and keep good airflow",
                       "Check leaves regularly so problems are caught early",
                       "Remove fallen or damaged leaves"],
    },
}


def pretty(name):
    n = re.sub(r"_+", " ", name).strip()
    return n.replace("Tomato Tomato", "Tomato").replace("YellowLeaf", "Yellow Leaf")


def lookup(name):
    n = pretty(name).lower()
    for key, info in DISEASES.items():
        if key in n:
            return info
    return None


def bullets(items):
    return "\n".join(f"- {i}" for i in items)


@st.cache_resource
def load():
    model = keras.models.load_model(HERE / "plant_model.keras")
    names = json.load(open(HERE / "class_names.json"))
    return model, names


model, class_names = load()

samples = {}
sample_dir = HERE / "samples"
if sample_dir.exists():
    for p in sorted(sample_dir.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            samples[p.stem.replace("_", " ").title()] = p

with st.sidebar:
    st.header("About")
    st.write("Upload a leaf photo and the model predicts the plant condition, "
             "then shows symptoms, treatment and prevention tips.")
    st.caption("Model: fine-tuned MobileNetV2, 93.4% test accuracy on 2,064 unseen images.")
    st.subheader("Tips for a good photo")
    st.markdown("- One leaf filling most of the frame\n- Plain background and good light\n- Sharp, not blurry")
    st.subheader("Supported plants")
    st.write("Tomato, potato and bell pepper (15 conditions).")

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf photo, or pick a sample leaf to see how it works.")

left, right = st.columns([1, 1.3])

with left:
    file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    choice = st.selectbox("...or try a sample leaf", ["None"] + list(samples))
    img = None
    if file:
        img = Image.open(file).convert("RGB")
    elif choice != "None":
        img = Image.open(samples[choice]).convert("RGB")
    if img is not None:
        st.image(img, caption="Leaf being analysed")

with right:
    if img is None:
        st.info("Upload a leaf photo or choose a sample on the left to see the diagnosis.")
    else:
        x = np.array(img.resize((224, 224)), dtype="float32")[None]
        probs = model.predict(x, verbose=0)[0]
        order = probs.argsort()[::-1][:3]
        top = class_names[order[0]]
        conf = float(probs[order[0]])
        info = lookup(top)

        if conf < 0.60:
            st.warning("Low confidence. The photo may be unclear, or not one of the supported leaves. "
                       "Try a sharper photo of a single leaf.")
        if "healthy" in top.lower():
            st.success(f"**{pretty(top)}**")
        else:
            st.error(f"**{pretty(top)}**")

        c1, c2 = st.columns(2)
        c1.metric("Confidence", f"{conf * 100:.1f}%")
        c2.metric("Type", info["type"] if info else "-")
        st.progress(min(conf, 1.0))

        chart = pd.DataFrame(
            {"Confidence (%)": [float(probs[i]) * 100 for i in order]},
            index=[pretty(class_names[i]) for i in order])
        st.bar_chart(chart)

        if info:
            t1, t2, t3 = st.tabs(["Symptoms", "Treatment", "Prevention"])
            with t1:
                st.markdown(bullets(info["symptoms"]))
            with t2:
                st.markdown(bullets(info["treatment"]))
            with t3:
                st.markdown(bullets(info["prevention"]))

        st.caption("General guidance only. For a serious outbreak, ask your local agriculture office "
                   "and always follow product labels.")