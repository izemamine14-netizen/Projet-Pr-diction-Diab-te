import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =========================
# LOAD
# =========================
model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# =========================
# UI
# =========================
st.title("🧠 Prédiction du diabète")

inputs = []

# intervalles réalistes
ranges = {
    "BMI": (10, 60),
    "Age": (18, 100),
    "PhysHlth": (0, 30),
    "MentHlth": (0, 30),
    "GenHlth": (1, 5),
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "Smoker": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1)
}

for f in features:
    if f in ranges:
        min_val, max_val = ranges[f]
        val = st.slider(f, min_val, max_val, int((min_val + max_val)/2))
    else:
        val = st.number_input(f, 0.0, 1.0, 0.0)

    inputs.append(val)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Prédire"):

    x = np.array(inputs).reshape(1, -1)
    x = scaler.transform(x)

    prob = model.predict(x)[0][0]

    st.write(f"📊 Probabilité : {prob * 100:.2f}%")

    if prob > 0.35:
        st.error("⚠️ Risque de diabète")
    else:
        st.success("✅ Pas de diabète")