import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================
# LOAD
# =========================
model = load_model("model.keras", compile=False)
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# =========================
# UI
# =========================
st.title("🧠 Prédiction du diabète")

inputs = []

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
    "Veggies": (0, 1),
    "Sex": (0, 1),
}

for f in features:
    if f in ranges:
        val = st.slider(f, ranges[f][0], ranges[f][1], int((ranges[f][0] + ranges[f][1]) / 2))
    else:
        val = st.number_input(f, value=0.0)

    inputs.append(val)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Prédire"):

    x = np.array(inputs).reshape(1, -1)
    x_scaled = scaler.transform(x)

    prob = model.predict(x_scaled, verbose=0)[0][0]

    st.write(f"📊 Probabilité : {prob * 100:.2f}%")

    if prob > 0.35:
        st.error("⚠️ Risque de diabète")
    else:
        st.success("✅ Pas de diabète")

# =========================
# SHAP
# =========================
st.subheader("🔍 Explication (SHAP)")

x = np.array(inputs).reshape(1, -1)
x_scaled = scaler.transform(x)

# Background
background = np.zeros((50, x_scaled.shape[1]))

explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(x_scaled)

# =========================
# NORMALISATION OUTPUT SHAP
# =========================
if isinstance(shap_values, list):
    values = shap_values[0]
    base = explainer.expected_value[0]
else:
    values = shap_values
    base = explainer.expected_value

values = np.array(values)[0].reshape(-1)
features_values = x_scaled[0].reshape(-1)

# Base scalaire
if isinstance(base, (list, np.ndarray)):
    base = base[0]

# =========================
# PLOT
# =========================
fig = plt.figure(figsize=(10, 5))

shap.plots._waterfall.waterfall_legacy(
    base,
    values,
    features=features_values,
    feature_names=features,
    show=False
)

st.pyplot(fig)