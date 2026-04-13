import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# 🔥 CHARGEMENT DU MODÈLE
# =========================
model = tf.keras.models.load_model("model.keras")  # ou model_reduced.keras

# =========================
# 🎯 TITRE INTERFACE
# =========================
st.title("🩺 Prédiction du Diabète")
st.write("Entrez les informations du patient pour prédire s'il est diabétique ou non.")

# =========================
# 📊 INPUTS UTILISATEUR
# =========================

pregnancies = st.number_input("Grossesses", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
blood_pressure = st.number_input("Pression artérielle", 0, 200, 70)
skin_thickness = st.number_input("Épaisseur de la peau", 0, 100, 20)
insulin = st.number_input("Insuline", 0, 900, 80)
bmi = st.number_input("IMC (BMI)", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Âge", 0, 120, 30)

# =========================
# 🔥 PRÉDICTION
# =========================
if st.button("Prédire"):

    # créer vecteur input
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    # prédiction
    prediction = model.predict(input_data)

    # résultat
    if prediction[0][0] > 0.5:
        st.error("⚠️ Le patient est susceptible d'être diabétique")
    else:
        st.success("✅ Le patient n'est probablement pas diabétique")

    st.write("Score :", float(prediction[0][0]))