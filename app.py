# File: app.py
# Streamlit + PyCaret – Life Insurance Premium Predictor
# v1: Formularz danych klienta
# v2: Model ML PyCaret (dataset insurance)
# v3: Sugestie optymalizacji

import streamlit as st
from pycaret.datasets import get_data
from pycaret.regression import setup, compare_models, predict_model, save_model, load_model
import pandas as pd
import os

st.set_page_config(page_title="Life Insurance Predictor", page_icon="💰", layout="centered")

st.title("💰 Life Insurance Premium Predictor")
st.caption("Aplikacja przewiduje miesięczną opłatę za ubezpieczenie na życie na podstawie danych demograficznych (PyCaret).")

# -------------------------------
# v1: Formularz danych użytkownika
# -------------------------------
st.header("1️⃣ Dane klienta")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Wiek", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Płeć", ["male", "female"])
    bmi = st.number_input("Wskaźnik BMI (waga / wzrost²)", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Liczba dzieci", min_value=0, max_value=10, value=0, step=1)
with col2:
    smoker = st.selectbox("Czy palacz?", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    height = st.number_input("Wzrost [cm]", min_value=120, max_value=220, value=175)
    weight = st.number_input("Waga [kg]", min_value=40, max_value=200, value=70)

# Przelicz BMI, jeśli użytkownik poda wzrost/wagę zamiast BMI
if bmi == 25.0 and height and weight:
    bmi = round(weight / ((height / 100) ** 2), 1)

# Dane w formie ramki
user_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

st.write("📋 Dane klienta:")
st.dataframe(user_data, use_container_width=True)

# -------------------------------
# v2: Model ML (PyCaret)
# -------------------------------
st.header("2️⃣ Model ML – przewidywanie opłaty")

model_path = "insurance_model.pkl"

if not os.path.exists(model_path):
    with st.spinner("Trening modelu ML (PyCaret)..."):
        data = get_data("insurance")
        s = setup(
            data,
            target="charges",
            session_id=123,
            silent=True,
            verbose=False,
        )
        best_model = compare_models(sort="R2")
        save_model(best_model, "insurance_model")
    st.success("Model został wytrenowany i zapisany.")
else:
    best_model = load_model("insurance_model")

if st.button("🔮 Przewiduj opłatę"):
    with st.spinner("Przewidywanie..."):
        prediction = predict_model(best_model, data=user_data)
        charge = float(prediction["Label"].iloc[0])
        st.success(f"💵 Przewidywana miesięczna opłata: **{charge:.2f} USD**")

        # -------------------------------
        # v3: Sugestie optymalizacji
        # -------------------------------
        st.header("3️⃣ Sugestie, jak obniżyć opłatę")
        tips = []
        if smoker == "yes":
            tips.append("🚭 Rzucenie palenia znacznie obniży składkę – palacze płacą nawet 2–3x więcej.")
        if bmi > 30:
            tips.append("⚖️ Zmniejszenie BMI poniżej 25 znacząco poprawia ocenę ryzyka.")
        if age < 25:
            tips.append("🎓 Młodsze osoby mogą skorzystać z planów studenckich lub rodzinnych.")
        if children > 3:
            tips.append("👨‍👩‍👧‍👦 Ubezpieczenia rodzinne często są tańsze przy wspólnych polisach.")
        if not tips:
            tips.append("✅ Twoje dane wyglądają dobrze! Trudno będzie znacząco obniżyć składkę.")
        for t in tips:
            st.markdown(f"- {t}")
