# app.py — Life Insurance Premium Predictor (PyCaret >=3.0)
import streamlit as st
import pandas as pd
import os
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment, load_model, save_model

st.set_page_config(page_title="Life Insurance Predictor", page_icon="💰", layout="centered")

st.title("💰 Life Insurance Premium Predictor")
st.caption("Przewidywanie miesięcznej opłaty za ubezpieczenie na życie (PyCaret)")

# --------------------------------
# Formularz użytkownika
# --------------------------------
st.header("1️⃣ Dane klienta")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Wiek", 18, 100, 30)
    sex = st.selectbox("Płeć", ["male", "female"])
    children = st.number_input("Liczba dzieci", 0, 10, 0)
with col2:
    smoker = st.selectbox("Czy palacz?", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    height = st.number_input("Wzrost [cm]", 120, 220, 175)
    weight = st.number_input("Waga [kg]", 40, 200, 70)

bmi = round(weight / ((height / 100) ** 2), 1)

user_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

st.write("📋 Dane klienta")
st.dataframe(user_data, use_container_width=True)

# --------------------------------
# Trening modelu (jeśli brak)
# --------------------------------
st.header("2️⃣ Model ML – przewidywanie opłaty")

model_path = "insurance_model"
exp = RegressionExperiment()

if not os.path.exists(model_path + ".pkl"):
    with st.spinner("Trening modelu ML (PyCaret)..."):
        data = get_data("insurance")
        exp.setup(data=data, target="charges", session_id=123)
        model = exp.create_model("lr")  # najprostszy i najszybszy
        exp.save_model(model, model_path)
    st.success("✅ Model wytrenowany i zapisany.")
else:
    model = exp.load_model(model_path)
    st.info("📦 Załadowano istniejący model z pliku.")

# --------------------------------
# Predykcja
# --------------------------------
if st.button("🔮 Przewiduj opłatę"):
    with st.spinner("Przewidywanie..."):
        pred = exp.predict_model(model, data=user_data)
        result = float(pred["prediction_label"].iloc[0])
        st.success(f"💵 Przewidywana miesięczna opłata: **{result:.2f} USD**")

        # --------------------------------
        # v3: Sugestie optymalizacji
        # --------------------------------
        st.header("3️⃣ Sugestie, jak obniżyć opłatę")
        tips = []
        if smoker == "yes":
            tips.append("🚭 Rzucenie palenia znacznie obniży składkę – palacze płacą nawet 2–3× więcej.")
        if bmi > 30:
            tips.append("⚖️ Redukcja BMI poniżej 25 znacząco zmniejsza ryzyko.")
        if age < 25:
            tips.append("🎓 Młodsze osoby mogą skorzystać z planów studenckich lub rodzinnych.")
        if children > 3:
            tips.append("👨‍👩‍👧‍👦 Ubezpieczenia rodzinne są tańsze przy wspólnych polisach.")
        if not tips:
            tips.append("✅ Twoje dane wyglądają dobrze – trudno będzie obniżyć składkę.")
        for t in tips:
            st.markdown(f"- {t}")
