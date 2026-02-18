import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    models = {}
    models["Diabetes"] = pickle.load(open("diabetes_model.pkl", "rb"))
    models["Heart Disease"] = pickle.load(open("heart_model.pkl", "rb"))
    models["Breast Cancer"] = pickle.load(open("breast_cancer_model.pkl", "rb"))
    return models

models = load_models()

# =========================
# PDF GENERATOR
# =========================
def generate_pdf(name, disease, df, prediction, risk):

    filename = "Medical_Report.pdf"
    pdf = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle(
        "TitleStyle",
        fontSize=18,
        textColor=colors.darkblue,
        alignment=1
    )

    elements.append(Paragraph(f"DOC.AI – {disease} Report", title_style))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(f"<b>Patient Name:</b> {name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 15))

    result_text = "Detected" if prediction == 1 else "Not Detected"
    elements.append(Paragraph(
        f"<b>Result:</b> {result_text}<br/><b>Risk:</b> {risk:.2f}%",
        styles["Normal"]
    ))

    pdf.build(elements)
    return filename


# =========================
# UI START
# =========================

st.title("🩺 DOC.AI – Multi Disease Prediction System")

disease_option = st.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Breast Cancer"]
)

name = st.text_input("Enter Patient Name")
age = st.number_input("Age", 0, 120)

# =========================
# DIABETES INPUT
# =========================
if disease_option == "Diabetes":

    gender = st.selectbox("Sex", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", ["yes", "no"])
    heart_disease = st.selectbox("Heart Disease", ["yes", "no"])
    bmi = st.number_input("BMI", 0.0, 60.0)
    HbA1c_level = st.number_input("HbA1c Level", 0.0, 15.0)
    blood_glucose_level = st.number_input("Blood Glucose Level", 0, 400)

    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "yes" else 0
    heart_disease = 1 if heart_disease == "yes" else 0

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

# =========================
# HEART DISEASE INPUT
# =========================
elif disease_option == "Heart Disease":

    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    thalach = st.number_input("Max Heart Rate")

    sex = 1 if sex == "Male" else 0

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach
    }])

# =========================
# BREAST CANCER INPUT
# =========================
elif disease_option == "Breast Cancer":

    radius = st.number_input("Mean Radius")
    texture = st.number_input("Mean Texture")
    perimeter = st.number_input("Mean Perimeter")
    area = st.number_input("Mean Area")

    input_data = pd.DataFrame([{
        "mean radius": radius,
        "mean texture": texture,
        "mean perimeter": perimeter,
        "mean area": area
    }])


# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict"):

    model = models[disease_option]

    # Ensure correct feature order
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"🔴 {disease_option} Detected (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"🟢 No {disease_option} Detected (Risk: {probability*100:.2f}%)")

    # Generate PDF
    pdf_file = generate_pdf(
        name,
        disease_option,
        input_data,
        prediction,
        probability * 100
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download Medical Report",
            f,
            file_name=f"{name}_{disease_option}_Report.pdf",
            mime="application/pdf"
        )
