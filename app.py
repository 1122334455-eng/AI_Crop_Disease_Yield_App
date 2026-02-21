import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import qrcode

# ---------------- Models Load ----------------
disease_model = load_model("../models/disease_model.h5")
class_names = joblib.load("../models/disease_classes.pkl")

yield_model = joblib.load("../models/yield_model.pkl")
yield_features = joblib.load("../models/yield_features.pkl")

# ---------------- Functions ----------------
def predict_disease(image):
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)
    preds = disease_model.predict(img)[0]
    top_indices = np.argsort(preds)[::-1][:3]
    top_predictions = [(class_names[i], preds[i]*100) for i in top_indices]
    return top_predictions

def predict_yield(input_dict):
    df = pd.DataFrame([input_dict])
    for col in yield_features:
        if col not in df:
            df[col] = 0
    df = df[yield_features]
    prediction = yield_model.predict(df)[0]
    return round(prediction, 2)

def generate_qr(url):
    qr = qrcode.QRCode(
        version=1,
        box_size=10,
        border=2
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="AI Crop System", layout="wide", page_icon="🌾")

# ---------------- Theme Selector ----------------
theme = st.sidebar.selectbox("Choose Theme", ["Light 🌞", "Dark 🌙", "Naswari 🟤", "Bright Green 💚"])
if theme == "Dark 🌙":
    st.markdown("<style>.stApp {background-color: #0e1117; color: #fff;}</style>", unsafe_allow_html=True)
elif theme == "Bright Green 💚":
    st.markdown("<style>.stApp {background-color: #00FF00; color: #000;}</style>", unsafe_allow_html=True)
elif theme == "Naswari 🟤":
    st.markdown("""
        <style>
        .stApp {background-color: #d2b48c; color: #000;}
        div.stFileUploader {background-color: #c19a6b; border-radius:10px; padding:10px;}
        .css-1v3fvcr h1 {background-color: #c19a6b; border-radius:8px; padding:10px;}
        .stImage > div > figcaption {color:#000; font-weight:bold;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp {background: linear-gradient(to right, #a8e6cf, #dcedc1); color: #333;}</style>", unsafe_allow_html=True)

# ---------------- Tabs ----------------
st.sidebar.markdown("### AI Crop System")
tab1, tab2, tab3 = st.tabs(["Disease Detection", "Yield Prediction", "Tips & Forecast"])

# ============================
# 🌿 DISEASE DETECTION TAB
# ============================
with tab1:
    st.header("🌿 Crop Disease Detection")
    crop_type = st.selectbox("Select Crop Type", ["Tomato", "Potato", "Wheat", "Other"])
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        col1.image(image, width=200)
        col2.write("Click the button below to detect disease")

        if st.button("Detect Disease"):
            top_preds = predict_disease(image)

            # ---------------- Top 3 Predictions ----------------
            st.subheader("Top 3 Disease Predictions:")
            for disease, conf in top_preds:
                st.info(f"🩺 {disease} - Confidence: {conf:.2f}%")

            # ---------------- Interactive Bar Chart ----------------
            diseases = [d[0] for d in top_preds]
            confidences = [d[1] for d in top_preds]
            fig = go.Figure([go.Bar(x=diseases, y=confidences, text=[f"{c:.2f}%" for c in confidences],
                                    textposition="auto", marker_color='green')])
            fig.update_layout(title="Disease Confidence Levels")
            st.plotly_chart(fig)

            # ---------------- Crop-specific Advice ----------------
            main_disease = top_preds[0][0].lower()
            if "healthy" in main_disease:
                st.success("✅ Plant is healthy")
            elif "blight" in main_disease and crop_type.lower() == "tomato":
                st.warning("⚠️ Advice: Use copper fungicide on tomato plants")
            elif "spot" in main_disease and crop_type.lower() == "tomato":
                st.warning("🌿 Advice: Apply neem oil spray on tomato leaves")
            else:
                st.error("❌ Advice: Consult agriculture expert")

            # ---------------- CSV Report Download ----------------
            report = pd.DataFrame({
                "Crop": [crop_type],
                "Top Disease": [top_preds[0][0]],
                "Confidence": [f"{top_preds[0][1]:.2f}%"]
            })
            csv = report.to_csv(index=False)
            st.download_button("📥 Download Disease Report (CSV)", csv, file_name="disease_report.csv")

# ============================
# 🌾 YIELD PREDICTION TAB
# ============================
with tab2:
    st.header("🌾 Crop Yield Prediction")
    st.markdown("Enter environmental values:")

    col1, col2, col3 = st.columns(3)
    rainfall = col1.number_input("Average Rainfall (mm)", 0.0, 5000.0, 1000.0)
    pesticides = col2.number_input("Pesticides (tonnes)", 0.0, 100000.0, 100.0)
    temperature = col3.number_input("Temperature (°C)", -10.0, 60.0, 25.0)

    if st.button("Predict Yield"):
        data = {
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temperature
        }
        prediction = predict_yield(data)
        st.success(f"🌾 Estimated Crop Yield: {prediction} tonnes per hectare")

        # ---------------- Yield Trend Chart ----------------
        history = pd.DataFrame({
            "Year": [2020, 2021, 2022, 2023],
            "Yield": [2.5, 2.8, 3.0, prediction]  # include current prediction
        })
        fig = px.line(history, x="Year", y="Yield", title="Yield Trend", markers=True)
        st.plotly_chart(fig)

        # ---------------- Forecast Next Year ----------------
        forecast = prediction * 1.05
        st.info(f"🌱 Next Year Forecast (Estimated 5% growth): {forecast:.2f} tonnes/ha")

        # ---------------- CSV Report Download ----------------
        report = pd.DataFrame({
            "Rainfall_mm": [rainfall],
            "Pesticides_tonnes": [pesticides],
            "Temperature_C": [temperature],
            "Estimated_Yield_tonnes_per_ha": [prediction],
            "Next_Year_Forecast_tonnes_per_ha": [forecast]
        })
        csv = report.to_csv(index=False)
        st.download_button("📥 Download Yield Report (CSV)", csv, file_name="yield_report.csv")

# ============================
# 💡 TIPS & FORECAST TAB
# ============================
with tab3:
    st.header("💡 Tips & Forecast")
    st.info("🌱 Keep soil moist but not waterlogged.")
    st.info("🌱 Avoid over-fertilizing, maintain proper nutrient balance.")
    st.info("🌱 Regularly inspect leaves for early disease detection.")
    st.info("🌱 Use crop-specific fungicides/pesticides as needed.")
    st.info("🌱 Monitor rainfall, temperature, and pesticide usage for best yield outcomes.")

# ---------------- Sidebar QR Code ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 Open App on Mobile")
app_url = "http://10.233.254.78:8501"  # <-- yahan deploy link dalen
qr_img = generate_qr(app_url)

# Convert PIL Image to bytes
buffer = BytesIO()
qr_img.save(buffer, format="PNG")
qr_bytes = buffer.getvalue()

st.sidebar.image(qr_bytes, caption="Scan to open on mobile browser", width=200)
st.sidebar.download_button("📥 Download QR Code", qr_bytes, file_name="AI_Crop_App_QR.png")
