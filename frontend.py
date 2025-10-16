import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load your trained model
model = joblib.load("heart_disease_model.joblib")

# Streamlit page setup
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")

# --- Create Tabs ---
tab1, tab2, tab3= st.tabs(["🩺 Prediction", "📊 Performance Metrics","Info"])

# -------------------------------
# TAB 1: PREDICTION PAGE
# -------------------------------
with tab1:
    st.write("Enter patient details below to check the likelihood of heart disease.")

    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=240)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

        with col2:
            restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
            ca = st.selectbox("No. of Major Vessels (ca)", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        # Convert inputs
        sex_value = 1 if sex == "Male" else 0

        # Arrange inputs in same order as training
        features = [[
            age, sex_value, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]]

        # Predict
        prediction = model.predict(features)[0]

        st.subheader("🧠 Prediction Result:")
        if prediction == 1:
            st.error("⚠️ High likelihood of Heart Disease")
        else:
            st.success("💚 Low likelihood of Heart Disease")

# -------------------------------
# TAB 2: PERFORMANCE METRICS PAGE
# -------------------------------
with tab2:
    st.write("### 📈 Model Performance Metrics")

    # Load dataset (for demonstration)
    try:
        data = pd.read_csv("heart.csv")
        X = data.drop("target", axis=1)
        y = data["target"]

        # Predict on all data
        y_pred = model.predict(X)

        # Metrics
        st.write("**Classification Report:**")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("⚠️ heart.csv not found. Please place it in the same folder to view performance metrics.")

with tab3:
    st.write("### ℹ️ Information")
    st.markdown("""
    This application predicts the likelihood of heart disease based on patient data.
    - **Developed by:** Sachin Durairaj, Karthikeyan V, Rahul Sai Shrinivas
    """)