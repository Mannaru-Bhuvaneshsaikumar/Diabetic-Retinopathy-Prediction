import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Diabetic Retinopathy Predictor", layout="wide")
st.title("🩺 Diabetic Retinopathy Risk Predictor")
st.write("Enter patient details below to predict the risk of diabetic retinopathy.")

# Input fields
inputs = {
    "age": st.slider("Age", 0, 100, 30),
    "systolic_bp": st.slider("Systolic Blood Pressure (mmHg)", 80, 200, 120),
    "diastolic_bp": st.slider("Diastolic Blood Pressure (mmHg)", 50, 130, 80),
    "cholesterol": st.slider("Cholesterol (mg/dl)", 100, 400, 180),
}

# Predict
if st.button("Predict Retinopathy Risk"):
    input_array = np.array(list(inputs.values())).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    prediction_proba = model.predict_proba(input_array)[0][1] * 100  # Probability in %

    # Display Risk Status
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetic Retinopathy!")
    else:
        st.success(f"✅ Low Risk of Diabetic Retinopathy.")

    st.markdown(f"### 🧮 Risk Probability: **{prediction_proba:.2f}%**")

    # Create a modern 3D-feeling gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba,
        title={'text': "Retinopathy Risk Level", 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': "black"},
            'bar': {'color': "darkblue", 'thickness': 0.2},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},  # Low risk zone
                {'range': [50, 75], 'color': 'yellow'},      # Medium risk zone
                {'range': [75, 100], 'color': 'red'}          # High risk zone
            ],
            'threshold': {
                'line': {'color': "black", 'width': 8},
                'thickness': 0.9,
                'value': prediction_proba
            }
        },
        number={
            'suffix': '%',
            'font': {'size': 36}
        }
    ))

    fig.update_layout(
        paper_bgcolor="lavender",
        font={'color': "black", 'family': "Arial"},
        height=550,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add dynamic final advice message
    if prediction_proba < 50:
        st.success("🩺 You are currently at low risk. Keep maintaining a healthy lifestyle!")
    elif 50 <= prediction_proba <= 75:
        st.warning("⚠️ Moderate risk detected. Please monitor your health carefully!")
    else:
        st.error("🚨 High risk detected! Please consult a healthcare professional immediately.")

st.markdown("---")
st.caption("Developed with ❤️ for Diabetic Retinopathy Prediction Awareness")
