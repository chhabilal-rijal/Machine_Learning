import streamlit as st
import pickle
import numpy as np

# ---------- Load Model Safely ----------
model = None
try:
    with open('logreg_model.pkl', 'rb') as f:
        model = pickle.load(f)
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid model.")
except Exception as e:
    class DummyModel:
        def predict(self, X):
            return [1]  # Example dummy output
    model = DummyModel()
    st.warning(f"⚠️ Failed to load real model. Using dummy model. Error: {e}")

# ---------- CSS Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600&display=swap');

body, .main {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #1e3c72, #2980b9);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Poppins', sans-serif;
    color: #f0f2f6;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.glow-title {
    font-size: 48px;
    font-weight: 700;
    color: #fff;
    text-align: center;
    animation: glow 2s ease-in-out infinite alternate;
    margin-bottom: 30px;
}
@keyframes glow {
    from {text-shadow: 0 0 10px #1abc9c, 0 0 20px #1abc9c;}
    to {text-shadow: 0 0 20px #16a085, 0 0 30px #16a085;}
}

/* Label styling for number inputs and select boxes */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-bottom: 6px !important;
    color: #00ffcc !important;
    text-shadow: 0 0 8px #00ffcc, 0 0 15px #00ffcc !important;
}

/* Font size and style inside number input fields (keeping spin buttons visible) */
div[data-testid="stNumberInput"] input {
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    color: #fff !important;
    background: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    width: 100% !important;
    max-width: 650px;
    border: 1px solid rgba(255,255,255,0.3) !important;
}

/* Select box outer styling (the button itself) */
.stSelectbox [data-baseweb="select"] > div[role="button"] {
    background: rgba(255,255,255,0.15) !important;
    color: #fff !important;
    border-radius: 12px !important;
    padding: 20px !important; /* This adds padding around the text */
    width: 100% !important;
    max-width: 650px;
    border: 1px solid rgba(255,255,255,0.3) !important;
}

/* The most aggressive attempt for selectbox value text */
/* This targets any direct text content within the 'control' part of the selectbox */
.stSelectbox [data-baseweb="select"] > div[role="button"] > div * {
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    color: #fff !important;
}

div.stButton > button {
    background-color: #16a085;
    color: white;
    font-size: 32px;
    font-weight: 600;
    border-radius: 15px;
    padding: 20px 50px;
    border: none;
    cursor: pointer;
    transition: 0.3s ease;
    margin-top: 30px !important;
}
div.stButton > button:hover {
    background-color: #1abc9c;
    transform: scale(1.05);
}
.price-box {
    background: rgba(255, 255, 255, 0.15);
    padding: 14px 24px;
    border-radius: 20px;
    border: 2px solid #1abc9c;
    box-shadow: 0 0 12px #00ffcc;
    color: #ffffff;
    text-align: center;
    max-width: 600px;
    min-width: 400px;
    margin-top: 720px;
    float: right;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.price-header {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    font-size: 24px;
    font-weight: 700;
    text-shadow: 0 0 8px #1abc9c, 0 0 12px #1abc9c;
    animation: glow 2s ease-in-out infinite alternate;
}
.price-value {
    font-size: 32px;
    font-weight: 700;
    color: #00ffcc;
    text-shadow: 0 0 6px #00ffcc;
    margin-top: 0;
    width: 100%;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- App Title ----------
st.markdown('<div class="glow-title">🧠 Diabetes Predictor:</div>', unsafe_allow_html=True)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'display_result' not in st.session_state: # New session state for display text
    st.session_state.display_result = None

# ---------- Inputs ----------
left_col, right_col = st.columns([1, 1])

with left_col:
    age = st.number_input("🎂 Age", min_value=1, max_value=120, value=30, step=1)
    hypertension = st.selectbox("💓 Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("❤️ Heart Disease", ["No", "Yes"])
    smoking_history = st.selectbox("🚬 Smoking History", [
        "never",
        "current",
        "former",
        "not current",
        "ever"
    ])
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hba1c_level = st.number_input("🩸 HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    blood_glucose_level = st.number_input("🩸 Blood Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=1.0)

    # Encode inputs
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    smoking_map = {
        "never": 0,
        "current": 1,
        "former": 2,
        "not current": 3,
        "ever": 4
    }
    smoking_val = smoking_map[smoking_history]

    if st.button("Predict Risk"):
        features = np.array([[age, hypertension_val, heart_disease_val,
                               smoking_val, bmi, hba1c_level, blood_glucose_level]])
        try:
            pred = model.predict(features)[0]
            st.session_state.result = pred

            # Map numerical prediction to text
            if pred == 1:
                st.session_state.display_result = "Diabetes"
            else:
                st.session_state.display_result = "No Diabetes"

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with right_col:
    # Use display_result for output
    if st.session_state.display_result is not None:
        st.markdown(f"""
        <div class="price-box">
            <div class="price-header">
                <span class="house-icon">🧠</span>
                Diabetes Prediction:
            </div>
            <div class="price-value">{st.session_state.display_result}</div>
        </div>
        """, unsafe_allow_html=True)