import streamlit as st
import pickle
import numpy as np

# ---------- Load KNN Model Safely ----------
model = None
try:
    with open('knn_iris.pkl', 'rb') as f:
        model = pickle.load(f)
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid model.")
except Exception as e:
    class DummyModel:
        def predict(self, X):
            return ["Unknown"]  # Example dummy output
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
    margin-bottom: 70px;
}
@keyframes glow {
    from {text-shadow: 0 0 10px #1abc9c, 0 0 20px #1abc9c;}
    to {text-shadow: 0 0 20px #16a085, 0 0 30px #16a085;}
}
div[data-testid="stNumberInput"] label > div,
div[data-testid="stSelectbox"] label > div {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #00ffcc !important;
    text-shadow: 0 0 8px #00ffcc, 0 0 15px #00ffcc !important;
}

div[data-testid="stNumberInput"] input {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #fff !important;
    background: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    width: 100% !important;
    max-width: 650px;
    border: 1px solid rgba(255,255,255,0.3) !important;
}

.stSelectbox [data-baseweb="select"] > div[role="button"] {
    background: rgba(255,255,255,0.15) !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 20px !important;
    width: 100% !important;
    max-width: 650px;
    border: 1px solid rgba(255,255,255,0.3) !important;
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
    margin-top: 550px;
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
st.markdown('<div class="glow-title">🌸 KNN Classifier</div>', unsafe_allow_html=True)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

# ---------- Inputs ----------
left_col, right_col = st.columns([1, 1])

with left_col:
    # Example features for iris-like dataset (adjust if needed)
    sepal_length = st.number_input("🌿 Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("🌿 Sepal Width", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("🌸 Petal Length", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    petal_width = st.number_input("🌸 Petal Width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    if st.button("Predict Species"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        try:
            pred = model.predict(features)[0]
            st.session_state.result = pred
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with right_col:
    if st.session_state.result is not None:
        st.markdown(f"""
        <div class="price-box">
            <div class="price-header">
                🌸 Predicted Species:
            </div>
            <div class="price-value">{st.session_state.result}</div>
        </div>
        """, unsafe_allow_html=True)
