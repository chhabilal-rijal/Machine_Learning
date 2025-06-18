import streamlit as st
import pickle
import numpy as np

# ---------- Load Model Safely ----------
model = None
try:
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid model.")
except Exception as e:
    class DummyModel:
        def predict(self, X):
            return [1234567.89]
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

/* Input label styles */
div[data-testid="stNumberInput"] label > div,
div[data-testid="stSelectbox"] label > div {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-bottom: 6px !important;
    color: #00ffcc !important;
    text-shadow: 0 0 8px #00ffcc, 0 0 15px #00ffcc !important;
}

/* Inputs styles */
.stNumberInput input,
.stSelectbox [data-baseweb="select"] > div[role="button"] {
    background: rgba(255,255,255,0.15) !important;
    color: #fff !important;
    font-size: 1.7rem !important;
    border-radius: 12px !important;
    padding: 20px !important;
    width: 100% !important;
    max-width: 650px;
    border: 1px solid rgba(255,255,255,0.3) !important;
}

.stSelectbox [data-baseweb="select"] > div[role="button"] * {
    color: #fff !important;
    font-size: 1.7rem !important;
}

.stSelectbox [data-baseweb="select"] svg {
    fill: #fff !important;
    width: 28px !important;
    height: 28px !important;
}

/* Popover options */
div[data-baseweb="popover"] div[role="option"] {
    color: #262730 !important;
    background-color: #f0f2f6 !important;
    font-size: 1.5rem !important;
    padding: 12px 20px !important;
}
div[data-baseweb="popover"] div[role="option"]:hover {
    background-color: #e0e2e6 !important;
}

/* Container styles */
div[data-testid*="stNumberInput"],
div[data-testid*="stSelectbox"],
div[data-testid*="stButton"] {
    margin-left: 0 !important;
    margin-right: auto !important;
    width: 100% !important;
    max-width: 550px;
}

/* Button styles */
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

/* Price box styles */
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
    margin-top: 920px;
    float: right;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}

/* Price label and icon */
.price-header {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    font-size: 24px;
    font-weight: 700;
    text-shadow: 0 0 8px #1abc9c, 0 0 12px #1abc9c;
    animation: glow 2s ease-in-out infinite alternate;
    margin: 0 auto 2px auto;
    line-height: 1.1;
}

/* Glowing house icon */
.house-icon {
    font-size: 36px;
    color: #1abc9c;
    text-shadow:
        0 0 6px #1abc9c,
        0 0 12px #1abc9c,
        0 0 18px #16a085;
    animation: glow 2s ease-in-out infinite alternate;
}

/* Price value */
.price-value {
    font-size: 32px;
    font-weight: 700;
    color: #00ffcc;
    text-shadow: 0 0 6px #00ffcc;
    margin-top: 0;
    width: 100%;
    text-align: center;
    line-height: 1.1;
}

/* Two column layout */
.left-column, .right-column {
    width: 48%;
    display: inline-block;
    vertical-align: top;
}
.right-column {
    float: right;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="glow-title">🏡 House Price Predictor</div>', unsafe_allow_html=True)

# Initialize price in session state if not exists
if 'price' not in st.session_state:
    st.session_state.price = None

# Two columns
left_col, right_col = st.columns([1, 1])

with left_col:
    area = st.number_input("📐 Area (sq ft)", min_value=300, max_value=10000, value=1000, step=50)
    bedrooms = st.selectbox("🛌 Bedrooms", list(range(1, 11)))
    bathrooms = st.selectbox("🛁 Bathrooms", list(range(1, 11)))
    stories = st.selectbox("🏢 Stories", list(range(1, 5)))
    mainroad = st.selectbox("🚗 Main Road Access", ["Yes", "No"])
    airconditioning = st.selectbox("❄️ Air Conditioning", ["Yes", "No"])
    parking = st.selectbox("🚘 Parking", list(range(0, 6)))
    prefarea = st.selectbox("🌳 Preferred Area", ["Yes", "No"])
    furnishingstatus = st.selectbox("🛋 Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

    # Encode inputs
    mainroad_val = 1 if mainroad == "Yes" else 0
    airconditioning_val = 1 if airconditioning == "Yes" else 0
    prefarea_val = 1 if prefarea == "Yes" else 0
    furnishing_map = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}
    furnishing_val = furnishing_map[furnishingstatus]

    # Predict button in left column
    if st.button("Predict Price"):
        features = np.array([[area, bedrooms, bathrooms, stories,
                              mainroad_val, airconditioning_val, parking,
                              prefarea_val, furnishing_val]])
        try:
            price = model.predict(features)[0]
            st.session_state.price = price
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with right_col:
    if st.session_state.price is not None:
        st.markdown(f"""
        <div class="price-box">
            <div class="price-header">
                <span class="house-icon">🏠</span>
                Estimated Price:
            </div>
            <div class="price-value">₹ {st.session_state.price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
