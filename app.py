import streamlit as st
import time
import pickle
import pandas as pd

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="University SGPA Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Modern College ML UI
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .header-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #c7d2fe;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
    .header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #a5b4fc;
        font-size: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Content Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }

    /* Validation Badge */
    .val-badge {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    .val-outstanding { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .val-excellent { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); }
    .val-good { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3); }
    .val-average { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .val-risk { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    /* Result Placeholder Card */
    .result-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        margin-top: 1.5rem;
    }
    .result-score {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    .status-chip {
        background: rgba(234, 179, 8, 0.15);
        color: #fde047;
        border: 1px solid rgba(250, 204, 21, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Section
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 SGPA Predictor")
    st.caption("College Machine Learning Project")
    st.divider()

    st.markdown("#### 📌 Project Details")
    st.info(
        "This application is designed to estimate student **SGPA** based on "
        "their **Internal Marks Average** using Machine Learning techniques."
    )

    st.markdown("#### ⚙️ System Status")
    st.success("⚡ **Model State**: Active (`sgpa_linear_model.pkl`)")

    st.markdown("#### 🛠️ Tech Stack")
    st.markdown("""
    - **Frontend**: Streamlit
    - **Language**: Python 3.x
    - **ML Framework**: Scikit-Learn (Planned)
    - **Target Output**: SGPA Scale (0.0 - 10.0)
    """)

    st.divider()
    st.caption("© 2026 University ML Project Team")

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
    <div class="header-card">
        <div class="header-badge">ACADEMIC ML MODEL DEMO</div>
        <h1 class="header-title">🎓 SGPA Predictor</h1>
        <p class="header-subtitle">Estimate your Semester Grade Point Average based on Internal Marks</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Section
# ---------------------------------------------------------
st.markdown("<div class=\"glass-card\">", unsafe_allow_html=True)
st.markdown("### 📝 Enter Input Parameters")

col1, col2 = st.columns([3, 2], gap="medium")

with col1:
    internal_marks = st.number_input(
        "Internal Marks Average (0.0 - 100.0)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=0.5,
        help="Enter the average percentage score obtained in internal assessments."
    )

with col2:
    st.markdown("**Expected Performance Tier:**")
    if internal_marks >= 90:
        st.markdown('<div class="val-badge val-outstanding">🌟 Outstanding Performance</div>', unsafe_allow_html=True)
    elif internal_marks >= 75:
        st.markdown('<div class="val-badge val-excellent">✨ Excellent Range</div>', unsafe_allow_html=True)
    elif internal_marks >= 60:
        st.markdown('<div class="val-badge val-good">👍 Good Standing</div>', unsafe_allow_html=True)
    elif internal_marks >= 40:
        st.markdown('<div class="val-badge val-average">⚠️ Average Range</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="val-badge val-risk">❗ Needs Improvement</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction Trigger & Result Section
# ---------------------------------------------------------
predict_button = st.button("🚀 Predict SGPA", use_container_width=True, type="primary")

if predict_button:
    with st.spinner("🔄 Running feature transformation & model inference..."):
        time.sleep(0.4)  # Smooth transition effect for presentation

        try:
            # 1. Load trained model artifact
            with open("sgpa_linear_model.pkl", "rb") as file:
                saved_data = pickle.load(file)
            
            model = saved_data["model"]
            scaler = saved_data["scaler"]
            
            # 2. Prepare data for inference
            input_data = pd.DataFrame({"AvgInternal": [internal_marks]})
            
            # 3. Scale input using the same scaler from training
            input_scaled = scaler.transform(input_data)
            
            # 4. Make prediction
            predicted_sgpa = model.predict(input_scaled)[0]
            
            # Bound the prediction between 0.0 and 10.0 scale logically
            predicted_sgpa = max(0.0, min(10.0, float(predicted_sgpa)))

            st.markdown("""
                <div class="result-card">
                    <span class="status-chip" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3);">
                        ✅ MODEL CONNECTED
                    </span>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.8rem; margin-bottom: 0;">Predicted SGPA Output</p>
                    <div class="result-score">{:.2f}</div>
                    <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 0.5rem;">
                        Internal Marks Input: <strong>{:.1f} / 100.0</strong>
                    </p>
                </div>
            """.format(predicted_sgpa, internal_marks), unsafe_allow_html=True)
            
        except FileNotFoundError:
            st.error("❌ Model file `sgpa_linear_model.pkl` not found. Please ensure it exists in the app directory.")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

# ---------------------------------------------------------
# Developer / Integration Note
# ---------------------------------------------------------
with st.expander("🛠️ Developer Guide: How to connect your ML Model"):
    st.code("""
# Example code to integrate your trained model in app.py:
import joblib

# 1. Load trained model artifact
model = joblib.load("sgpa_predictor_model.pkl")

# 2. Make prediction inside the predict block
predicted_sgpa = model.predict([[internal_marks]])[0]
st.success(f"Predicted SGPA: {predicted_sgpa:.2f}")
    """, language="python")