import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# Import modular backend handlers
from model_handler import load_model, predict_sgpa
from data_manager import load_dataset, get_student_by_id, batch_predict_df
from evaluator import compute_eval_metrics, plot_actual_vs_predicted, plot_residuals, plot_error_distribution

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="University SGPA Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS Styling (Dark Glassmorphism Theme)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Global Base */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Custom Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px -5px rgba(67, 56, 202, 0.35);
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        text-align: center;
    }
    .header-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #c7d2fe;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        display: inline-block;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }
    .header-title {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #a5b4fc;
        font-size: 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Content Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(12px);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    /* Performance Badges */
    .val-badge {
        padding: 0.65rem 1.1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.92rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .val-outstanding { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .val-excellent { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); }
    .val-good { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3); }
    .val-average { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .val-risk { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    /* Prediction Output Display Card */
    .result-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }
    .result-score {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.4rem 0;
        line-height: 1.1;
    }
    .grade-chip {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.4);
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* Custom Metric Display Box */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Dataset Resources (Cached)
# ---------------------------------------------------------
@st.cache_resource
def get_cached_model():
    return load_model()

@st.cache_data
def get_cached_data():
    return load_dataset("data.csv")

try:
    model, scaler, model_source = get_cached_model()
except Exception as e:
    st.error(f"Error initializing Machine Learning model: {e}")
    st.stop()

dataset_df = get_cached_data()
predicted_dataset_df = batch_predict_df(dataset_df, model, scaler)

# ---------------------------------------------------------
# Sidebar Section
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 SGPA Predictor")
    st.caption("Linear Regression Assessment System")
    st.divider()

    st.markdown("#### 📌 Active Model Info")
    st.info(f"**Loaded Artifact**: `{os.path.basename(model_source)}`\n\n**Feature**: Internal Mark (0.0 to 40.0)")

    st.markdown("#### 📊 Dataset Quick Stats")
    if not dataset_df.empty:
        st.write(f"• **Total Students**: {len(dataset_df)}")
        st.write(f"• **Avg Internal**: {dataset_df['AvgInternal'].mean():.2f} / 40.0")
        if "SGPA" in dataset_df.columns:
            st.write(f"• **Avg Actual SGPA**: {dataset_df['SGPA'].mean():.2f}")
    else:
        st.warning("No dataset loaded.")

    st.divider()
    st.markdown("#### ⚙️ Quick Actions")
    if st.button("🔄 Refresh Data & Model", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("© 2026 University ML Project Team")

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
    <div class="header-card">
        <div class="header-badge">ACADEMIC ML INFRASTRUCTURE</div>
        <h1 class="header-title">🎓 University SGPA Predictor</h1>
        <p class="header-subtitle">Estimate Semester Grade Point Average from Internal Assessment Marks (Out of 40)</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Navigation (3 Tabs)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Single Prediction",
    "🔍 Student Database",
    "📊 Performance Analytics"
])

# ---------------------------------------------------------
# TAB 1: Single Prediction
# ---------------------------------------------------------
with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Single Student Prediction")
    st.write("Enter the student's internal mark score (out of 40) to estimate their predicted SGPA.")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        internal_mark = st.number_input(
            "Internal Mark (0.0 to 40.0)",
            min_value=0.0,
            max_value=40.0,
            value=34.0,
            step=0.25,
            help="Enter internal assessment average score out of 40."
        )


    with col2:
        st.markdown("**Assessment Tier:**")
        if internal_mark >= 36.0:
            st.markdown('<div class="val-badge val-outstanding">🌟 Outstanding Performance (Top Tier)</div>', unsafe_allow_html=True)
        elif internal_mark >= 32.0:
            st.markdown('<div class="val-badge val-excellent">✨ Excellent Range</div>', unsafe_allow_html=True)
        elif internal_mark >= 28.0:
            st.markdown('<div class="val-badge val-good">👍 Good Standing</div>', unsafe_allow_html=True)
        elif internal_mark >= 24.0:
            st.markdown('<div class="val-badge val-average">⚠️ Average Range</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="val-badge val-risk">❗ Below Average / Risk Zone</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Trigger Prediction
    if st.button("🚀 Predict SGPA Output", type="primary", use_container_width=True):
        with st.spinner("Processing feature scaling & model inference..."):
            time.sleep(0.2)
            predicted_sgpa = predict_sgpa(internal_mark, model, scaler)

            # Determine grade letter
            if predicted_sgpa >= 9.0:
                grade_letter = "O (Outstanding)"
            elif predicted_sgpa >= 8.0:
                grade_letter = "A+ (Excellent)"
            elif predicted_sgpa >= 7.0:
                grade_letter = "A (Very Good)"
            elif predicted_sgpa >= 6.0:
                grade_letter = "B+ (Good)"
            elif predicted_sgpa >= 5.0:
                grade_letter = "B (Above Average)"
            elif predicted_sgpa >= 4.0:
                grade_letter = "C (Pass)"
            else:
                grade_letter = "F (Fail)"

            st.markdown(f"""
                <div class="result-card">
                    <span class="header-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4);">
                        ✅ MODEL INFERENCE SUCCESSFUL
                    </span>
                    <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 1rem; margin-bottom: 0;">Predicted SGPA Score</p>
                    <div class="result-score">{predicted_sgpa:.2f}</div>
                    <p style="color: #cbd5e1; font-size: 1rem; margin-bottom: 0.6rem;">
                        Internal Mark Input: <strong>{internal_mark:.2f} / 40.0</strong>
                    </p>
                    <div class="grade-chip">Expected Grade: {grade_letter}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("💡 View Target SGPA Guidance"):
                st.markdown("""
                - **To achieve SGPA 8.0+**: Aim for an internal score of **35.0 / 40.0** or higher.
                - **To achieve SGPA 9.0+**: Aim for an internal score of **38.0 / 40.0** or higher.
                - **Internal Weightage Notice**: Internal marks are a major predictor of semester grade outcomes.
                """)

# ---------------------------------------------------------
# TAB 2: Dataset & Student Database Explorer
# ---------------------------------------------------------
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Student Dataset Explorer")
    st.write("Search for student records by Student ID or filter dataset by internal mark range.")

    c1, c2 = st.columns([2, 2])
    with c1:
        search_query = st.text_input("🔍 Search Student ID (e.g., LPRP24CS131, PRP24CS078)", value="")
    with c2:
        mark_filter = st.slider("Filter by Internal Mark Range (out of 40)", 0.0, 40.0, (20.0, 40.0))

    st.markdown("</div>", unsafe_allow_html=True)

    # Process Search and Filter
    display_df = predicted_dataset_df.copy()

    if search_query.strip():
        display_df = get_student_by_id(display_df, search_query)

    if not display_df.empty:
        display_df = display_df[
            (display_df["AvgInternal"] >= mark_filter[0]) & 
            (display_df["AvgInternal"] <= mark_filter[1])
        ]

    st.markdown(f"**Showing {len(display_df)} student record(s):**")

    if not display_df.empty:
        st.dataframe(
            display_df.style.format({
                "AvgInternal": "{:.2f}",
                "SGPA": "{:.2f}" if "SGPA" in display_df.columns else None,
                "Predicted_SGPA": "{:.2f}",
                "Error": "{:.2f}" if "Error" in display_df.columns else None,
                "Abs_Error": "{:.2f}" if "Abs_Error" in display_df.columns else None,
            }),
            use_container_width=True,
            height=380
        )
    else:
        st.warning("No student records matched your search criteria.")

# ---------------------------------------------------------
# TAB 3: Model Evaluation & Performance Analytics
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📊 Model Evaluation & Metrics")
    metrics = compute_eval_metrics(predicted_dataset_df)

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val">{metrics['mae']}</div>
                    <div class="stat-lbl">Mean Absolute Error (MAE)</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val">{metrics['rmse']}</div>
                    <div class="stat-lbl">Root Mean Squared Error (RMSE)</div>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val">{metrics['r2']}</div>
                    <div class="stat-lbl">R² Score (Variance Explained)</div>
                </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val">{metrics['count']}</div>
                    <div class="stat-lbl">Evaluated Student Records</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plot Graphs
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### 📈 Actual vs. Predicted SGPA")
            fig1 = plot_actual_vs_predicted(predicted_dataset_df)
            st.pyplot(fig1)

        with col_g2:
            st.markdown("#### 📉 Residual Plot (Prediction Errors)")
            fig2 = plot_residuals(predicted_dataset_df)
            st.pyplot(fig2)

        st.markdown("---")
        col_g3, col_g4 = st.columns([1, 1])
        with col_g3:
            st.markdown("#### 📊 Error Magnitude Distribution")
            fig3 = plot_error_distribution(predicted_dataset_df)
            st.pyplot(fig3)

        with col_g4:
            st.markdown("#### ⚠️ Top Residual Outliers")
            if "Abs_Error" in predicted_dataset_df.columns:
                outliers = predicted_dataset_df.sort_values(by="Abs_Error", ascending=False).head(5)
                st.dataframe(
                    outliers[["StudentID", "AvgInternal", "SGPA", "Predicted_SGPA", "Error"]].style.format({
                        "AvgInternal": "{:.2f}",
                        "SGPA": "{:.2f}",
                        "Predicted_SGPA": "{:.2f}",
                        "Error": "{:.2f}"
                    }),
                    use_container_width=True
                )
    else:
        st.info("Evaluation metrics require actual SGPA column in dataset.")