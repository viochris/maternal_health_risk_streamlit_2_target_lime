# ==============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# ==============================================================================
import streamlit as st
import requests
import datetime
import streamlit.components.v1 as components

# Importing the core machine learning and data processing pipelines from the external module
# This keeps the main UI file clean and separates the frontend from the backend logic
from function import prepare_data, predict, explain

# ==============================================================================
# 2. STREAMLIT PAGE CONFIGURATION & UI SETUP
# ==============================================================================
# Must be the very first Streamlit command executed to configure the browser tab and layout
st.set_page_config(
    page_title="Maternal Health AI Predictor", 
    page_icon="🩺", 
    layout="centered"
)

# Custom CSS & Hero Section for a Modern Medical Dashboard Look
# We inject raw HTML/CSS to bypass Streamlit's default styling constraints 
# and create a premium, clinical-themed gradient aesthetic.
st.markdown(
    """
    <style>
    /* Main Container */
    .hero-container { 
        text-align: center; 
        padding-bottom: 2rem; 
    }

    /* Gradient Title - Clinical / Health Theme */
    .gradient-text { 
        font-size: 2.8rem; 
        font-weight: 800; 
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0.5rem; 
    }

    /* Sub-hook styling */
    .sub-hook { 
        font-size: 1.2rem; 
        font-weight: 500; 
        color: #A0AEC0; 
        margin-bottom: 2rem; 
    }

    /* Description Box with Medical Cyan Accent Border */
    .description-box { 
        background-color: #1E1E2E; 
        padding: 1.5rem 2rem; 
        border-radius: 8px; 
        border-left: 4px solid #00C9FF; 
        text-align: left; 
        font-size: 1rem; 
        line-height: 1.6; 
        color: #E2E8F0; 
        margin-top: 1.5rem; 
    }
    </style>

    <div class="hero-container">
        <div class="gradient-text">🩺 Maternal Health Dashboard</div>
        <div class="sub-hook">AI-powered risk assessment for pregnancy health.</div>
        <div class="description-box">
            Input the patient's clinical metrics below. Our Machine Learning model will 
            evaluate the data to predict potential maternal health risks, ensuring timely and accurate medical insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 3. PATIENT DATA INPUT FORM
# ==============================================================================
# Using st.form ensures that the UI does NOT trigger a full script rerun every time 
# a user adjusts a single number. The model will only execute when the submit button is clicked.
with st.form("maternal_health_form"):
    st.markdown("### 🏥 Patient Clinical Metrics")

    # Split the input form into two visually balanced columns to save vertical screen space
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            label="Patient Age (Years)",
            value=25,
            min_value=10,
            max_value=70,
            help="Enter the patient's age in years."
        )

        systolic_bp = st.number_input(
            label="Systolic BP (mmHg)",
            value=120,
            min_value=70,
            max_value=160,
            help="Upper blood pressure metric (pressure in arteries when the heart beats)."
        )

        diastolic_bp = st.number_input(
            label="Diastolic BP (mmHg)",
            value=80,
            min_value=49, 
            max_value=100,
            help="Lower blood pressure metric (pressure in arteries when the heart rests)."
        )

    with col2:
        blood_glucose = st.number_input(
            label="Blood Glucose (mmol/L)",
            value=7.5,
            min_value=6.0,
            max_value=19.0,
            help="Patient's blood sugar level."
        )

        body_temp = st.number_input(
            label="Body Temperature (°F)",
            value=98.6,
            min_value=98.0,
            max_value=103.0,
            help="Core body temperature measured in Fahrenheit."
        )

        heart_rate = st.number_input(
            label="Heart Rate (BPM)",
            value=75,
            min_value=60, 
            max_value=90,
            help="Resting heart rate in beats per minute."
        )

    # Form submission button that triggers the ML pipeline below
    submitted = st.form_submit_button("Analyze Health Risk 🚀", use_container_width=True)

if submitted:
    try:
        # ---------------------------------------------------------
        # 1. FETCH PREDICTION (LOCAL INFERENCE)
        # ---------------------------------------------------------
        # Using st.spinner to provide visual feedback while the CPU processes the data
        with st.spinner("🤖 AI is analyzing patient metrics for health risk..."):
            
            # Construct the DataFrame using the imported function
            df_testing = prepare_data(
                age=age,
                systolic_bp=systolic_bp, 
                diastolic_bp=diastolic_bp, 
                blood_glucose=blood_glucose, 
                body_temp=body_temp, 
                heart_rate=heart_rate
            )
            
            # Safely halt if DataFrame construction fails
            if df_testing is None:
                st.error("🚨 **[DATA ERROR]** Failed to prepare the input metrics for prediction.")
                st.stop()

            # Execute the ML prediction locally
            prediction, prediction_conf, low_risk_score, elevated_risk_score = predict(df_testing)

            # Validate that all required metrics were successfully returned
            if not all([prediction, prediction_conf, low_risk_score is not None, elevated_risk_score is not None]):
                st.error("🚨 **[MODEL ERROR]** The model returned incomplete prediction metrics.")
                st.stop()

            low_risk_score_percentage = (low_risk_score * 100).round(2)
            elevated_risk_score_percentage = (elevated_risk_score * 100).round(2)

            # Display the main results in a modern, customized HTML card (Medical Cyan Theme)
            st.markdown(
                f"""
                <div style='background-color: #1E1E1E; padding: 25px; border-radius: 12px; border-left: 6px solid #00C9FF; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-top: 20px;'>
                    <h3 style='color: #00C9FF; margin-top: 0; font-family: sans-serif;'>🎉 Analysis Complete!</h3>
                    <p style='font-size: 16px; color: #E0E0E0; margin-bottom: 5px; font-family: sans-serif;'>Predicted Risk Level:</p>
                    <p style='font-size: 32px; font-weight: bold; color: #92FE9D; margin-top: 0; margin-bottom: 15px; font-family: monospace; text-transform: uppercase;'>
                        {prediction}
                    </p>
                    <hr style='border-color: #333333;'>
                    <p style='color: #A0A0A0; margin-bottom: 5px; font-size: 14px; font-family: sans-serif;'>
                        📊 <strong>Probability Breakdown:</strong> Low: {low_risk_score_percentage}% | Elevated: {elevated_risk_score_percentage}%
                    </p>
                    <p style='color: #A0A0A0; margin-bottom: 0; font-size: 14px; font-family: sans-serif;'>
                        🤖 <strong>Model Confidence Score:</strong> {prediction_conf}
                    </p>
                </div>
                <br>
                """,
                unsafe_allow_html=True
            )

            # Render visual progress bars for probability distribution
            # Streamlit progress bars accept float values between 0.0 and 1.0
            st.progress(low_risk_score, text=f"Low Risk Probability: {low_risk_score_percentage}%")
            st.progress(elevated_risk_score, text=f"Elevated Risk Probability: {elevated_risk_score_percentage}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 2. FETCH LIME EXPLANATION (LOCAL GENERATION)
        # ---------------------------------------------------------
        with st.spinner("🔍 AI is generating the reasoning behind this prediction..."):
            
            # Re-prepare data for isolated LIME generation
            df_testing_lime = prepare_data(
                age=age, 
                systolic_bp=systolic_bp, 
                diastolic_bp=diastolic_bp, 
                blood_glucose=blood_glucose, 
                body_temp=body_temp, 
                heart_rate=heart_rate
            )
            
            if df_testing_lime is None:
                st.error("🚨 **[DATA ERROR]** Failed to prepare the input metrics for LIME explanation.")
                st.stop()

            # Execute the local XAI function
            html_data, explanation_figure, feature_weights = explain(df_testing_lime)

            if not html_data:
                st.error("🚨 **[EXPLANATION ERROR]** The model failed to generate a LIME HTML explanation.")
                st.stop()

            # Wrap the raw HTML inside a premium custom container using inline CSS
            lime_html_with_bg = f"""
            <div style='background-color: #1E1E1E; padding: 25px; border-radius: 12px; 
                        box-shadow: 0 8px 16px rgba(0,0,0,0.5); border-top: 6px solid #00C9FF; 
                        font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;'>
                <style>
                    /* Override LIME's default black text elements for dark mode compatibility */
                    svg text {{ fill: #E0E0E0 !important; }}
                    table {{ color: #E0E0E0 !important; }}
                    .lime-table th, .lime-table td {{ border-color: #444444 !important; }}
                </style>
                <h2 style='color: #FFFFFF; margin-top: 0; margin-bottom: 5px; font-weight: 700;'>
                    🧠 AI Decision Breakdown
                </h2>
                <p style='color: #A0AEC0; font-size: 15px; margin-top: 0; margin-bottom: 25px; font-weight: 500;'>
                    A transparent view of which specific health metrics positively or negatively impacted the risk prediction.
                </p>
                {html_data}
            </div>
            """
            
            # Render the generated LIME HTML output directly inside the Streamlit UI
            components.html(lime_html_with_bg, height=850, scrolling=True)

            # Render the static Matplotlib figure generated by LIME
            st.pyplot(explanation_figure)

            st.markdown("### ⚖️ Feature Impact Weights")
            for feature, weight in feature_weights:
                st.markdown(f"- **{feature}**: `{weight:+.2f}`")

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (LOCAL ML EXECUTION)
    # ---------------------------------------------------------
    # Replaced network errors with local processing errors since there is no API involved
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        st.error("💥 **[CRITICAL FAILURE]** Process aborted during execution!")

        # 1. Handling Missing Data/Keys
        if error_type == "KeyError" or "key" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** A required data field is missing. Details: `{error_raw}`")
            st.stop()
        
        # 2. Handling Data Type Mismatches
        elif error_type == "TypeError" or "type" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Incorrect data type passed to the model. Details: `{error_raw}`")
            st.stop()
        
        # 3. Handling Value/Shape Mismatch
        elif error_type == "ValueError" or "value" in error_msg or "shape" in error_msg:
            st.error(f"🚨 **[MODEL ERROR] {error_type}:** The input data shape or value does not match the model's requirements. Details: `{error_raw}`")
            st.stop()
        
        # 4. Handling Corrupted Model Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            st.error(f"🚨 **[SYSTEM ERROR] {error_type}:** Internal model architecture error. Details: `{error_raw}`")
            st.stop()

        # 5. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            st.error(f"🚨 **[MODEL ERROR] {error_type}:** The loaded machine learning model is not trained. Details: `{error_raw}`")
            st.stop()
        
        # 6. Fallback for any other UI/Local errors
        else:
            st.error(f"🚨 **[UNKNOWN ERROR] {error_type}:** An unexpected system error occurred during execution. Details: `{error_raw}`")
            st.stop()

# ==============================================================================
# FOOTER CONFIGURATION
# ==============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown(
    f"""
    <div style='text-align: center; color: #A0AEC0; font-size: 14px; font-family: sans-serif; padding-bottom: 20px;'>
        <p style='margin-bottom: 8px;'>
            🛠️ <strong>Built With:</strong> Frontend UI (Streamlit) | Backend API (FastAPI) | Machine Learning (Scikit-Learn) | Explainable AI (LIME) | Data Processing (Pandas & Numpy)
        </p>
        <p style='margin-bottom: 8px;'>
            👨‍💻 Developed by <strong>Silvio Christian Joe</strong> &nbsp;|&nbsp; <a href='https://github.com/viochris' target='_blank' style='color: #00C9FF; text-decoration: none;'>GitHub (@viochris)</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)