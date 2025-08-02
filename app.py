import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import tempfile
from datetime import datetime

# Load model and metadata
rf_model = joblib.load('models/rf_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# encode categorical inputs
def encode_input(input_dict):
    df = pd.DataFrame([input_dict])
    for col in encoders:
        df[col] = encoders[col].transform(df[col])
    return df

# Prediction logic
def predict_hypertension(input_data):
    input_df = encode_input(input_data)
    input_df = input_df[feature_columns]
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0][1]
    return prediction, probability

def get_risk_category(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"

def get_recommendation(category):
    if category == "Low":
        return "Maintain your current lifestyle and routine checkups."
    elif category == "Moderate":
        return "Consider lifestyle improvements and consult your doctor."
    else:
        return "Urgent medical consultation recommended. Follow prescribed treatments."

def colored_progress_bar(prob, color):
    percent = int(prob * 100)
    bar_html = f"""
    <div style="background-color:#ddd; border-radius:5px; padding:3px; width:100%; margin-bottom:10px;">
      <div style="width:{percent}%; background-color:{color}; height:20px; border-radius:5px;"></div>
    </div>
    <p style="text-align:center; margin:0;">{percent}%</p>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

st.title("🩺 Hypertension Risk Predictor")

# Customized CSS
st.markdown(
    """
    <style>
        /* App background */
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }

        /* Headers & labels */
        h1, h2, h3, h4, h5, h6, label {
            color: #1B4965 !important;
        }

        /* Sliders */
        .stSlider > div > div > div > div {
            background: linear-gradient(to right, #6EEB83, #1B98E0) !important;
        }

        /* Selectboxes & radios */
        .stSelectbox, .stRadio {
            background-color: #f0f9ff !important;
            border-radius: 5px !important;
            padding: 5px !important;
        }

        /* Info box (st.info, st.warning, etc.) */
        div[role="alert"] {
            color: #2780F5 !important;
            font-weight: bold !important;
            background-color: #f0f9ff !important;
            border: 1px solid #1B98E0 !important;
            border-radius: 15px !important;
        }

        /* Predict button */
        div.stButton > button {
            background-color: #2780F5 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
        }
        div.stButton > button:hover {
            background-color: #1a5fb4 !important;
            color: white !important;
        }

        /* Download PDF button */
        div.stDownloadButton > button {
            background-color: #2780F5 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
        }
        div.stDownloadButton > button:hover {
            background-color: #1a5fb4 !important;
            color: white !important;
        }

        /* Expander headers */
        .streamlit-expanderHeader {
            background-color: #e6f2ff !important;
            color: #1B4965 !important;
            font-weight: bold !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# User inputs
with st.expander("Personal Info", expanded=True):
    age = st.slider("Age", 18, 100, 50, help="Your age in years")
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, help="Body Mass Index")

with st.expander("Lifestyle Factors", expanded=True):
    salt_intake = st.slider("Salt Intake (grams)", 0.0, 20.0, 10.0, help="Daily salt consumption")
    exercise_level = st.selectbox("Exercise Level", encoders['Exercise_Level'].classes_, help="How active you are")
    smoking_status = st.selectbox("Smoking Status", encoders['Smoking_Status'].classes_, help="Do you smoke?")
    medication = st.selectbox("Medication", encoders['Medication'].classes_, help="Current hypertension medication")

with st.expander("Health History", expanded=True):
    bp_history = st.selectbox("Blood Pressure History", encoders['BP_History'].classes_, help="Your past blood pressure diagnosis")
    family_history = st.selectbox("Family History of Hypertension", encoders['Family_History'].classes_, help="Hypertension in family?")
    stress_score = st.slider("Stress Score (0-10)", 0, 10, 5, help="Your average stress level")
    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, help="Avg. sleep per night")

# Gather inputs
input_features = {
    'Age': age,
    'Salt_Intake': salt_intake,
    'Stress_Score': stress_score,
    'BP_History': bp_history,
    'Sleep_Duration': sleep_duration,
    'BMI': bmi,
    'Medication': medication,
    'Family_History': family_history,
    'Exercise_Level': exercise_level,
    'Smoking_Status': smoking_status,
}

# check if inputs have changed
def inputs_changed(new_inputs):
    if 'prev_inputs' not in st.session_state:
        return True
    return new_inputs != st.session_state.prev_inputs

# Initialize session state
if st.button("Predict Hypertension Risk"):
    if inputs_changed(input_features) or 'predicted' not in st.session_state:
        pred, proba = predict_hypertension(input_features)
        st.session_state.prediction = pred
        st.session_state.proba = proba
        st.session_state.prev_inputs = input_features
        st.session_state.predicted = True

# If prediction exists, show results
if st.session_state.get("predicted", False):
    pred = st.session_state.prediction
    proba = st.session_state.proba

    risk_cat = get_risk_category(proba)
    recommendation = get_recommendation(risk_cat)

    if risk_cat == "Low":
        color = "green"
        st.markdown(f"<h3 style='color:green;'>Prediction: No Hypertension (Low Risk)</h3>", unsafe_allow_html=True)
    elif risk_cat == "Moderate":
        color = "orange"
        st.markdown(f"<h3 style='color:orange;'>Prediction: {'Hypertension' if pred == 1 else 'No Hypertension'} (Moderate Risk)</h3>", unsafe_allow_html=True)
    else:
        color = "red"
        st.markdown(f"<h3 style='color:red;'>Prediction: {'Hypertension' if pred == 1 else 'No Hypertension'} (High Risk)</h3>", unsafe_allow_html=True)

    colored_progress_bar(proba, color)
    st.info(recommendation)

    # Current time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Report header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Hypertension Risk Report", ln=True, align='C')
    pdf.ln(10)

    # pdf timestamp
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Generated on: {timestamp}", ln=True)
    pdf.ln(5)

    # Section: user inputs
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Personal & Health Info", ln=True)
    pdf.set_font("Arial", size=12)

    info_lines = [
        f"Age: {age}",
        f"BMI: {bmi}",
        f"Salt Intake: {salt_intake} grams",
        f"Stress Score: {stress_score}",
        f"Sleep Duration: {sleep_duration} hours",
        f"BP History: {bp_history}",
        f"Medication: {medication}",
        f"Family History: {family_history}",
        f"Exercise Level: {exercise_level}",
        f"Smoking Status: {smoking_status}",
    ]
    for line in info_lines:
        pdf.multi_cell(0, 10, line)

    # Section: prediction
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Prediction Result", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Risk Probability: {proba:.2f}")
    pdf.multi_cell(0, 10, f"Risk Category: {risk_cat}")
    pdf.multi_cell(0, 10, f"Recommendation: {recommendation}")

    # footer
    pdf.ln(5)
    pdf.set_font("Arial", style="", size=11)
    pdf.multi_cell(0, 10, "Thank you for using the Hypertension Risk Predictor!")

    # website link
    pdf.ln(5)
    website_url = "https://viduraabeysinghe.netlify.app/"
    pdf.set_text_color(0, 0, 255)
    pdf.set_font("Arial", style="U", size=11)
    pdf.cell(0, 10, "Visit my website", ln=True, link=website_url)
    pdf.set_text_color(0, 0, 0)

    # Save PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        with open(tmpfile.name, "rb") as file:
            st.download_button(
                label="Download PDF Report",
                data=file,
                file_name="hypertension_report.pdf",
                mime="application/pdf"
            )

    st.markdown("<br>", unsafe_allow_html=True)
