import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

# Enhanced Streamlit Configuration
st.set_page_config(
    page_title="Healthcare Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and preprocessor with error handling
@st.cache_resource
def load_models():
    """Load model and preprocessor with caching"""
    try:
        model = joblib.load('model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        # Load metadata if available
        try:
            metadata = joblib.load('model_metadata.pkl')
        except:
            metadata = None
            
        return model, preprocessor, metadata
    except FileNotFoundError:
        st.error("Model files not found. Please run model training first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, preprocessor, model_metadata = load_models()

# Enhanced App Header
st.title('🏥 Healthcare Readmission Risk Prediction Dashboard')
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
<h3 style='color: #1f77b4; margin-top: 0;'>Intelligent Clinical Decision Support System</h3>
<p style='margin-bottom: 0;'>Predict 30-day readmission risk for diabetic patients using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Display model information if available
if model_metadata:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model Type", 
            value=model_metadata.get('model_type', 'Unknown').replace('_', ' ').title()
        )
    
    with col2:
        st.metric(
            label="Training Samples", 
            value=f"{model_metadata.get('training_samples', 0):,}"
        )
    
    with col3:
        auc_score = model_metadata.get('performance', {}).get('roc_auc', 0)
        st.metric(
            label="Model AUC Score", 
            value=f"{auc_score:.3f}" if auc_score else "N/A"
        )
    
    with col4:
        st.metric(
            label="Features Count", 
            value=model_metadata.get('features', 0)
        )

def create_patient_input():
    """Create patient input interface"""
    st.subheader('Patient Information')
    
    # Demographics
    race = st.selectbox('Race', 
        ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'Unknown'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 0, 100, 50)

    # Admission details
    st.subheader('Admission Details')
    admission_type_map = {
        'Emergency': 1, 'Urgent': 2, 'Elective': 3, 
        'Newborn': 4, 'Not Available': 5
    }
    admission_type_choice = st.selectbox('Admission Type', list(admission_type_map.keys()))
    admission_type_id = admission_type_map[admission_type_choice]

    time_in_hospital = st.slider('Time in Hospital (days)', 1, 14, 5)
    
    # Clinical measurements
    st.subheader('Clinical Measurements')
    num_procedures = st.slider('Number of Procedures', 0, 10, 1)
    num_medications = st.slider('Number of Medications', 1, 80, 20)
    num_diagnoses = st.slider('Number of Diagnoses', 1, 16, 9)
    
    # Lab results
    st.subheader('Laboratory Results')
    a1c_result = st.selectbox('A1C Result', ['None', 'Normal', '>7', '>8'])
    glucose_serum = st.selectbox('Max Glucose Serum', ['None', 'Normal', '>200', '>300'])
    
    # Medications
    st.subheader('Medications')
    diabetes_med = st.selectbox('Diabetes Medication', ['Yes', 'No'])
    change = st.selectbox('Medication Changed', ['No', 'Ch'])
    insulin = st.selectbox('Insulin', ['No', 'Up', 'Down', 'Steady'])
    metformin = st.selectbox('Metformin', ['No', 'Up', 'Down', 'Steady'])
    
    # Create dataframe for prediction
    data = {
        'race': race, 'gender': gender, 'age': age,
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': 1,
        'admission_source_id': 7,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': 50,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': 0, 'number_emergency': 0, 'number_inpatient': 0,
        'diag_1': '250', 'diag_2': '0', 'diag_3': '0',
        'number_diagnoses': num_diagnoses,
        'max_glu_serum': glucose_serum,
        'A1Cresult': a1c_result,
        'change': change, 'diabetesMed': diabetes_med,
        'insulin': insulin, 'metformin': metformin,
    }

    # Add default medication columns
    med_columns = [
        'repaglinide','nateglinide','chlorpropamide',
        'glimepiride','acetohexamide','glipizide','glyburide',
        'tolbutamide','pioglitazone','rosiglitazone','acarbose',
        'miglitol','troglitazone','tolazamide','examide',
        'citoglipton','glyburide-metformin',
        'glipizide-metformin','glimepiride-pioglitazone',
        'metformin-rosiglitazone','metformin-pioglitazone'
    ]
    for col in med_columns:
        data[col] = 'No'

    return pd.DataFrame(data, index=[0])

def make_prediction(input_df):
    """Make prediction and display results"""
    if model is None or preprocessor is None:
        st.error("Model not loaded. Please check model files.")
        return
    
    try:
        # Align with expected columns
        expected_cols = preprocessor.feature_names_in_
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0 if col.startswith(('num_', 'number_')) or col.endswith('_id') else 'No'
        input_df = input_df.reindex(columns=expected_cols, fill_value='No')

        # Preprocess and predict
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        # Display results
        st.subheader('🎯 Risk Assessment Results')
        
        risk_probability = prediction_proba[0][1]
        if prediction[0] == 1:
            st.error(f"🚨 **HIGH RISK** of 30-day readmission")
            risk_color = "red"
        else:
            st.success(f"✅ **LOW RISK** of 30-day readmission")
            risk_color = "green"
        
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_probability * 100,
            title = {'text': "Readmission Risk (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}]
            }
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Detailed metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Risk Probability", f"{risk_probability:.1%}")
        with col_m2:
            st.metric("Confidence", f"{max(prediction_proba[0]):.1%}")
        with col_m3:
            st.metric("Risk Category", "High" if prediction[0] == 1 else "Low")
        
        # Feature importance
        if hasattr(model, "feature_importances_"):
            st.subheader('📊 Model Feature Importance')
            importances = model.feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True).tail(15)
            
            fig_imp = px.bar(
                importance_df, 
                x='importance', 
                y='feature',
                orientation='h',
                title="Top 15 Most Important Features"
            )
            fig_imp.update_layout(height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs([
    "🩺 Risk Prediction", 
    "ℹ️ Model Information",
    "🔗 API Access"
])

with tab1:
    st.header("Individual Patient Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_df = create_patient_input()
    
    with col2:
        make_prediction(input_df)

with tab2:
    st.header("ℹ️ Model Information & Performance")
    
    if model_metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Information")
            st.write(f"- **Model Type:** {model_metadata.get('model_type', 'Unknown')}")
            st.write(f"- **Training Date:** {model_metadata.get('training_date', 'Unknown')}")
            st.write(f"- **Training Samples:** {model_metadata.get('training_samples', 0):,}")
            st.write(f"- **Features:** {model_metadata.get('features', 0)}")
        
        with col2:
            if 'performance' in model_metadata:
                perf = model_metadata['performance']
                st.subheader("Model Performance")
                st.write(f"- **ROC-AUC Score:** {perf.get('roc_auc', 0):.4f}")
                
                if 'confusion_matrix' in perf:
                    cm = perf['confusion_matrix']
                    fig_cm = px.imshow(
                        cm, 
                        text_auto=True, 
                        title="Confusion Matrix",
                        x=['Low Risk', 'High Risk'],
                        y=['Low Risk', 'High Risk']
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("No model metadata available. Please retrain the model.")

with tab3:
    st.header("🔗 API Integration")
    st.info("FastAPI endpoint available for real-time predictions.")
    
    st.subheader("Quick Start")
    st.code("uv run uvicorn api:app --reload", language="bash")
    
    st.subheader("API Endpoints")
    st.write("- **Health Check:** GET `/health`")
    st.write("- **Prediction:** POST `/predict`")
    st.write("- **Model Info:** GET `/model/info`")
    st.write("- **Example Data:** GET `/predict/example`")
    
    st.subheader("Example Usage")
    st.code("""
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
patient_data = {
    "race": "Caucasian",
    "gender": "Male", 
    "age": 65,
    "admission_type_id": 1,
    "time_in_hospital": 5,
    "num_procedures": 1,
    "num_medications": 20,
    "number_diagnoses": 9
}

response = requests.post("http://localhost:8000/predict", json=patient_data)
print(response.json())
    """, language="python")

# Sidebar
if __name__ == "__main__":
    st.sidebar.markdown("### 🏥 Healthcare AI Dashboard")
    st.sidebar.info("Advanced machine learning for clinical decision support.")
    
    st.sidebar.subheader("Quick Actions")
    if st.sidebar.button("Test Model"):
        if model is not None:
            st.sidebar.success("✅ Model is loaded and ready!")
        else:
            st.sidebar.error("❌ Model not loaded")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 1.0.0")
    st.sidebar.markdown("**Updated:** 2024")