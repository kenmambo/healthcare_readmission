import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessor
model = joblib.load('../model.pkl')
preprocessor = joblib.load('../preprocessor.pkl')

# App title
st.title('Hospital Readmission Risk Prediction')

# Sidebar for inputs
st.sidebar.header('Patient Information')

def user_input_features():
    """Create input widgets for patient data"""
    # Demographics
    race = st.sidebar.selectbox('Race', 
        ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 0, 100, 50)

    # Admission type mapping (numeric IDs used in dataset)
    admission_type_map = {
        'Emergency': 1,
        'Urgent': 2,
        'Elective': 3,
        'Newborn': 4,
        'Not Available': 5
    }
    admission_type_choice = st.sidebar.selectbox('Admission Type', list(admission_type_map.keys()))
    admission_type_id = admission_type_map[admission_type_choice]

    # Fixed discharge disposition + admission source (IDs from dataset)
    # You can expand this if you want multiple choices
    discharge_disposition_id = 1   # Discharged to home
    admission_source_id = 7        # Emergency Room

    time_in_hospital = st.sidebar.slider('Time in Hospital (days)', 1, 14, 5)
    num_procedures = st.sidebar.slider('Number of Procedures', 0, 10, 1)
    num_medications = st.sidebar.slider('Number of Medications', 1, 80, 20)

    # Diagnoses
    num_diagnoses = st.sidebar.slider('Number of Diagnoses', 1, 16, 9)

    # Medications
    diabetes_med = st.sidebar.selectbox('Diabetes Medication', ['Yes', 'No'])
    change = st.sidebar.selectbox('Medication Changed', ['No', 'Ch'])

    # Create dataframe
    data = {
        'race': race,
        'gender': gender,
        'age': age,
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': 50,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 0,
        'diag_1': '250',  # Diabetes
        'diag_2': '0',
        'diag_3': '0',
        'number_diagnoses': num_diagnoses,
        'change': change,
        'diabetesMed': diabetes_med,
    }

    # Medication columns default 'No'
    med_columns = [
        'metformin','repaglinide','nateglinide','chlorpropamide',
        'glimepiride','acetohexamide','glipizide','glyburide',
        'tolbutamide','pioglitazone','rosiglitazone','acarbose',
        'miglitol','troglitazone','tolazamide','examide',
        'citoglipton','insulin','glyburide-metformin',
        'glipizide-metformin','glimepiride-pioglitazone',
        'metformin-rosiglitazone','metformin-pioglitazone'
    ]
    for col in med_columns:
        data[col] = 'No'

    # Update insulin if diabetes meds are given
    if diabetes_med == 'Yes':
        data['insulin'] = 'Steady'

    return pd.DataFrame(data, index=[0])

# Get input
input_df = user_input_features()

# Align with expected columns from preprocessor
expected_cols = preprocessor.feature_names_in_
for col in expected_cols:
    if col not in input_df.columns:
        # If numeric -> 0, else -> 'No'
        input_df[col] = 0 if np.issubdtype(type(preprocessor.feature_names_in_[0]), np.number) else 'No'
input_df = input_df.reindex(columns=expected_cols)

# Display input data
st.subheader('Patient Information')
st.write(input_df)

# Preprocess input
processed_input = preprocessor.transform(input_df)

# Make prediction
prediction = model.predict(processed_input)
prediction_proba = model.predict_proba(processed_input)

# Display prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.error('High Risk of Readmission')
else:
    st.success('Low Risk of Readmission')

st.subheader('Prediction Probability')
st.write(f"Probability of readmission: {prediction_proba[0][1]:.2%}")

# Feature importance
st.subheader('Feature Importance')
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=preprocessor.get_feature_names_out()[indices])
    plt.title('Top 10 Important Features')
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.info("This model does not provide feature importances.")
