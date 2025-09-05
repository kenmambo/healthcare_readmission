import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load dataset from CSV file"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean and preprocess the dataset with enhanced clinical features"""
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Drop irrelevant columns but keep more clinical information
    df = df.drop(columns=[
        'encounter_id', 'patient_nbr', 'weight', 'payer_code', 
        'medical_specialty'  # Keeping A1Cresult and max_glu_serum for clinical features
    ])
    
    # Convert target to binary: readmitted <30 days = 1, else 0
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # Map age to numerical values first, before feature engineering
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)
    
    # Enhanced clinical feature engineering
    df = engineer_clinical_features(df)
    
    # Handle missing values with domain-specific strategies
    df['race'] = df['race'].fillna('Unknown')
    df['diag_1'] = df['diag_1'].fillna('0')
    df['diag_2'] = df['diag_2'].fillna('0')
    df['diag_3'] = df['diag_3'].fillna('0')
    
    # Fill A1C and glucose results with 'None' if missing
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].fillna('None')
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
    
    # Map age to numerical values (this was moved earlier in the process)
    # age_map = {
    #     '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    #     '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    #     '[80-90)': 85, '[90-100)': 95
    # }
    # df['age'] = df['age'].map(age_map)
    
    # Identify numerical and categorical columns more carefully
    # Ensure we only treat truly numeric columns as numeric (not ID columns)
    numeric_column_names = [
        'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'total_med_changes', 'total_visits', 'high_procedure_count',
        'long_stay', 'high_med_count', 'multiple_diagnoses', 'emergency_admission',
        'insulin_user', 'diabetesMed_user', 'elderly_patient'
    ]
    
    # Only include columns that exist and are actually numeric (exclude ID columns)
    num_cols = [col for col in numeric_column_names if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    # All other columns (except target) are categorical - specifically include ID columns
    cat_cols = [col for col in df.columns if col not in num_cols and col != 'readmitted']
    
    # Create preprocessing pipelines with feature selection
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Debug: Print column information
    print(f"   Numeric columns ({len(num_cols)}): {num_cols[:5]}..." if len(num_cols) > 5 else f"   Numeric columns: {num_cols}")
    print(f"   Categorical columns ({len(cat_cols)}): {cat_cols[:5]}..." if len(cat_cols) > 5 else f"   Categorical columns: {cat_cols}")
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')
    
    return df, preprocessor

def engineer_clinical_features(df):
    """Engineer additional clinical features from existing data"""
    # Total medication count (sum of all medication changes)
    med_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    
    # Count medications that were changed or started
    df['total_med_changes'] = 0
    for col in med_columns:
        if col in df.columns:
            df['total_med_changes'] += (df[col] != 'No').astype(int)
    
    # Healthcare utilization intensity
    df['total_visits'] = (df['number_outpatient'] + 
                         df['number_emergency'] + 
                         df['number_inpatient'])
    
    # Severity indicators
    df['high_procedure_count'] = (df['num_procedures'] > df['num_procedures'].median()).astype(int)
    df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
    df['high_med_count'] = (df['num_medications'] > df['num_medications'].median()).astype(int)
    
    # Diagnosis complexity
    df['multiple_diagnoses'] = (df['number_diagnoses'] > 5).astype(int)
    
    # Emergency indicators
    df['emergency_admission'] = (df['admission_type_id'].isin([1, 2])).astype(int)
    
    # Medication management indicators
    df['insulin_user'] = 0
    if 'insulin' in df.columns:
        df['insulin_user'] = (df['insulin'] != 'No').astype(int)
    
    df['diabetesMed_user'] = 0
    if 'diabetesMed' in df.columns:
        df['diabetesMed_user'] = (df['diabetesMed'] == 'Yes').astype(int)
    
    # Age-based risk categories
    df['elderly_patient'] = (df['age'] >= 65).astype(int)
    
    return df

def prepare_data(df, preprocessor):
    """Prepare data for modeling"""
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y