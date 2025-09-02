import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Load dataset from CSV file"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean and preprocess the dataset"""
    # Drop irrelevant columns
    df = df.drop(columns=[
        'encounter_id', 'patient_nbr', 'weight', 'payer_code', 
        'medical_specialty', 'max_glu_serum', 'A1Cresult'
    ])
    
    # Convert target to binary: readmitted <30 days = 1, else 0
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # Handle missing values
    df['race'] = df['race'].fillna('Unknown')
    df['diag_1'] = df['diag_1'].fillna('0')
    df['diag_2'] = df['diag_2'].fillna('0')
    df['diag_3'] = df['diag_3'].fillna('0')
    
    # Map age to numerical values
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)
    
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('readmitted')
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    return df, preprocessor

def prepare_data(df, preprocessor):
    """Prepare data for modeling"""
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y