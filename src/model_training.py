import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import sys
from tqdm import tqdm
import time
# sys.path.append(r"C:\Users\kexma\code\healthcare_readmission\src")
from data_preprocessing import load_data, clean_data, prepare_data

def train_model():
    """Train and evaluate the readmission prediction model"""
    print("Loading and preparing data...")
    # Load and prepare data
    df = load_data('data\dataset_diabetes\diabetic_data.csv')
    df_clean, preprocessor = clean_data(df)
    X, y = prepare_data(df_clean, preprocessor)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Handling class imbalance with SMOTE...")
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("Training Random Forest model...")
    # Use simpler model for faster training
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        verbose=0  # Disable sklearn verbose to use our own progress tracking
    )
    
    # Train with custom progress tracking
    print("Training progress:")
    start_time = time.time()
    
    # Fit the model
    rf.fit(X_train_res, y_train_res)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("Making predictions...")
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print("\nBest Parameters:", {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2})
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Save model and preprocessor
    print("Saving model and preprocessor...")
    joblib.dump(rf, '../model.pkl')
    joblib.dump(preprocessor, '../preprocessor.pkl')
    
    print("Training completed successfully!")
    return rf, preprocessor

if __name__ == "__main__":
    train_model()
