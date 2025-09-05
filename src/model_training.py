import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import sys
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with 'uv add shap' for model interpretability.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with 'uv add xgboost' for XGBoost training.")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with 'uv add lightgbm' for LightGBM training.")

import warnings
warnings.filterwarnings('ignore')

# Import local modules with error handling
try:
    from shap_analysis import SHAPAnalyzer
except ImportError:
    SHAPAnalyzer = None
    
try:
    from time_series_analysis import TimeSeriesAnalyzer
except ImportError:
    TimeSeriesAnalyzer = None

from data_preprocessing import load_data, clean_data, prepare_data

def train_enhanced_model(data_path='data/dataset_diabetes/diabetic_data.csv', 
                        model_type='random_forest', 
                        perform_shap_analysis=True,
                        save_models=True,
                        sample_size=10000):  # Add sample size parameter
    """Train and evaluate enhanced readmission prediction models with multiple algorithms"""
    
    print("=" * 60)
    print("ENHANCED HEALTHCARE READMISSION PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    print("\n1. Loading and preparing data...")
    # Load and prepare data with sampling for memory efficiency
    df = load_data(data_path)
    print(f"   Original dataset shape: {df.shape}")
    
    # Sample data for memory efficiency
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Sampled to {sample_size} records for memory efficiency")
    
    df_clean, preprocessor = clean_data(df)
    print(f"   Cleaned dataset shape: {df_clean.shape}")
    print(f"   Features after engineering: {len(df_clean.columns) - 1}")
    
    X, y = prepare_data(df_clean, preprocessor)
    print(f"   Final processed shape: {X.shape}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    print("\n2. Handling class imbalance with SMOTE (reduced sampling)...")
    # Handle class imbalance with SMOTE - use smaller k_neighbors for memory efficiency
    smote = SMOTE(random_state=42, k_neighbors=3)  # Reduce k_neighbors
    try:
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE: {X_train_res.shape[0]} samples")
        print(f"   New class distribution: {np.bincount(y_train_res)}")
    except Exception as e:
        print(f"   SMOTE failed due to memory: {e}")
        print("   Proceeding without SMOTE...")
        X_train_res, y_train_res = X_train, y_train
    
    # Train multiple models
    models = {}
    results = {}
    
    if model_type in ['random_forest', 'all']:
        print("\n3a. Training Random Forest model...")
        models['random_forest'] = train_random_forest(X_train_res, y_train_res, X_test, y_test)
        results['random_forest'] = evaluate_model(models['random_forest'], X_test, y_test, 'Random Forest')
    
    if model_type in ['xgboost', 'all'] and XGBOOST_AVAILABLE:
        print("\n3b. Training XGBoost model...")
        models['xgboost'] = train_xgboost(X_train_res, y_train_res, X_test, y_test)
        results['xgboost'] = evaluate_model(models['xgboost'], X_test, y_test, 'XGBoost')
    elif model_type in ['xgboost', 'all']:
        print("   Skipping XGBoost (not available)")
    
    if model_type in ['lightgbm', 'all'] and LIGHTGBM_AVAILABLE:
        print("\n3c. Training LightGBM model...")
        models['lightgbm'] = train_lightgbm(X_train_res, y_train_res, X_test, y_test)
        results['lightgbm'] = evaluate_model(models['lightgbm'], X_test, y_test, 'LightGBM')
    elif model_type in ['lightgbm', 'all']:
        print("   Skipping LightGBM (not available)")
    
    if model_type in ['logistic', 'all']:
        print("\n3d. Training Logistic Regression model...")
        models['logistic'] = train_logistic_regression(X_train_res, y_train_res, X_test, y_test)
        results['logistic'] = evaluate_model(models['logistic'], X_test, y_test, 'Logistic Regression')
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    best_model = models[best_model_name]
    
    print(f"\n4. Best model: {best_model_name} (AUC: {results[best_model_name]['roc_auc']:.4f})")
    
    # SHAP Analysis
    if perform_shap_analysis and SHAP_AVAILABLE and SHAPAnalyzer:
        print("\n5. Performing SHAP analysis...")
        try:
            shap_analyzer = SHAPAnalyzer(
                model=best_model, 
                X_train=X_train_res, 
                X_test=X_test,
                feature_names=preprocessor.get_feature_names_out()
            )
            
            # Create explainer and calculate SHAP values
            shap_analyzer.create_explainer('tree')
            shap_analyzer.calculate_shap_values(use_test_set=True, max_samples=500)
            
            # Get feature importance
            feature_importance = shap_analyzer.get_feature_importance(top_k=20)
            print("\n   Top 10 most important features (SHAP):")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"   {idx+1:2d}. {row['feature'][:50]:<50} {row['importance']:.4f}")
            
            # Save SHAP plots
            try:
                fig = shap_analyzer.plot_feature_importance(top_k=20)
                fig.write_html('shap_feature_importance.html')
                print("   SHAP feature importance plot saved as 'shap_feature_importance.html'")
            except Exception as e:
                print(f"   Warning: Could not save SHAP plots: {e}")
                
        except Exception as e:
            print(f"   Warning: SHAP analysis failed: {e}")
    elif perform_shap_analysis:
        print("\n5. Skipping SHAP analysis (dependencies not available)")
    
    # Time series analysis (if applicable)
    print("\n6. Performing temporal pattern analysis...")
    if TimeSeriesAnalyzer:
        try:
            ts_analyzer = TimeSeriesAnalyzer(df=df)
            patterns = ts_analyzer.analyze_readmission_patterns()
            
            if patterns.get('total_readmissions', 0) > 0:
                print(f"   Total readmission events analyzed: {patterns['total_readmissions']}")
                print(f"   30-day readmission rate: {patterns['readmission_rate_30d']:.2%}")
                print(f"   Average days between admissions: {patterns['avg_days_between_admissions']:.1f}")
            else:
                print("   No multiple admission patterns found in data")
                
        except Exception as e:
            print(f"   Warning: Time series analysis failed: {e}")
    else:
        print("   Skipping time series analysis (module not available)")
    
    # Save models
    if save_models:
        print("\n7. Saving models and preprocessor...")
        joblib.dump(best_model, 'model.pkl')
        joblib.dump(preprocessor, 'preprocessor.pkl')
        
        # Save model metadata
        metadata = {
            'model_type': best_model_name,
            'training_samples': len(X_train_res),
            'test_samples': len(X_test),
            'features': len(preprocessor.get_feature_names_out()),
            'performance': results[best_model_name],
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(metadata, 'model_metadata.pkl')
        print(f"   Best model ({best_model_name}) saved as 'model.pkl'")
        print("   Preprocessor saved as 'preprocessor.pkl'")
        print("   Metadata saved as 'model_metadata.pkl'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return best_model, preprocessor, results

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with simplified hyperparameter tuning"""
    
    # Simplified hyperparameter grid for memory efficiency
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=1)  # Use single job to save memory
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=2, scoring='roc_auc', n_jobs=1, verbose=1  # Reduce CV folds and jobs
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
        
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    return xgb_model

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
        
    lgb_model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    return lgb_model

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with regularization"""
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    grid_search = GridSearchCV(
        lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    
    print(f"\n   Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"   ROC-AUC Score: {roc_auc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return {
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': cm
    }

# Legacy function for backward compatibility
def train_model():
    """Legacy function - calls enhanced training with smaller sample for testing"""
    return train_enhanced_model(sample_size=2000)  # Further reduce sample size

if __name__ == "__main__":
    train_model()
