"""
SHAP (SHapley Additive exPlanations) analysis module for model interpretability.
Provides feature importance and model explanation capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class SHAPAnalyzer:
    """
    Provides SHAP-based model interpretability and feature importance analysis.
    """
    
    def __init__(self, model=None, X_train=None, X_test=None, feature_names=None):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data 
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.analyzed_data = None
        
    def create_explainer(self, explainer_type='tree'):
        """
        Create SHAP explainer based on model type.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: uv add shap")
            
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
            print("Created TreeExplainer")
        else:
            # Default to TreeExplainer for ensemble models
            self.explainer = shap.TreeExplainer(self.model)
            print("Created TreeExplainer (default)")
    
    def calculate_shap_values(self, use_test_set=True, max_samples=100):
        """
        Calculate SHAP values for the dataset.
        """
        if self.explainer is None:
            self.create_explainer()
        
        data_to_explain = self.X_test if use_test_set else self.X_train
        
        # Limit samples for computational efficiency
        if len(data_to_explain) > max_samples:
            indices = np.random.choice(len(data_to_explain), max_samples, replace=False)
            data_to_explain = data_to_explain[indices]
            print(f"Analyzing {max_samples} random samples")
        
        print("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(data_to_explain)
        
        # For binary classification, use positive class SHAP values
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]
        
        self.analyzed_data = data_to_explain
        print(f"SHAP values calculated for {len(data_to_explain)} samples")
    
    def get_feature_importance(self, top_k=20):
        """
        Get global feature importance based on SHAP values.
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Calculate mean absolute SHAP values for each feature
        importance_scores = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance_scores)],
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def plot_feature_importance(self, top_k=20, title="SHAP Feature Importance"):
        """
        Create feature importance plot using SHAP values.
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive plots")
            return None
            
        importance_df = self.get_feature_importance(top_k)
        
        # Create plotly bar chart
        fig = px.bar(
            importance_df.head(top_k), 
            x='importance', 
            y='feature',
            orientation='h',
            title=title,
            labels={'importance': 'Mean |SHAP Value|', 'feature': 'Features'}
        )
        
        fig.update_layout(
            height=max(400, top_k * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def generate_simple_report(self):
        """
        Generate a simple SHAP analysis report.
        """
        if not SHAP_AVAILABLE:
            return "SHAP not available. Install with: uv add shap"
            
        if self.shap_values is None:
            self.calculate_shap_values()
        
        importance_df = self.get_feature_importance(10)
        
        report = """
        SHAP ANALYSIS REPORT
        ===================
        
        Top 10 Most Important Features:
        """
        
        for idx, row in importance_df.iterrows():
            report += f"\\n        {idx+1}. {row['feature']}: {row['importance']:.4f}"
        
        report += """
        
        Key Insights:
        - Features with higher SHAP values have more impact on predictions
        - SHAP values explain individual prediction contributions
        - Use SHAP for model interpretability and feature selection
        """
        
        return report

def main():
    """Demonstrate SHAP analysis functionality"""
    print("SHAP Analysis Module for Model Interpretability")
    print("This module provides model explanation capabilities.")
    print("\\nKey Features:")
    print("- Global feature importance analysis")
    print("- Individual prediction explanations")
    print("- Interactive visualizations (when plotly available)")

if __name__ == "__main__":
    main()