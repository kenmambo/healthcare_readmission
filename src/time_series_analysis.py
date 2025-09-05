"""
Time-series analysis module for healthcare readmission patterns.
Analyzes temporal patterns and multiple admission sequences.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """
    Analyzes temporal patterns in hospital readmissions for diabetic patients.
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the time series analyzer.
        
        Args:
            data_path (str): Path to the dataset
            df (pd.DataFrame): Pre-loaded DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.patient_sequences = None
        self.temporal_features = None
        
    def simulate_temporal_data(self):
        """
        Simulate temporal data since the original dataset doesn't have dates.
        Creates realistic admission dates and patterns.
        """
        np.random.seed(42)
        
        # Create base date range (1999-2008 as per dataset description)
        start_date = datetime(1999, 1, 1)
        end_date = datetime(2008, 12, 31)
        
        # Generate admission dates
        date_range = (end_date - start_date).days
        admission_dates = []
        
        for idx, row in self.df.iterrows():
            # Random date within range
            random_days = np.random.randint(0, date_range)
            admission_date = start_date + timedelta(days=random_days)
            admission_dates.append(admission_date)
        
        self.df['admission_date'] = admission_dates
        
        # Generate discharge dates based on time_in_hospital
        self.df['discharge_date'] = self.df.apply(
            lambda row: row['admission_date'] + timedelta(days=row.get('time_in_hospital', 1)), 
            axis=1
        )
        
        return self.df
    
    def create_patient_sequences(self):
        """
        Create patient admission sequences for temporal pattern analysis.
        """
        if 'admission_date' not in self.df.columns:
            self.simulate_temporal_data()
        
        # Group by patient to create sequences (using encounter_id as proxy)
        if 'patient_nbr' not in self.df.columns:
            # Create synthetic patient numbers for analysis
            self.df['patient_nbr'] = np.random.randint(1, len(self.df)//2, len(self.df))
        
        patient_groups = self.df.groupby('patient_nbr')
        sequences = []
        
        for patient_id, group in patient_groups:
            if len(group) > 1:  # Only patients with multiple admissions
                group_sorted = group.sort_values('admission_date')
                
                # Calculate time between admissions
                group_sorted['days_since_last_admission'] = group_sorted['admission_date'].diff().dt.days
                
                # Calculate readmission within 30 days of previous discharge
                group_sorted['readmission_30d'] = 0
                for i in range(1, len(group_sorted)):
                    prev_discharge = group_sorted.iloc[i-1]['discharge_date']
                    current_admission = group_sorted.iloc[i]['admission_date']
                    if (current_admission - prev_discharge).days <= 30:
                        group_sorted.iloc[i, group_sorted.columns.get_loc('readmission_30d')] = 1
                
                sequences.append(group_sorted)
        
        if sequences:
            self.patient_sequences = pd.concat(sequences)
        else:
            self.patient_sequences = pd.DataFrame()
            
        return self.patient_sequences
    
    def analyze_readmission_patterns(self):
        """
        Analyze patterns in readmission timing and frequency.
        """
        if self.patient_sequences is None:
            self.create_patient_sequences()
        
        if self.patient_sequences.empty:
            print("No patients with multiple admissions found.")
            return {'total_readmissions': 0}
        
        # Readmission rate analysis
        total_readmissions = len(self.patient_sequences)
        readmissions_30d = self.patient_sequences['readmission_30d'].sum()
        
        # Time-based patterns
        self.patient_sequences['month'] = self.patient_sequences['admission_date'].dt.month
        self.patient_sequences['day_of_week'] = self.patient_sequences['admission_date'].dt.dayofweek
        self.patient_sequences['quarter'] = self.patient_sequences['admission_date'].dt.quarter
        
        patterns = {
            'total_readmissions': total_readmissions,
            'readmissions_30d': readmissions_30d,
            'readmission_rate_30d': readmissions_30d / total_readmissions if total_readmissions > 0 else 0,
            'avg_days_between_admissions': self.patient_sequences['days_since_last_admission'].mean(),
            'median_days_between_admissions': self.patient_sequences['days_since_last_admission'].median(),
            'monthly_patterns': self.patient_sequences.groupby('month')['readmission_30d'].agg(['count', 'sum', 'mean']),
            'weekly_patterns': self.patient_sequences.groupby('day_of_week')['readmission_30d'].agg(['count', 'sum', 'mean']),
            'quarterly_patterns': self.patient_sequences.groupby('quarter')['readmission_30d'].agg(['count', 'sum', 'mean'])
        }
        
        return patterns
    
    def visualize_temporal_patterns(self):
        """
        Create comprehensive visualizations of temporal patterns.
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive plots")
            return None
            
        patterns = self.analyze_readmission_patterns()
        
        if not patterns or patterns['total_readmissions'] == 0:
            print("No data available for visualization.")
            return None
        
        # Create simple visualization
        fig = go.Figure()
        
        # Add monthly patterns
        monthly_data = patterns['monthly_patterns']
        if not monthly_data.empty:
            fig.add_trace(go.Bar(
                x=monthly_data.index, 
                y=monthly_data['count'], 
                name='Monthly Admissions'
            ))
        
        fig.update_layout(
            title="Temporal Readmission Patterns",
            xaxis_title="Month",
            yaxis_title="Count"
        )
        
        return fig
    
    def generate_report(self):
        """
        Generate a comprehensive temporal analysis report.
        """
        patterns = self.analyze_readmission_patterns()
        
        report = f"""
        TEMPORAL ANALYSIS REPORT
        ========================
        
        Dataset Overview:
        - Total readmission events: {patterns.get('total_readmissions', 0)}
        - 30-day readmissions: {patterns.get('readmissions_30d', 0)}
        - 30-day readmission rate: {patterns.get('readmission_rate_30d', 0):.2%}
        
        Temporal Patterns:
        - Average days between admissions: {patterns.get('avg_days_between_admissions', 0):.1f}
        - Median days between admissions: {patterns.get('median_days_between_admissions', 0):.1f}
        
        Key Insights:
        - Analysis completed on simulated temporal data
        - Real implementation would use actual admission timestamps
        - Pattern detection helps identify high-risk periods
        
        Recommendations:
        1. Focus intervention efforts on identified high-risk temporal patterns
        2. Implement targeted follow-up protocols during peak readmission periods
        3. Consider seasonal factors in discharge planning
        4. Develop patient-specific monitoring based on patterns
        """
        
        return report

def main():
    """Demonstrate time series analysis functionality"""
    print("Time Series Analysis Module for Healthcare Readmission")
    print("This module provides temporal pattern analysis capabilities.")
    print("\\nKey Features:")
    print("- Patient admission sequence analysis")
    print("- Temporal pattern identification")
    print("- Interactive visualizations")
    print("- Comprehensive reporting")

if __name__ == "__main__":
    main()