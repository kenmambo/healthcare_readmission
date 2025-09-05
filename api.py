"""FastAPI real-time prediction API for healthcare readmission risk."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Readmission Risk Prediction API",
    description="Real-time API for predicting 30-day hospital readmission risk in diabetic patients",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_metadata = {}

# Pydantic models for request/response
class PatientData(BaseModel):
    """Patient data model for prediction requests."""
    
    # Demographics
    race: str = Field(default="Caucasian", description="Patient race")
    gender: str = Field(description="Patient gender (Male/Female)")
    age: int = Field(ge=0, le=100, description="Patient age in years")
    
    # Admission details
    admission_type_id: int = Field(ge=1, le=8, description="Admission type ID")
    discharge_disposition_id: int = Field(ge=1, le=30, default=1, description="Discharge disposition ID")
    admission_source_id: int = Field(ge=1, le=25, default=7, description="Admission source ID")
    
    # Clinical measurements
    time_in_hospital: int = Field(ge=1, le=14, description="Days in hospital")
    num_lab_procedures: int = Field(ge=0, le=150, default=50, description="Number of lab procedures")
    num_procedures: int = Field(ge=0, le=10, description="Number of procedures")
    num_medications: int = Field(ge=1, le=80, description="Number of medications")
    
    # Visit history
    number_outpatient: int = Field(ge=0, le=50, default=0, description="Outpatient visits")
    number_emergency: int = Field(ge=0, le=20, default=0, description="Emergency visits")
    number_inpatient: int = Field(ge=0, le=20, default=0, description="Inpatient visits")
    
    # Diagnoses
    diag_1: str = Field(default="250", description="Primary diagnosis code")
    diag_2: str = Field(default="0", description="Secondary diagnosis code")
    diag_3: str = Field(default="0", description="Tertiary diagnosis code")
    number_diagnoses: int = Field(ge=1, le=16, description="Total number of diagnoses")
    
    # Lab results
    max_glu_serum: Optional[str] = Field(default="None", description="Maximum glucose serum level")
    A1Cresult: Optional[str] = Field(default="None", description="A1C test result")
    
    # Medications
    change: str = Field(default="No", description="Medication change status")
    diabetesMed: str = Field(default="No", description="Diabetes medication prescribed")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    prediction: int = Field(description="Prediction: 1 for high risk, 0 for low risk")
    probability: float = Field(description="Probability of readmission")
    risk_level: str = Field(description="Risk level: High Risk or Low Risk")
    confidence: float = Field(description="Model confidence")
    timestamp: str = Field(description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: str
    model_loaded: bool
    version: str

# Startup event to load model
@app.on_event("startup")
async def load_model():
    """Load the trained model and preprocessor on startup."""
    global model, preprocessor, model_metadata
    
    try:
        model_path = Path("model.pkl")
        preprocessor_path = Path("preprocessor.pkl")
        
        if model_path.exists() and preprocessor_path.exists():
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            
            # Load metadata if available
            try:
                model_metadata = joblib.load("model_metadata.pkl")
            except:
                model_metadata = {}
            
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model files not found. Please run model training first.")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def prepare_patient_data(patient: PatientData) -> pd.DataFrame:
    """Convert patient data to DataFrame format expected by the model."""
    
    # Convert to dictionary
    patient_dict = patient.dict()
    
    # Create DataFrame
    df = pd.DataFrame([patient_dict])
    
    # Add missing columns with default values if needed
    expected_columns = [
        'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
    ]
    
    # Add medication columns
    med_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    
    for col in med_columns:
        if col not in df.columns:
            df[col] = 'No'
    
    return df

# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "metadata": model_metadata,
        "model_type": str(type(model).__name__),
        "feature_count": len(preprocessor.feature_names_in_) if preprocessor else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission(patient: PatientData):
    """Predict readmission risk for a single patient."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare data
        patient_df = prepare_patient_data(patient)
        
        # Ensure all expected columns are present
        expected_cols = preprocessor.feature_names_in_
        for col in expected_cols:
            if col not in patient_df.columns:
                if col.startswith(('num_', 'number_')) or col.endswith('_id'):
                    patient_df[col] = 0
                else:
                    patient_df[col] = 'No'
        
        patient_df = patient_df.reindex(columns=expected_cols, fill_value='No')
        
        # Preprocess
        processed_data = preprocessor.transform(patient_df)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        # Format response
        response = PredictionResponse(
            prediction=int(prediction),
            probability=float(probabilities[1]),
            risk_level="High Risk" if prediction == 1 else "Low Risk",
            confidence=float(max(probabilities)),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction made: {response.risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/example")
async def get_prediction_example():
    """Get an example of prediction input format."""
    return {
        "example_patient": {
            "race": "Caucasian",
            "gender": "Male",
            "age": 65,
            "admission_type_id": 1,
            "time_in_hospital": 5,
            "num_procedures": 1,
            "num_medications": 20,
            "number_diagnoses": 9,
            "change": "No",
            "diabetesMed": "Yes"
        },
        "description": "Use this format for POST /predict requests"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )