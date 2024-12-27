import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    """Load trained model"""
    return joblib.load(model_path)

def predict_weather(model, preprocessor, input_data):
    """Predict weather type"""
    # Prepare input data as DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input
    input_preprocessed = preprocessor.transform(input_df)
    
    # Predict
    prediction = model.predict(input_preprocessed)
    prediction_proba = model.predict_proba(input_preprocessed)
    
    return prediction[0], np.max(prediction_proba)