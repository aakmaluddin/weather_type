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
    
    # Ensure all necessary columns are present in the input
    required_columns = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
    missing_columns = set(required_columns) - set(input_data.keys())
    
    if missing_columns:
        for column in missing_columns:
            # Provide default values for missing columns
            if column in preprocessor.transformers_[0][2]:  # Numeric columns
                input_data[column] = np.nan
            else:  # Categorical columns
                input_data[column] = 'missing'
    
    # Preprocess input
    input_df = pd.DataFrame([input_data])
    input_preprocessed = preprocessor.transform(input_df)
    
    # Predict
    prediction = model.predict(input_preprocessed)
    
    return prediction[0]
