import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

def load_model(model_path: str):
    """
    Load trained model from file
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        object: Loaded model
    """
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {str(e)}")

def get_default_value(column_name: str, column_type: str) -> Any:
    """
    Get default values for missing columns based on column type
    
    Args:
        column_name (str): Name of the column
        column_type (str): Type of the column ('numeric' or 'categorical')
        
    Returns:
        Any: Default value for the column
    """
    defaults = {
        'numeric': {
            'Temperature': 20.0,
            'Humidity': 50,
            'Wind Speed': 0.0,
            'Precipitation (%)': 0,
            'Atmospheric Pressure': 1013.0,
            'UV Index': 5
        },
        'categorical': {
            'Cloud Cover': 'partly cloudy',
            'Season': 'Spring',
            'Location': 'inland'
        }
    }
    
    return defaults[column_type].get(column_name, np.nan if column_type == 'numeric' else 'unknown')

def predict_weather(model: Any, preprocessor: Any, input_data: Dict) -> Dict:
    """
    Predict weather type with detailed probability analysis
    
    Args:
        model: Trained model object
        preprocessor: Fitted preprocessor object
        input_data (dict): Dictionary containing input features
        
    Returns:
        dict: Dictionary containing prediction results including:
            - predicted_class: The predicted weather type
            - confidence: Confidence score for the prediction
            - probabilities: Dictionary of all weather types and their probabilities
            - input_validation: Dictionary containing information about missing/default values
    """
    try:
        # Track missing values and defaults used
        input_validation = {
            'missing_columns': [],
            'default_values_used': {}
        }
        
        # Get required columns from preprocessor
        numeric_cols = preprocessor.transformers_[0][2]
        categorical_cols = preprocessor.transformers_[1][2]
        required_columns = numeric_cols + categorical_cols
        
        # Check for missing columns and fill with defaults
        for column in required_columns:
            if column not in input_data:
                input_validation['missing_columns'].append(column)
                column_type = 'numeric' if column in numeric_cols else 'categorical'
                default_value = get_default_value(column, column_type)
                input_data[column] = default_value
                input_validation['default_values_used'][column] = default_value
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Preprocess input
        input_preprocessed = preprocessor.transform(input_df)
        
        # Get predictions and probabilities
        prediction = model.predict(input_preprocessed)[0]
        probability = model.predict_proba(input_preprocessed)[0]
        
        # Get probabilities for all classes
        class_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(model.classes_, probability)
        }
        
        # Sort probabilities in descending order
        sorted_probabilities = dict(
            sorted(class_probabilities.items(), 
                  key=lambda x: x[1], 
                  reverse=True)
        )
        
        # Prepare result dictionary
        result = {
            'predicted_class': prediction,
            'confidence': float(np.max(prediction_proba)),
            'probabilities': sorted_probabilities,
            'input_validation': input_validation
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")
