import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load dataset"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocessing data"""
    # Remove outliers using z-score
    numeric_data = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_data))
    df_cleaned = df[(z_scores < 3).all(axis=1)]
    
    # Handle missing values
    df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)
    df_cleaned.fillna('missing', inplace=True)
    
    # Separate features and target
    X = df_cleaned.drop('Weather Type', axis=1)
    y = df_cleaned['Weather Type']
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Reconstruct DataFrame with appropriate column names
    # Get feature names from transformers
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + cat_feature_names.tolist()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    return X_transformed_df, y, preprocessor
