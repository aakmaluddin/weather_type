import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, feature_importance
from src.prediction import predict_weather

def main():
    st.title('Weather Classification System')
    st.sidebar.header('Navigation')
    
    page = st.sidebar.radio(
        'Choose a Page', 
        ['Data Overview', 'Model Training', 'Weather Prediction']
    )
    
    # Load data
    data_path = 'data/weather_classification.csv'
    df = load_data(data_path)
    
    if page == 'Data Overview':
        st.header('Dataset Overview')
        st.dataframe(df.head())
        
        st.subheader('Dataset Statistics')
        st.dataframe(df.describe())
        
        st.subheader('Weather Type Distribution')
        weather_counts = df['Weather Type'].value_counts()
        st.bar_chart(weather_counts)
    
    elif page == 'Model Training':
        st.header('Model Training')
        
        # Preprocess data
        X, y, preprocessor = preprocess_data(df)
        
        if st.button('Train Model'):
            with st.spinner('Training Model...'):
                model = train_model(X, y)
                
                # Feature Importance
                st.subheader('Feature Importance')
                feature_importance(model, X.columns)
                
                st.image('feature_importance.png')
                st.image('confusion_matrix.png')
    
    elif page == 'Weather Prediction':
        st.header('Weather Prediction')
        
        # Load trained model and preprocessor
        model = joblib.load('models/weather_classifier.pkl')
        X, y, preprocessor = preprocess_data(df)
        
        # Input fields based on dataset columns
        input_data = {}
        cols = st.columns(3)
        
        with cols[0]:
            input_data['Temperature'] = st.number_input('Temperature', min_value=-50.0, max_value=50.0, step=0.1)
            input_data['Humidity'] = st.number_input('Humidity (%)', min_value=0, max_value=100, step=1)
            input_data['Wind Speed'] = st.number_input('Wind Speed', min_value=0.0, max_value=100.0, step=0.1)
        
        with cols[1]:
            input_data['Precipitation (%)'] = st.number_input('Precipitation (%)', min_value=0, max_value=100, step=1)
            input_data['Cloud Cover'] = st.selectbox('Cloud Cover', ['clear', 'partly cloudy', 'overcast'])
            input_data['Atmospheric Pressure'] = st.number_input('Atmospheric Pressure', min_value=900.0, max_value=1100.0, step=0.1)
        
        with cols[2]:
            input_data['UV Index'] = st.number_input('UV Index', min_value=0, max_value=15, step=1)
            input_data['Season'] = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
            input_data['Location'] = st.selectbox('Location', ['inland', 'mountain', 'coastal'])
        
        if st.button('Predict Weather'):
            prediction, probability = predict_weather(model, preprocessor, input_data)
            
            st.subheader('Prediction Results')
            st.success(f'Predicted Weather: {prediction}')
            st.info(f'Prediction Confidence: {probability*100:.2f}%')

if __name__ == '__main__':
    main()
