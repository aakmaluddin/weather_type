import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, plot_feature_importance, plot_confusion_matrix
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
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error('Dataset not found! Please ensure the file exists.')
        return

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
        try:
            X, y, preprocessor = preprocess_data(df)
        except Exception as e:
            st.error(f'Data preprocessing failed: {e}')
            return

        if st.button('Train Model'):
            with st.spinner('Training Model...'):
                try:
                    model = train_model(X, y)

                    # Feature Importance
                    st.subheader('Feature Importance')
                    fig_importance = plot_feature_importance(model, X.columns)
                    st.pyplot(fig_importance)

                    # Confusion Matrix
                    st.subheader('Confusion Matrix')
                    fig_confusion = plot_confusion_matrix(model, X, y)
                    st.pyplot(fig_confusion)

                except Exception as e:
                    st.error(f'Model training failed: {e}')

    elif page == 'Weather Prediction':
        st.header('Weather Prediction')

        # Load trained model and preprocessor
        try:
            model = joblib.load('models/weather_classifier.pkl')
            _, _, preprocessor = preprocess_data(df)
        except FileNotFoundError:
            st.error('Trained model not found! Please train the model first.')
            return
        except Exception as e:
            st.error(f'Error loading model: {e}')
            return

        # Input fields for prediction
        input_data = {}
        cols = st.columns(3)

        with cols[0]:
            input_data['Temperature'] = st.number_input('Temperature (°C)', min_value=-50.0, max_value=50.0, step=0.1)
            input_data['Humidity'] = st.number_input('Humidity (%)', min_value=0, max_value=100, step=1)
            input_data['Wind Speed'] = st.number_input('Wind Speed (km/h)', min_value=0.0, max_value=100.0, step=0.1)

        with cols[1]:
            input_data['Precipitation (%)'] = st.number_input('Precipitation (%)', min_value=0, max_value=100, step=1)
            input_data['Cloud Cover'] = st.selectbox('Cloud Cover', ['clear', 'partly cloudy', 'overcast'])
            input_data['Atmospheric Pressure'] = st.number_input('Atmospheric Pressure (hPa)', min_value=900.0, max_value=1100.0, step=0.1)

        with cols[2]:
            input_data['UV Index'] = st.number_input('UV Index', min_value=0, max_value=15, step=1)
            input_data['Season'] = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
            input_data['Location'] = st.selectbox('Location', ['inland', 'mountain', 'coastal'])

        if st.button('Predict Weather'):
            try:
                prediction, confidence = predict_weather(model, preprocessor, input_data)

                st.subheader('Classification Results')
                st.success(f'Predicted Weather Type: {prediction}')
                st.info(f'Prediction Confidence: {confidence:.2f}%')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

if __name__ == '__main__':
    main()
