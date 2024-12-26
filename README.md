# App for Learning about weather_Prediction
 
# 🌦️ Weather Prediction System

## Overview
The Climate Prediction System is a machine learning-based web application developed with **Streamlit**. It allows users to predict weather conditions (drizzle, rain, fog, sun, snow) based on various climate features such as temperature, precipitation, wind speed, and more. The app uses multiple machine learning models like **Random Forest**, **Logistic Regression**, and **Support Vector Machine (SVM)** to make accurate predictions.

The app also provides **tips, precautions, and recommendations** for the predicted weather, helping users make informed decisions based on the forecast.

## Features
- **Weather Prediction**: Predicts the weather condition (drizzle, rain, fog, sun, snow) based on input climate data.
- **Model Selection**: Users can choose between three models: Random Forest, Logistic Regression, and Support Vector Machine.
- **Interactive Input**: Enter max/min temperature, precipitation, wind speed, year, and month for prediction.
- **Weather Tips & Recommendations**: Get personalized tips based on the predicted weather condition.
- **Model Accuracy**: View the accuracy of the selected model.

## Requirements
To run this app locally, you'll need to have the following libraries installed:

- `pandas`
- `scikit-learn`
- `streamlit`
- `Pillow` (for displaying images)

You can install the dependencies using pip:

bash
pip install pandas scikit-learn streamlit Pillow


Run the Streamlit app:
   bash
   streamlit run app.py
   

   The app should now be live at `http://localhost:8501` in your browser.

## How It Works
- The app loads weather data from the `weather.csv` file, performs feature engineering, and trains three different machine learning models: Random Forest, Logistic Regression, and SVM.
- The user inputs climate data, and the app uses the selected model to predict the weather condition.
- Based on the prediction, the app displays tips, precautions, and recommendations for the predicted weather.
- The app also provides the accuracy of the selected model.

## Weather Tips and Recommendations

- **Drizzle**: A light rain shower. Recommended to carry an umbrella and wear waterproof clothing.
- **Rain**: Moderate to heavy rain expected. Be sure to carry an umbrella, wear waterproof footwear, and avoid outdoor activities.
- **Fog**: Visibility will be low. Drive cautiously and use fog lights if driving. Avoid outdoor activities if possible.
- **Sun**: Ideal weather for outdoor activities. Drink plenty of water, wear sunscreen, and stay hydrated.
- **Snow**: Snowfall expected. Wear warm clothing, ensure safe driving, and avoid unnecessary travel.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository and submit a pull request. You can also raise issues to report bugs or suggest improvements.

## Developed by
**Salman Ahmed**

