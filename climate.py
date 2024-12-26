import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from PIL import Image

# Load the dataset
data = pd.read_csv('weather.csv')

# Feature Engineering: Extract year and month from date
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Encode 'weather' as categorical values (e.g., drizzle = 1, rain = 2, etc.)
data['weather'] = data['weather'].map({'drizzle': 1, 'rain': 2, 'fog': 3, 'sun': 4, 'snow': 5})

# Drop rows with NaN values in 'weather' column
data = data.dropna(subset=['weather'])

# Features and target (including 'month' and 'year' for training)
X = data[['temp_max', 'temp_min', 'precipitation', 'wind', 'year', 'month']]  
y = data['weather'] 

# Train models
scaler = StandardScaler()

# Scale all features
X_scaled = scaler.fit_transform(X)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
logreg_model = LogisticRegression(random_state=42)
svm_model = SVC(kernel='linear', random_state=42)

# Fit models
rf_model.fit(X_scaled, y)
logreg_model.fit(X_scaled, y)
svm_model.fit(X_scaled, y)

# Streamlit App
st.set_page_config(page_title="Climate Prediction", page_icon="🌦️", layout="wide")
st.title("🌦️ **Climate Prediction System**")

# Upload an icon image for the app header
icon_image = Image.open('weather-icon.jpg')  # Replace with your own image path
st.image(icon_image, width=120)

st.markdown("""
    ### Welcome to the Climate Prediction App!
    Predict the weather condition based on various climate features such as temperature, precipitation, wind speed, and more.
    """)

# Layout for user input (dividing into columns for better user experience)
col1, col2 = st.columns(2)
with col1:
    temp_max = st.number_input("Max Temperature (°C)", min_value=-30.0, max_value=50.0, value=12.8, help="Enter the maximum temperature.")
    temp_min = st.number_input("Min Temperature (°C)", min_value=-30.0, max_value=50.0, value=5.0, help="Enter the minimum temperature.")
    wind = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=4.7, help="Enter the wind speed in meters per second.")
with col2:
    precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0, help="Enter the amount of precipitation in mm.")
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2012, help="Enter the year of the observation.")
    month = st.number_input("Month", min_value=1, max_value=12, value=1, help="Enter the month of the observation.")

# Add model selection dropdown with weather icons
model_option = st.selectbox("Choose the Model for Prediction", 
                            ("Random Forest", "Logistic Regression", "SVM"), 
                            help="Select a machine learning model to predict the weather condition.")

# Add relevant icons for each model
model_images = {
    "Random Forest": 'rf-model-icon.jpg', 
    "Logistic Regression": 'logreg-icon.jpg', 
    "SVM": 'svm-icon.jpg'
}

model_image = Image.open(model_images[model_option])
st.image(model_image, caption=f"{model_option} Model", width=150)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'temp_max': [temp_max],
    'temp_min': [temp_min],
    'precipitation': [precipitation],
    'wind': [wind],
    'year': [year],
    'month': [month]
})

# Scale the input data
input_scaled = scaler.transform(input_data)

# Prediction logic
if st.button("Predict"):
    if model_option == "Random Forest":
        prediction = rf_model.predict(input_scaled)
    elif model_option == "Logistic Regression":
        prediction = logreg_model.predict(input_scaled)
    else:
        prediction = svm_model.predict(input_scaled)
    
    # Display the prediction result
    weather_condition = {1: "Drizzle", 2: "Rain", 3: "Fog", 4: "Sun", 5: "Snow"}
    predicted_weather = weather_condition.get(prediction[0], 'Unknown')
    st.write(f"**Predicted Weather Condition**: {predicted_weather}")

    # Weather Tips, Precautions, and Recommendations
    st.write("### Tips & Recommendations for the Predicted Weather:")

    if predicted_weather == "Drizzle":
        st.write("☔ **Drizzle**: A light rain shower. Recommended to carry an umbrella and wear waterproof clothing.")
    elif predicted_weather == "Rain":
        st.write("🌧️ **Rain**: Moderate to heavy rain expected. Be sure to carry an umbrella, wear waterproof footwear, and avoid outdoor activities.")
    elif predicted_weather == "Fog":
        st.write("🌫️ **Fog**: Visibility will be low. Drive cautiously and use fog lights if driving. Avoid outdoor activities if possible.")
    elif predicted_weather == "Sun":
        st.write("☀️ **Sunny**: Ideal weather for outdoor activities. Drink plenty of water, wear sunscreen, and stay hydrated.")
    elif predicted_weather == "Snow":
        st.write("❄️ **Snow**: Snowfall expected. Wear warm clothing, ensure safe driving, and avoid unnecessary travel.")

    # Display model accuracy with a progress bar
    if model_option == "Random Forest":
        accuracy = rf_model.score(X_scaled, y)
    elif model_option == "Logistic Regression":
        accuracy = logreg_model.score(X_scaled, y)
    else:
        accuracy = svm_model.score(X_scaled, y)
    
    st.write(f"**Model Accuracy**: {accuracy:.2f}")
    st.progress(accuracy)  # Progress bar to show model accuracy

# Add some styling and footer
st.markdown("""
    ---
    **Developed by:** Your Name  
    [Visit GitHub Repository](https://github.com/yourusername/yourrepo)  
    """, unsafe_allow_html=True)

# Add a floating action button for better user experience
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 12px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
