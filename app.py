import streamlit as st

# --- Streamlit App Configuration (must be FIRST) ---
st.set_page_config("Energy Load Forecasting", layout="centered")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import joblib
import requests
import speech_recognition as sr
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
import base64

# Load saved model and scaler with error handling
@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler with proper error handling"""
    try:
        # Clear any existing keras sessions
        tf.keras.backend.clear_session()
        
        # Try loading the model
        model = tf.keras.models.load_model("pjm_lstm_model.keras", compile=False)
        
        # Manually compile the model to avoid optimizer loading issues
        model.compile(
            optimizer='rmsprop',
            loss='mse',
            metrics=['mae']
        )
        
        scaler = joblib.load("pjm_scaler.pkl")
        
        return model, scaler, "Model loaded successfully"
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check if 'pjm_lstm_model.keras' and 'pjm_scaler.pkl' exist in the current directory")
        st.stop()

# Load model and scaler
model, scaler, load_status = load_model_and_scaler()

# Dummy user credentials
USER_CREDENTIALS = {"admin@example.com": "admin123"}

# Voice recognition function
def listen_for_command():
    """Listen for voice commands"""
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Say 'start to predict' or 'logout'")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
        
        command = r.recognize_google(audio).lower()
        return command
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition: {e}"
    except sr.WaitTimeoutError:
        return "Listening timeout"
    except Exception as e:
        return f"Error: {e}"

# OpenWeather API function
def get_current_weather():
    """Fetch current weather data from OpenWeather API"""
    try:
        api_key = st.secrets["openweather"]["api_key"]
        # Philadelphia coordinates (same as used in training data)
        lat, lon = 39.9526, -75.1652
        
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        return temperature, humidity, True
        
    except Exception as e:
        st.warning(f"Failed to fetch weather data: {str(e)}. Using random values.")
        # Fallback to random values if API fails
        return random.uniform(20, 35), random.uniform(30, 70), False

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("pjm_with_weather.csv", parse_dates=["Datetime"])
    df.sort_values("Datetime", inplace=True)
    return df

# Create LSTM sequences
def create_sequence(data, seq_len=24):
    return np.expand_dims(data[-seq_len:], axis=0)

# Detect spike or anomaly (enhanced logic)
def detect_anomaly(actual, predicted, threshold=0.15):
    """Enhanced anomaly detection with multiple criteria"""
    # Calculate percentage error
    percentage_error = abs(actual - predicted) / actual
    
    # Multiple anomaly criteria
    criteria = {
        'high_percentage_error': percentage_error > threshold,
        'extreme_spike': abs(predicted - actual) > 5000,  # MW threshold
        'sudden_change': False  # Will be set based on historical comparison
    }
    
    return any(criteria.values()), criteria, percentage_error

def calculate_anomaly_score(predictions, historical_data, temperatures, humidities):
    """Calculate comprehensive anomaly scores"""
    scores = []
    alerts = []
    
    # Get recent historical statistics
    recent_loads = historical_data["PJM_Load_MW"].tail(24).values  # Last 24 hours
    hist_mean = np.mean(recent_loads)
    hist_std = np.std(recent_loads)
    
    for i, pred in enumerate(predictions):
        # Z-score based anomaly detection
        z_score = abs(pred - hist_mean) / hist_std if hist_std > 0 else 0
        
        # Temperature-based anomaly (extreme temperatures affecting load)
        temp_anomaly = temperatures[i] > 40 or temperatures[i] < -10
        
        # Load deviation from normal range
        load_deviation = abs(pred - hist_mean) / hist_mean if hist_mean > 0 else 0
        
        # Composite anomaly score (0-1 scale)
        score = min(1.0, (z_score / 3) + (load_deviation * 0.5) + (0.3 if temp_anomaly else 0))
        scores.append(score)
        
        # Generate alerts based on score
        if score > 0.8:
            alerts.append(f"🚨 CRITICAL: Hour +{i+1} (Score: {score:.2f})")
        elif score > 0.6:
            alerts.append(f"⚠️ HIGH: Hour +{i+1} (Score: {score:.2f})")
        elif score > 0.4:
            alerts.append(f"🟡 MEDIUM: Hour +{i+1} (Score: {score:.2f})")
    
    return scores, alerts

# Prediction history management
def store_prediction_sample(predictions, temperatures, humidities, data_source, timestamp):
    """Store prediction sample in session state"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    sample = {
        'timestamp': timestamp,
        'predictions': predictions.copy(),
        'temperatures': temperatures.copy(), 
        'humidities': humidities.copy(),
        'data_source': data_source
    }
    
    # Keep only last 2 samples
    st.session_state.prediction_history.append(sample)
    if len(st.session_state.prediction_history) > 2:
        st.session_state.prediction_history.pop(0)
    
    return len(st.session_state.prediction_history)

def get_prediction_history():
    """Get stored prediction history"""
    return st.session_state.get('prediction_history', [])

def compare_with_previous_predictions(current_predictions, history):
    """Compare current predictions with previous ones"""
    if not history:
        return [], "No previous predictions to compare"
    
    comparison_results = []
    
    for i, sample in enumerate(history):
        prev_preds = sample['predictions']
        timestamp = sample['timestamp']
        
        # Calculate differences
        differences = [abs(curr - prev) for curr, prev in zip(current_predictions, prev_preds[:len(current_predictions)])]
        avg_diff = np.mean(differences)
        max_diff = max(differences) if differences else 0
        
        comparison_results.append({
            'sample_index': i + 1,
            'timestamp': timestamp,
            'avg_difference': avg_diff,
            'max_difference': max_diff,
            'data_source': sample['data_source']
        })
    
    return comparison_results, "Comparison completed"

# Email Alert System
def get_email_config():
    """Get email configuration from secrets"""
    try:
        email_config = {
            'smtp_server': st.secrets.get("email", {}).get("smtp_server", "smtp.gmail.com"),
            'smtp_port': st.secrets.get("email", {}).get("smtp_port", 587),
            'sender_email': st.secrets.get("email", {}).get("sender_email", ""),
            'sender_password': st.secrets.get("email", {}).get("sender_password", ""),
            'admin_email': st.secrets.get("email", {}).get("admin_email", "admin@example.com"),
            'enabled': st.secrets.get("email", {}).get("enabled", False)
        }
        return email_config
    except Exception:
        return {
            'smtp_server': "smtp.gmail.com",
            'smtp_port': 587,
            'sender_email': "",
            'sender_password': "",
            'admin_email': "admin@example.com",
            'enabled': False
        }

def create_anomaly_email_content(predictions, temperatures, humidities, anomaly_scores, anomaly_alerts, data_source):
    """Create HTML email content for anomaly alerts"""
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #ff6b6b; color: white; padding: 15px; border-radius: 5px; }}
            .content {{ margin: 20px 0; }}
            .table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #f2f2f2; }}
            .critical {{ background-color: #ffebee; color: #c62828; }}
            .high {{ background-color: #fff3e0; color: #ef6c00; }}
            .medium {{ background-color: #f3e5f5; color: #7b1fa2; }}
            .normal {{ background-color: #e8f5e8; color: #2e7d32; }}
            .footer {{ margin-top: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🚨 Energy Load Forecasting - Anomaly Alert</h2>
            <p>Critical anomalies detected in energy demand predictions</p>
        </div>
        
        <div class="content">
            <p><strong>Alert Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Data Source:</strong> {data_source}</p>
            
            <h3>Detected Anomalies:</h3>
            <ul>
    """
    
    for alert in anomaly_alerts:
        html_content += f"<li>{alert}</li>"
    
    html_content += """
            </ul>
            
            <h3>4-Hour Forecast Details:</h3>
            <table class="table">
                <tr>
                    <th>Hour</th>
                    <th>Predicted Load (MW)</th>
                    <th>Temperature (°C)</th>
                    <th>Humidity (%)</th>
                    <th>Anomaly Score</th>
                    <th>Risk Level</th>
                </tr>
    """
    
    for i in range(len(predictions)):
        score = anomaly_scores[i]
        risk_class = (
            'critical' if score > 0.8 else 
            'high' if score > 0.6 else 
            'medium' if score > 0.4 else 
            'normal'
        )
        risk_text = (
            '🚨 CRITICAL' if score > 0.8 else 
            '⚠️ HIGH' if score > 0.6 else 
            '🟡 MEDIUM' if score > 0.4 else 
            '✅ NORMAL'
        )
        
        html_content += f"""
                <tr class="{risk_class}">
                    <td>+{i+1}h</td>
                    <td>{predictions[i]:.2f}</td>
                    <td>{temperatures[i]:.1f}</td>
                    <td>{humidities[i]:.1f}</td>
                    <td>{score:.3f}</td>
                    <td>{risk_text}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="footer">
            <p><strong>Energy Load Forecasting System</strong></p>
            <p>This is an automated alert. Please review the predictions and take appropriate action if necessary.</p>
            <p><em>Generated at """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def send_anomaly_email(predictions, temperatures, humidities, anomaly_scores, anomaly_alerts, data_source):
    """Send email alert for anomalies"""
    try:
        email_config = get_email_config()
        
        if not email_config['enabled'] or not email_config['sender_email']:
            return False, "Email alerts not configured or disabled"
        
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = f"🚨 Energy Load Anomaly Alert - {datetime.now().strftime('%H:%M:%S')}"
        message["From"] = email_config['sender_email']
        message["To"] = email_config['admin_email']
        
        # Create HTML content
        html_content = create_anomaly_email_content(
            predictions, temperatures, humidities, 
            anomaly_scores, anomaly_alerts, data_source
        )
        
        # Create plain text version
        text_content = f"""
Energy Load Forecasting - Anomaly Alert

Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {data_source}

Detected Anomalies:
"""
        for alert in anomaly_alerts:
            text_content += f"- {alert}\n"
        
        text_content += "\n4-Hour Forecast Summary:\n"
        for i in range(len(predictions)):
            text_content += f"Hour +{i+1}: {predictions[i]:.2f} MW (Score: {anomaly_scores[i]:.3f})\n"
        
        # Attach both versions
        text_part = MIMEText(text_content, "plain")
        html_part = MIMEText(html_content, "html")
        message.attach(text_part)
        message.attach(html_part)
        
        # Send email
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(message)
        
        return True, "Email alert sent successfully"
        
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

def check_and_send_alerts(predictions, temperatures, humidities, anomaly_scores, anomaly_alerts, data_source):
    """Check if alerts should be sent and send them"""
    # Only send alerts for critical or high anomalies
    critical_alerts = [alert for alert in anomaly_alerts if "CRITICAL" in alert or "HIGH" in alert]
    
    if critical_alerts:
        success, message = send_anomaly_email(
            predictions, temperatures, humidities, 
            anomaly_scores, anomaly_alerts, data_source
        )
        return success, message, len(critical_alerts)
    
    return False, "No critical alerts to send", 0

def send_test_email():
    """Send a test email to verify configuration"""
    try:
        email_config = get_email_config()
        
        if not email_config['enabled'] or not email_config['sender_email']:
            return False, "Email alerts not configured or disabled"
        
        # Create test message
        message = MIMEMultipart()
        message["Subject"] = "🧪 Energy Forecasting System - Test Email"
        message["From"] = email_config['sender_email']
        message["To"] = email_config['admin_email']
        
        # Test content
        test_content = f"""
Energy Load Forecasting System - Test Email

This is a test email to verify your email configuration.

Configuration Details:
- SMTP Server: {email_config['smtp_server']}
- SMTP Port: {email_config['smtp_port']}
- Sender: {email_config['sender_email']}
- Admin: {email_config['admin_email']}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

If you received this email, your configuration is working correctly!

Best regards,
Energy Load Forecasting System
        """
        
        message.attach(MIMEText(test_content, "plain"))
        
        # Send email
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(message)
        
        return True, "Test email sent successfully"
        
    except Exception as e:
        return False, f"Failed to send test email: {str(e)}"

# Predict next 4 hours load
def forecast_multiple_hours(df, scaler, hours=4, use_manual=False, manual_data=None):
    # Get weather data based on input mode
    if use_manual and manual_data:
        current_temp = manual_data["temperature"]
        current_humidity = manual_data["humidity"]
        api_success = False
        source = "manual"
    else:
        current_temp, current_humidity, api_success = get_current_weather()
        source = "api" if api_success else "random"
    
    # Take last 24 hours of historical data
    past_24 = df[["PJM_Load_MW", "Temperature_C", "Humidity_%"]].values[-24:]
    
    # Use manual current load if provided
    if use_manual and manual_data and "current_load" in manual_data:
        # Replace the last load value with manual input
        past_24[-1, 0] = manual_data["current_load"]
    
    predictions = []
    temperatures = []
    humidities = []
    
    # Create rolling window for predictions
    current_sequence = past_24.copy()
    
    for hour in range(hours):
        # Simulate weather changes for future hours
        if hour == 0:
            temp = current_temp
            humidity = current_humidity
        else:
            # Add small random variations for future hours
            temp = current_temp + np.random.normal(0, 2)  # ±2°C variation
            humidity = max(10, min(100, current_humidity + np.random.normal(0, 5)))  # ±5% variation
        
        temperatures.append(temp)
        humidities.append(humidity)
        
        # Use the last predicted load or actual load
        if hour == 0:
            if use_manual and manual_data and "current_load" in manual_data:
                last_load = manual_data["current_load"]
            else:
                last_load = df["PJM_Load_MW"].values[-1]
        else:
            last_load = predictions[-1]
        
        # Create new row for prediction
        new_row = np.array([last_load, temp, humidity])
        
        # Update sequence for prediction
        full_seq = np.vstack([current_sequence[-23:], new_row])
        
        # Scale and predict
        scaled_seq = scaler.transform(full_seq)
        X_input = create_sequence(scaled_seq)
        
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        pred = scaler.inverse_transform([[pred_scaled, temp, humidity]])[0][0]
        
        predictions.append(pred)
        
        # Update sequence for next iteration
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return predictions, temperatures, humidities, source

# Weather impact analysis
def analyze_weather_impact(df, scaler, base_temp, base_humidity):
    """Analyze how weather changes affect load predictions"""
    
    # Temperature variations (-10°C to +10°C from base)
    temp_variations = np.arange(base_temp - 10, base_temp + 11, 2)
    temp_impacts = []
    
    # Humidity variations (-20% to +20% from base)
    humidity_variations = np.arange(max(10, base_humidity - 20), min(100, base_humidity + 21), 5)
    humidity_impacts = []
    
    # Base prediction with current weather
    past_23 = df[["PJM_Load_MW", "Temperature_C", "Humidity_%"]].values[-23:]
    last_load = df["PJM_Load_MW"].values[-1]
    
    # Test temperature impact
    for temp in temp_variations:
        new_row = [last_load, temp, base_humidity]
        full_seq = np.vstack([past_23, new_row])
        scaled_seq = scaler.transform(full_seq)
        X_input = create_sequence(scaled_seq)
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        pred = scaler.inverse_transform([[pred_scaled, temp, base_humidity]])[0][0]
        temp_impacts.append(pred)
    
    # Test humidity impact
    for humidity in humidity_variations:
        new_row = [last_load, base_temp, humidity]
        full_seq = np.vstack([past_23, new_row])
        scaled_seq = scaler.transform(full_seq)
        X_input = create_sequence(scaled_seq)
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        pred = scaler.inverse_transform([[pred_scaled, base_temp, humidity]])[0][0]
        humidity_impacts.append(pred)
    
    return temp_variations, temp_impacts, humidity_variations, humidity_impacts

# --- Streamlit App ---
st.title("🔌 Energy Load Forecasting with LSTM")

# Show model loading status
if load_status:
    st.success(f"✅ {load_status}")
else:
    st.error("❌ Model loading failed")

# Session login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login
if not st.session_state.logged_in:
    st.subheader("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# Logout
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# Voice Command Section
st.subheader("🎙️ Voice Commands")
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🎤 Listen", help="Click to start voice recognition"):
        with st.spinner("Listening for voice command..."):
            command = listen_for_command()
            st.session_state.last_command = command
            
            if "start to predict" in command or "predict" in command:
                st.session_state.start_prediction = True
                st.success(f"✅ Command recognized: '{command}' - Starting prediction!")
            elif "logout" in command:
                st.session_state.logged_in = False
                st.success(f"✅ Command recognized: '{command}' - Logging out!")
                st.rerun()
            else:
                st.warning(f"⚠️ Command '{command}' not recognized. Try 'start to predict' or 'logout'")

with col2:
    if 'last_command' in st.session_state:
        st.info(f"Last command: {st.session_state.last_command}")
    else:
        st.info("No voice commands yet. Say 'start to predict' or 'logout'")

# Load data
df = load_data()

# Manual Input Section
st.subheader("📊 Input Options")

input_mode = st.radio(
    "Choose data source:",
    ["Real-time Weather API", "Manual Input"],
    horizontal=True
)

manual_values = {}
if input_mode == "Manual Input":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        manual_values["temperature"] = st.number_input(
            "Temperature (°C)", 
            min_value=-20.0, 
            max_value=50.0, 
            value=25.0, 
            step=0.1
        )
    
    with col2:
        manual_values["humidity"] = st.number_input(
            "Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0, 
            step=0.1
        )
    
    with col3:
        manual_values["current_load"] = st.number_input(
            "Current Load (MW)", 
            min_value=0.0, 
            max_value=100000.0, 
            value=float(df["PJM_Load_MW"].iloc[-1]), 
            step=10.0
        )
    
    st.info("📝 Manual values will be used for prediction instead of real-time API data")

# Sidebar for API status and prediction history
with st.sidebar:
    st.header("🌦️ Weather API Status")
    
    # Test API connection
    try:
        api_key = st.secrets["openweather"]["api_key"]
        if api_key == "YOUR_OPENWEATHER_API_KEY_HERE":
            st.error("⚠️ Please update your API key in secrets.toml")
        else:
            # Test API call
            temp, humidity, api_success = get_current_weather()
            if api_success:
                st.success("✅ API Connected")
                st.metric("Current Temp", f"{temp:.1f}°C")
                st.metric("Current Humidity", f"{humidity:.1f}%")
            else:
                st.warning("⚠️ API connection failed")
    except Exception:
        st.error("❌ API key not configured")
    
    # Prediction History Summary
    st.header("📈 Prediction History")
    history = get_prediction_history()
    
    if history:
        st.metric("Stored Samples", len(history))
        
        # Show latest sample summary
        latest = history[-1]
        st.write("**Latest Prediction:**")
        st.write(f"Time: {latest['timestamp'].strftime('%H:%M:%S')}")
        st.write(f"Source: {latest['data_source']}")
        avg_load = np.mean(latest['predictions'])
        st.write(f"Avg Load: {avg_load:.1f} MW")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No predictions stored yet")
    
    # Anomaly Detection Settings
    st.header("⚠️ Anomaly Settings")
    
    # Email configuration status
    email_config = get_email_config()
    st.write("**Email Alerts:**")
    if email_config['enabled'] and email_config['sender_email']:
        st.success("✅ Email alerts enabled")
        st.write(f"📧 Admin: {email_config['admin_email']}")
        
        # Test email button
        if st.button("🧪 Send Test Email"):
            with st.spinner("Sending test email..."):
                success, message = send_test_email()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    else:
        st.warning("⚠️ Email alerts disabled")
        st.write("Configure in secrets.toml to enable")
    
    # Allow users to adjust anomaly thresholds
    st.write("**Detection Thresholds:**")
    st.write("🚨 Critical: Score > 0.8")
    st.write("⚠️ High: Score > 0.6") 
    st.write("🟡 Medium: Score > 0.4")
    st.write("✅ Normal: Score ≤ 0.4")
    
    st.info("Anomaly scores are calculated based on:\n- Historical load patterns\n- Temperature extremes\n- Load deviation from normal range\n\n📧 Email alerts sent for Critical/High anomalies only")

# User Timestamp Input (not used directly, for now)
st.subheader("📅 Choose Timestamp to Predict")
user_time = st.time_input("Select Hour", value=datetime.now().time())
st.info("The prediction will be for 1 hour ahead of your selected timestamp.")

# Predict button with voice command integration
predict_button_clicked = st.button("🔮 Predict Next 4 Hours")
voice_prediction_triggered = st.session_state.get('start_prediction', False)

if predict_button_clicked or voice_prediction_triggered:
    # Reset voice command flag
    if voice_prediction_triggered:
        st.session_state.start_prediction = False
        st.success("🎙️ Voice command triggered prediction!")
    
    # Determine input mode
    use_manual_input = input_mode == "Manual Input"
    
    with st.spinner("Fetching weather data and generating 4-hour forecast..."):
        predictions, temperatures, humidities, data_source = forecast_multiple_hours(
            df, scaler, hours=4, 
            use_manual=use_manual_input, 
            manual_data=manual_values if use_manual_input else None
        )

    actual = df["PJM_Load_MW"].values[-1]
    
    # Store this prediction sample
    current_timestamp = datetime.now()
    sample_count = store_prediction_sample(predictions, temperatures, humidities, data_source, current_timestamp)
    
    # Get prediction history for comparison
    history = get_prediction_history()
    comparison_results, comparison_status = compare_with_previous_predictions(predictions, history)
    
    # Create timestamps for next 4 hours
    current_time = datetime.now()
    future_timestamps = [current_time + timedelta(hours=i+1) for i in range(4)]
    
    # Calculate enhanced anomaly scores
    anomaly_scores, anomaly_alerts = calculate_anomaly_score(predictions, df, temperatures, humidities)
    
    # Show Results
    st.subheader("📈 4-Hour Forecast Results")
    
    # Data source indicator
    if data_source == "manual":
        weather_source = "📝 Manual Input"
    elif data_source == "api":
        weather_source = "🌐 Real-time API"
    else:
        weather_source = "🎲 Random values (API failed)"
    
    st.info(f"Data source: {weather_source} | Prediction sample #{sample_count} stored")
    
    # Display predictions in a table with anomaly scores
    forecast_df = pd.DataFrame({
        'Hour': [f"+{i+1}h" for i in range(4)],
        'Time': [ts.strftime("%H:%M") for ts in future_timestamps],
        'Predicted Load (MW)': [f"{pred:.2f}" for pred in predictions],
        'Temperature (°C)': [f"{temp:.1f}" for temp in temperatures],
        'Humidity (%)': [f"{hum:.1f}" for hum in humidities],
        'Anomaly Score': [f"{score:.3f}" for score in anomaly_scores],
        'Risk Level': [
            '🚨 CRITICAL' if score > 0.8 else 
            '⚠️ HIGH' if score > 0.6 else 
            '🟡 MEDIUM' if score > 0.4 else 
            '✅ NORMAL' 
            for score in anomaly_scores
        ]
    })
    
    st.dataframe(forecast_df, use_container_width=True)
    
    # Enhanced anomaly alerts
    if anomaly_alerts:
        st.subheader("🚨 Anomaly Detection Alerts")
        
        # Check and send email alerts
        email_sent, email_message, alert_count = check_and_send_alerts(
            predictions, temperatures, humidities, 
            anomaly_scores, anomaly_alerts, data_source
        )
        
        # Display email status
        if email_sent:
            st.success(f"📧 Email alert sent to admin ({alert_count} critical alerts)")
        elif alert_count > 0:
            st.info(f"📧 {email_message}")
        
        # Display alerts in UI
        for alert in anomaly_alerts:
            if "CRITICAL" in alert:
                st.error(alert)
            elif "HIGH" in alert:
                st.warning(alert)
            else:
                st.info(alert)
    else:
        st.success("✅ All forecasts are within normal range.")
    
    # Prediction History Comparison
    if len(history) > 0:
        st.subheader("📊 Prediction History Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Comparison with Previous Predictions:**")
            for comp in comparison_results:
                st.write(f"**Sample {comp['sample_index']}** ({comp['timestamp'].strftime('%H:%M:%S')})")
                st.write(f"- Data source: {comp['data_source']}")
                st.write(f"- Avg difference: {comp['avg_difference']:.2f} MW")
                st.write(f"- Max difference: {comp['max_difference']:.2f} MW")
                
                if comp['avg_difference'] > 1000:
                    st.warning(f"⚠️ Significant change from sample {comp['sample_index']}")
                st.write("---")
        
        with col2:
            st.write("**Stored Prediction Samples:**")
            for i, sample in enumerate(history):
                st.write(f"**Sample {i+1}:** {sample['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"Source: {sample['data_source']}")
                avg_pred = np.mean(sample['predictions'])
                st.write(f"Avg predicted load: {avg_pred:.2f} MW")
                st.write("---")

    # First Graph - 4-Hour Load Forecast
    st.subheader("📊 4-Hour Load Forecast")
    
    # Plot 4-hour forecast
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    
    # Historical data (last 6 hours)
    hist_data = df.tail(6)
    hist_timestamps = hist_data["Datetime"].tolist()
    hist_loads = hist_data["PJM_Load_MW"].tolist()
    
    # Plot historical and predicted data
    ax1.plot(hist_timestamps, hist_loads, marker='o', label="Historical Load", 
            linewidth=3, color='blue', alpha=0.7, markersize=8)
    ax1.plot(future_timestamps, predictions, marker='s', label="Predicted Load", 
            linewidth=3, color='red', linestyle='--', markersize=8)
    
    # Mark current time
    ax1.axvline(x=current_time, color='green', linestyle='-', alpha=0.8, 
               label='Current Time', linewidth=3)
    
    ax1.set_title("Energy Load Forecast - Next 4 Hours", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Time", fontsize=14)
    ax1.set_ylabel("Load (MW)", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Second Graph - Weather Impact Analysis
    st.subheader("🌡️ Weather Impact Analysis")
    
    # Weather impact analysis
    base_temp = temperatures[0]
    base_humidity = humidities[0]
    
    temp_vars, temp_impacts, hum_vars, hum_impacts = analyze_weather_impact(
        df, scaler, base_temp, base_humidity)
    
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Temperature impact
    ax2.plot(temp_vars, temp_impacts, marker='o', linewidth=3, color='orange', markersize=8)
    ax2.axvline(x=base_temp, color='red', linestyle='--', alpha=0.7, 
               label=f'Current: {base_temp:.1f}°C', linewidth=3)
    ax2.set_title("Load Sensitivity to Temperature Changes", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Temperature (°C)", fontsize=14)
    ax2.set_ylabel("Predicted Load (MW)", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Humidity impact
    ax3.plot(hum_vars, hum_impacts, marker='s', linewidth=3, color='cyan', markersize=8)
    ax3.axvline(x=base_humidity, color='red', linestyle='--', alpha=0.7, 
               label=f'Current: {base_humidity:.1f}%', linewidth=3)
    ax3.set_title("Load Sensitivity to Humidity Changes", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Humidity (%)", fontsize=14)
    ax3.set_ylabel("Predicted Load (MW)", fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    st.pyplot(fig2)
