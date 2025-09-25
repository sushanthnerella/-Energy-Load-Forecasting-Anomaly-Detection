# -Energy-Load-Forecasting-Anomaly-Detection

# ⚡ Energy Load Forecasting & Anomaly Detection

This project focuses on **predicting short-term energy demand** and **detecting anomalies/spikes** in energy consumption using advanced deep learning techniques.

## 🚀 Features

* **LSTM-based Forecasting**: Predicts energy load **1 hour in advance**.
* **Anomaly Detection**: Identifies sudden spikes/drops in consumption.
* **Auto-Alerts**: Sends notifications via **Email/SMS** when anomalies are detected.
* **Visualization Dashboard**: Interactive plots to monitor load forecasts and anomalies in real-time.

## 🛠 Tech Stack

* **Python**, **TensorFlow/Keras** (LSTM Model)
* **Pandas**, **NumPy**, **Matplotlib/Seaborn** (Data Analysis & Visualization)
* **Streamlit / Flask** (Dashboard/Deployment)
* **Twilio / SMTP** (SMS & Email Alerts)
* **Scikit-learn** (Preprocessing & Evaluation Metrics)

## 📊 Workflow

1. **Data Preprocessing**

   * Load historical energy consumption + weather features.
   * Normalize & window the time series for LSTM input.

2. **Model Training**

   * LSTM model trained to predict **next 1-hour energy demand**.
   * Evaluation with RMSE, MAPE.

3. **Anomaly Detection**

   * Compare actual vs. predicted values.
   * Mark deviations above a threshold as anomalies.

4. **Alerts & Monitoring**

   * Trigger SMS/Email notifications for detected anomalies.
   * Display forecasts & anomalies in a live dashboard.

## 📂 Project Structure

```
Energy-Forecasting-Anomaly-Detection/
│── data/                # Raw & preprocessed datasets  
│── notebooks/           # Jupyter notebooks for EDA & experiments  
│── src/  
│   ├── preprocessing.py # Data cleaning & feature engineering  
│   ├── model.py         # LSTM model training & prediction  
│   ├── anomaly.py       # Anomaly detection logic  
│   ├── alerts.py        # Email/SMS notification handler  
│── app.py               # Streamlit/Flask app  
│── requirements.txt     # Dependencies  
│── README.md            # Project documentation  
```

## ⚡ Example Output

* Forecasted vs Actual energy load curve.
* Highlighted anomaly regions.
* Auto-generated alerts with timestamps.

## 📩 Alerts Example

* **Email**: "⚠️ Energy spike detected at 14:00 hrs. Load: 35% higher than forecast."
* **SMS**: "Energy anomaly detected at 14:00 hrs. Check dashboard."

## 🔮 Future Improvements

* Multi-step forecasting (e.g., 6–24 hrs ahead).
* Integration with cloud platforms (AWS/GCP) for scalability.
* Adaptive anomaly thresholds using probabilistic models.

---
