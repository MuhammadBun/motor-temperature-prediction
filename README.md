# Real-time DNN PM Temperature Prediction

A Streamlit web application for real-time motor temperature prediction using Deep Neural Networks with industrial-grade accuracy.

**Live Demo:** [motor-temperature-prediction.streamlit.app](https://motor-temperature-prediction.streamlit.app)

## Performance Summary

| Metric | Performance | Industrial Standard |
|--------|-------------|---------------------|
| Mean Absolute Error (MAE) | 1.28 °C | ±2–3°C |
| Root Mean Squared Error (RMSE) | 1.78 °C | < 2.0°C |
| R² Score (Variance Explained) | 0.991 (99.1%) | > 0.95 |
| 95% Error Range | ±3.59 °C | ±5°C |

**Key Achievements:**
- Predicts within ±1.3°C on average
- Explains 99.1% of temperature variance  
- 95% of predictions within ±3.6°C
- Validated on 1.3+ million industrial samples

## Features

**Live Simulation:**
- Real-time PM temperature forecasting
- Dynamic charts with 80°C threshold alerts
- Accuracy metrics updated every 0.3 seconds
- NORMAL/ALERT status indicators

**Technical Features:**
- Responsive design for all devices
- Real-time data smoothing
- Customizable display settings
- Professional dark theme UI

## Installation

**Prerequisites:**
- Python 3.8+
- pip package manager

**Local Development:**
```bash
git clone https://github.com/MuhammadBun/motor-temperature-prediction.git
cd motor-temperature-prediction
pip install -r requirements.txt
streamlit run streamlit_motor_test.py
```

### Project Structure
    motor-temperature-prediction/
    ├── streamlit_motor_test.py
    ├── style.css
    ├── requirements.txt
    ├── README.md
    ├── .gitattributes
    ├── models/
    │   └── motor_dnn_best.keras
    └── data/
        ├── X_test_reloaded.csv
        └── y_test_reloaded.csv