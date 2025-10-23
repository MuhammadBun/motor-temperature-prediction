# Real-time DNN PM Temperature Prediction

A live Streamlit web application for real-time motor temperature prediction using Deep Neural Networks with interactive visualization and industrial-grade accuracy.

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

## Live Demo
**Experience the real-time prediction:** [motor-temperature-prediction.streamlit.app](https://motor-temperature-prediction.streamlit.app)

## 📊 Project Highlights

### 🎯 **Industrial-Grade Performance**
| Metric | Performance | Industrial Standard |
|--------|-------------|---------------------|
| **Mean Absolute Error (MAE)** | **1.28 °C** | ±2–3°C |
| **Root Mean Squared Error (RMSE)** | **1.78 °C** | < 2.0°C |
| **R² Score (Variance Explained)** | **0.991 (99.1%)** | > 0.95 |
| **95% Error Range** | **±3.59 °C** | ±5°C |

### **Exceeds Industrial Requirements**
- **Predicts within ±1.3°C on average** - superior accuracy
- **Explains 99.1% of temperature variance** - highly reliable
- **95% of predictions within ±3.6°C** - consistent performance
- **Validated on 1.3+ million samples** from actual PMSM operations

## Features

### Live Simulation
- **Real-time PM Temperature Prediction**: Instant temperature forecasting
- **Interactive Visualization**: Dynamic charts with threshold alerts
- **Live Performance Metrics**: Accuracy, predicted vs actual temperatures updated every 0.3 seconds
- **Alert System**: Visual indicators for critical temperature thresholds (80°C)
- **Status Monitoring**: NORMAL/ALERT indicators based on current temperature

### 🛠 Technical Features
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Real-time Smoothing**: Advanced algorithms for stable visualization
- **Customizable Display**: Adjustable figure sizes and container width
- **Professional UI**: Dark theme with high-contrast visual elements

## Demo Overview

The application displays in real-time:
- **Live temperature predictions** vs actual values with smooth animation
- **Accuracy metrics** dynamically updated during simulation
- **Critical threshold line** at 80°C for immediate safety awareness
- **Status indicators** (NORMAL/ALERT) with color-coded visual feedback
- **Performance statistics** showing model confidence and reliability

## Industrial Application

### Use Cases
- **Predictive Maintenance**: Early detection of motor overheating
- **Industrial Automation**: Real-time thermal monitoring in manufacturing
- **Energy Efficiency**: Optimize motor performance and reduce downtime
- **Safety Systems**: Prevent equipment damage through temperature alerts

### Data Source
- **Dataset**: [Electric Motor Temperature Dataset](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature)
- **Samples**: 1,337,097 sensor readings from PMSM motors
- **Features**: Voltage, current, speed, coolant, ambient temperature
- **Target**: Permanent magnet temperature (`pm`)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Development

1. **Clone the repository**
git clone https://github.com/MuhammadBun/motor-temperature-prediction.git
cd motor-temperature-prediction

2. **Install dependencies**
pip install -r requirements.txt

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the application**
streamlit run streamlit_motor_test.py

4. **Open your browser**
- Navigate to http://localhost:8501
- The simulation starts automatically

5. **Project Structure**
motor-temperature-prediction/
├── streamlit_motor_test.py     # Main Streamlit application
├── style.css                   # Custom CSS styling
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitattributes              # Git LFS configuration
├── models/
│   └── motor_dnn_best.keras    # Trained DNN model (688 KB)
└── data/
    ├── X_test_reloaded.csv     # Test features dataset (45 MB)
    └── y_test_reloaded.csv     # Test targets dataset (5 MB)

