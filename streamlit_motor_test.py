import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.models import load_model

# =========================
# Load external CSS
# =========================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# Plotting function - OPTIMIZED
# =========================
def plot_live(collected_preds, collected_actuals, threshold=80, fig_width=8, fig_height=4, use_container_width=False):
    # Create figure with optimized settings for speed
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=80, facecolor="#051224")  # Reduced DPI
    ax.set_facecolor("#051224")

    # Shadow background
    fig.patches.extend([Rectangle(
        (0, 0), 1, 1, transform=fig.transFigure,
        facecolor='#0a1a2f', alpha=0.6, zorder=-1
    )])

    # Line colors
    color = "lime" if (collected_preds[-1] <= threshold) else "red"

    # Plot data with optimized parameters
    ax.plot(collected_preds, label="Predicted", color=color, linewidth=1.5)
    ax.plot(collected_actuals, label="Actual", color="white", alpha=0.9, linewidth=1.5)
    ax.axhline(y=threshold, color="white", linestyle="--", linewidth=3.0, label="Threshold")

    # Grid and limits
    ax.grid(True, color="#4c5663", linestyle=':', linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)

    # White box in upper left corner with opacity
    white_box = Rectangle((0.02, 0.85), 0.25, 0.12, transform=fig.transFigure, 
                         facecolor='white', alpha=0.2, edgecolor='white', linewidth=1.5, zorder=10)
    fig.patches.append(white_box)

    # Add text inside the white box
    current_temp = collected_preds[-1] if collected_preds else 0
    status = "NORMAL" if current_temp <= threshold else "ALERT"
    
    ax.text(0.04, 0.93, f"Status: {status}", transform=fig.transFigure, 
            fontsize=12, fontweight='bold', color='white', zorder=11)
    ax.text(0.04, 0.88, f"Temp: {current_temp:.1f}°C", transform=fig.transFigure, 
            fontsize=10, fontweight='bold', color='white', zorder=11)

    # Titles and labels - ALL WHITE TEXT
    ax.set_title("Live DNN PM Temperature Prediction", color="white", fontsize=22, pad=25, fontweight='bold')
    ax.set_xlabel("Time (sample)", color="white", fontsize=16, fontweight='semibold', labelpad=15)  # WHITE
    ax.set_ylabel("PM Temperature (°C)", color="white", fontsize=16, fontweight='semibold', labelpad=15)  # WHITE

    # Set axis tick colors to white
    ax.tick_params(axis='x', colors='white', labelsize=12)  # WHITE X-axis
    ax.tick_params(axis='y', colors='white', labelsize=12)  # WHITE Y-axis

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=6, label='Predicted'),
        plt.Line2D([0], [0], color='white', lw=5, label='Actual'),  # Changed from cyan to white
        plt.Line2D([0], [0], color='white', linestyle='--', lw=4, label='Threshold')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right', framealpha=0.97, frameon=True,
              facecolor='#0a1a2f', edgecolor='#2a4a6f', fontsize=14, borderpad=2, handlelength=2, handletextpad=1)
    
    # Set legend text color to white
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout(pad=4.0)

    return fig, use_container_width

# =========================
# Streamlit app
# =========================
def main():
    st.set_page_config(
        page_title="Real-time DNN PM Prediction", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Load CSS
    local_css("style.css")

    st.title("Real-time DNN PM Temperature Prediction Simulator")

    # Paths 
    features_path = "data/X_test_reloaded.csv"
    target_path = "data/y_test_reloaded.csv"
    model_path = "model/motor_dnn_best.keras"

    # Loading spinner
    with st.spinner("Loading model and datasets... Please wait"):
        time.sleep(0.3)

        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}")
            return
        model = load_model(model_path)

        if not os.path.exists(features_path):
            st.error(f"Features CSV not found at {features_path}")
            return
        X_test = pd.read_csv(features_path).values

        if not os.path.exists(target_path):
            st.error(f"Target CSV not found at {target_path}")
            return
        y_test = pd.read_csv(target_path).values.flatten()

    st.success("Model and datasets loaded successfully!")
    st.write(f"Loaded {len(X_test)} samples with {X_test.shape[1]} features.")

    # Simulation parameters
    threshold = 80
    smoothing_factor = 15

    # Optional: user input for figure size
    fig_width = st.sidebar.slider("Figure width", min_value=6, max_value=40, value=20)
    fig_height = st.sidebar.slider("Figure height", min_value=4, max_value=20, value=10)
    use_container_width = st.sidebar.checkbox("Stretch to container width", value=False)

    # Placeholders
    chart_placeholder = st.empty()
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metric_pred = metrics_col1.empty()
    metric_actual = metrics_col2.empty()
    metric_acc = metrics_col3.empty()

    # Initialize
    i = 0
    collected_preds = []
    collected_actuals = []
    last_pred = None
    last_actual = None
    last_update_time = time.time()

    # Auto-start the simulation
    st.session_state.running = True

    # Simulation loop - OPTIMIZED
    while st.session_state.running and i < len(X_test):
        x = X_test[i].reshape(1, -1)
        pred = model.predict(x, verbose=0)[0][0]
        actual = y_test[i]

        # Smoothing
        if last_pred is not None and smoothing_factor > 1:
            pred = last_pred + (pred - last_pred) / smoothing_factor
            actual = last_actual + (actual - last_actual) / smoothing_factor
        last_pred = pred
        last_actual = actual

        collected_preds.append(pred)
        collected_actuals.append(actual)

        # Plot figure only every 3 samples for faster updates
        if i % 3 == 0:  # Reduced plot frequency
            fig, use_container_width_flag = plot_live(
                collected_preds, collected_actuals, threshold,
                fig_width=fig_width, fig_height=fig_height,
                use_container_width=use_container_width
            )
            chart_placeholder.pyplot(fig, use_container_width=use_container_width_flag)
            plt.close(fig)

        # Update metrics every 0.3 sec
        if time.time() - last_update_time >= 0.3:
            accuracy = max(0, 100 * (1 - abs(actual - pred) / max(actual, 1e-5)))

            metric_pred.markdown(
                f"""
                <div class="scoreboard-box">
                    <div class="score-value">{pred:.2f}°C</div>
                    <div class="score-label">Predicted</div>
                </div>
                """, unsafe_allow_html=True
            )
            metric_actual.markdown(
                f"""
                <div class="scoreboard-box">
                    <div class="score-value">{actual:.2f}°C</div>
                    <div class="score-label">Actual</div>
                </div>
                """, unsafe_allow_html=True
            )
            metric_acc.markdown(
                f"""
                <div class="scoreboard-box">
                    <div class="score-value-green">{accuracy:.1f}%</div>
                    <div class="score-label-green">Accuracy</div>
                </div>
                """, unsafe_allow_html=True
            )

            last_update_time = time.time()

        i += 1
        time.sleep(0.005)  # Reduced sleep time for faster simulation

    # Show completion message when done
    if i >= len(X_test):
        st.success("Simulation completed!")

if __name__ == "__main__":
    main()