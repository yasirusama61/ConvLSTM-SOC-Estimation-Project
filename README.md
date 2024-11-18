# 🔋 SOC Estimation using ConvLSTM

This repository implements a **ConvLSTM-based model** for **State of Charge (SOC) Estimation** in Lithium-ion batteries. ConvLSTM combines the strengths of convolutional and LSTM architectures to model both spatial and temporal dependencies, achieving highly accurate SOC predictions.

---

## 📖 Motivation

Accurate SOC estimation is critical for optimizing battery performance, lifespan, and safety. While traditional **Temporal CNNs** and **LSTMs** were successful in modeling battery data, we wanted to explore whether a **hybrid architecture like ConvLSTM** could further improve prediction accuracy.

### Why ConvLSTM?
- **Temporal Dependency**: LSTM layers capture long-term dependencies in sequential data.
- **Spatial Features**: Convolutional layers effectively extract spatial features.
- **Hybrid Approach**: ConvLSTM combines both, making it ideal for capturing intricate relationships in battery data.
- **Performance**: With an average **Mean Absolute Error (MAE)** of ~0.0073 across varying temperatures, ConvLSTM outperforms standalone architectures like Temporal CNN and LSTM.

---

## 📂 Project Structure
SOC_Estimation_ConvLSTM/
   ├── data/                      # Input datasets
   ├── models/                    # Trained models
   ├── notebooks/                 # Jupyter notebooks
   ├── scripts/                   # Python scripts
   ├── results/                   # Saved plots and metrics
   ├── README.md                  # Project description
   ├── requirements.txt           # Dependencies
   ├── LICENSE                    # License information
   └── .gitignore                 # Ignored files and folders

---

## 🏗️ Model Architecture

### Input Data:
- **Sequence Features**: Voltage, Current, Temperature, SOC Rolling Average, Current Rolling Average.
- **Target**: State of Charge (SOC).

### ConvLSTM Model:
1. **Input Layer**:
   - Accepts sequences in the shape `(samples, time steps, rows, cols, channels)`.

2. **ConvLSTM Layer**:
   - Extracts spatio-temporal features.

3. **Dense Layers**:
   - **Dense Layer 1**:
     - Units: 64, Activation: ReLU, Regularization: L2.
   - **Dropout Layer 1**: Rate: 0.3.
   - **Dense Layer 2**:
     - Units: 32, Activation: ReLU, Regularization: L2.
   - **Dropout Layer 2**: Rate: 0.3.

4. **Output Layer**:
   - Units: 1, Activation: Linear (SOC prediction).

### Model Compilation:
- **Loss**: Mean Squared Error (MSE).
- **Optimizer**: Adam with learning rate `0.001`.
- **Metrics**: MAE, R².

---

## 📊 Results

### Performance Metrics:
| Temperature | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | R-squared (R²) |
|-------------|----------------------------|---------------------------|----------------|
| -10°C       | 0.006861                  | 0.000099                 | 0.998376       |
| 25°C        | 0.007124                  | 0.000153                 | 0.998181       |
| 10°C        | 0.008792                  | 0.000218                 | 0.997099       |

### Visualizations:
Plots comparing actual vs. predicted SOC are available in the [`results/`](results/) folder.

---

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yasirusama61/SOC_Estimation_ConvLSTM.git
cd SOC_Estimation_ConvLSTM
