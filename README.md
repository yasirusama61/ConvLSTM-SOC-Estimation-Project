# 🔋 SOC Estimation using ConvLSTM

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8-blue" alt="Python">
  <img src="https://img.shields.io/badge/tensorflow-2.x-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/status-In_Progress-yellow" alt="Status">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen" alt="Contributions">
</p>

---

This repository implements a **ConvLSTM-based model** for **State of Charge (SOC) Estimation** in Lithium-ion batteries.  
ConvLSTM combines the strengths of convolutional and LSTM architectures to model both spatial and temporal dependencies, achieving **highly accurate SOC predictions**.

---

## 📖 Motivation

Accurate SOC estimation is critical for optimizing battery performance, lifespan, and safety. While traditional **Temporal CNNs** and **LSTMs** were successful in modeling battery data, we wanted to explore whether a **hybrid architecture like ConvLSTM** could further improve prediction accuracy.

### 💡 Why ConvLSTM?

#### 🚀 Previous Models:
- **Temporal CNN**: Efficient for local temporal features but struggled with long-term dependencies.
- **LSTM**: Good at capturing temporal dependencies but lacked spatial feature extraction capabilities.

#### 🔄 ConvLSTM Advantages:
1. **Hybrid Design**:
   - Integrates **Convolutional layers** for spatial features and **LSTM** for temporal learning.
2. **Enhanced Accuracy**:
   - Achieves the lowest error metrics among tested architectures.
3. **Robust Performance**:
   - Reliable results across diverse operating conditions.
4. **Reduced Noise**:
   - Superior at minimizing fluctuations during rapid SOC transitions.

---

# 📂 Project Structure

The **SOC Estimation using ConvLSTM** project is organized as follows:

- **`models/`**  
  - Stores trained model files and checkpoints.
  - Example:
    - `convolstm_model.keras` - Best ConvLSTM model.

- **`scripts/`**  
  - Contains Python scripts for reproducibility.
  - Example:
    - `train.py` - Script for training the ConvLSTM model.
    - `evaluate.py` - Script for model evaluation and metrics calculation.
    - `plot_results.py` - Script for generating plots.

- **`results/`**  
  - Saved plots and metrics for analysis.
  - Organized into subfolders:
    - `training_loss.png` - Training and validation loss plot.
    - `residual_analysis/` - Residual plots across different temperatures.
    - `soc_predictions/` - SOC prediction plots for various temperature conditions.

- **`README.md`**  
  - Contains the project documentation, including descriptions, results, and insights.

- **`requirements.txt`**  
  - Lists all the dependencies required for running the project.

- **`LICENSE`**  
  - Contains license information for the project.

- **`.gitignore`**  
  - Specifies files and folders to be ignored by Git.

---

## 🏗️ Model Architecture

### 🚀 Input Data
- **Sequence Features**:  
  Voltage, Current, Temperature, SOC Rolling Average, Current Rolling Average.  
- **Target**:  
  State of Charge (SOC).  

### 🔄 Input Sequence Preparation
- Input sequences were created using a **sliding window approach** to capture temporal dependencies. 
- **Window Size**:  
  Each sequence consists of **100 time steps** to effectively learn SOC trends.  
- **Shape Transformation**:  
  After preprocessing, the sequences were reshaped into **ConvLSTM-compatible format**:  
  `(samples, time steps, rows, cols, channels)`.

---

### 🧠 Updated ConvLSTM Model Architecture

The ConvLSTM model architecture combines convolutional layers and LSTM layers for capturing **spatio-temporal dependencies**, followed by **global average pooling** for dimensionality reduction. Here’s a breakdown of the model:

#### **Model Layers**:

| **Layer Name**            | **Description**                                                                                      |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| **Input Layer**           | Accepts sequences in the shape `(samples, time steps, rows, cols, channels)`.                       |
| **ConvLSTM Layer**        | Extracts spatio-temporal dependencies:                                                              |
|                           | - **Filters**: 64                                                                                  |
|                           | - **Kernel Size**: (1, 3)                                                                           |
|                           | - **Activation**: ReLU                                                                              |
|                           | - **Padding**: Same                                                                                 |
|                           | - **Regularization**: L2 (0.001)                                                                    |
| **Global Average Pooling**| Aggregates global information from feature maps for improved stability and reduced dimensionality.   |
| **Dense Layer 1**         | Fully connected layer:                                                                              |
|                           | - **Units**: 64                                                                                     |
|                           | - **Activation**: ReLU                                                                              |
|                           | - **Regularization**: L2 (0.001)                                                                    |
| **Dropout Layer 1**       | Applies dropout to reduce overfitting: **Rate**: 0.3.                                               |
| **Dense Layer 2**         | Fully connected layer:                                                                              |
|                           | - **Units**: 32                                                                                     |
|                           | - **Activation**: ReLU                                                                              |
|                           | - **Regularization**: L2 (0.001)                                                                    |
| **Dropout Layer 2**       | Applies dropout to reduce overfitting: **Rate**: 0.3.                                               |
| **Output Layer**          | Final prediction layer:                                                                             |
|                           | - **Units**: 1                                                                                      |
|                           | - **Activation**: Sigmoid (for SOC predictions normalized between 0 and 1).                         |

---

### 🛠️ Model Compilation
- **Loss Function**: Mean Squared Error (MSE).  
- **Optimizer**: Adam with a learning rate of **0.001**.  
- **Metrics**:  
  - Mean Absolute Error (MAE).  
  - R² (Coefficient of Determination).  
  - RMSE (Root Mean Squared Error).

This enhanced architecture, coupled with temperature-specific features and robust regularization, has significantly reduced prediction errors and improved the model's generalization across varying temperature conditions.

---

## 📉 Training Loss Analysis

The training and validation loss plot provides critical insights into the model's learning behavior during the training process.

![Training Loss Plot](results/training_plot.png)

### Observations:
1. **Initial High Loss:** 
   - The training loss starts at a relatively high value, reflecting the initial randomness of the model weights.
   - The validation loss decreases rapidly in the initial epochs, indicating the model is learning useful patterns.

2. **Smooth Decrease in Training Loss:** 
   - The training loss steadily decreases as the epochs progress, demonstrating the model's ability to optimize its parameters effectively.

3. **Validation Loss Stability:**
   - The validation loss stabilizes after approximately 10 epochs, with minor fluctuations, indicating that the model is reaching a point of generalization.

4. **Overfitting Risk:**
   - In the later epochs (20–25), the validation loss begins to increase slightly, while the training loss continues to decrease. This may suggest early signs of overfitting.

### Conclusion:
The training plot highlights the efficiency of the ConvLSTM architecture in learning temporal and spatial dependencies for SOC estimation. However, early stopping and regularization strategies are critical to avoid overfitting and improve generalization on unseen data.


## 📊 Results and Insights

### Overview
The **ConvLSTM** model was evaluated for **State of Charge (SOC)** estimation under various temperature conditions: **-10°C, 0°C, 10°C, and 25°C**. The results demonstrate the model's strong performance and improvements over previous approaches. Below, we provide detailed metrics, visualizations, and insights.

---

### 🏆 Performance Metrics

| 🌡️ Temperature | 📉 MAE (Mean Absolute Error) | 📊 MSE (Mean Squared Error) | 📈 R² (R-squared) | 🔄 RMSE (Root Mean Squared Error) | 🎯 Average Error (%) |
|-----------------|------------------------------|-----------------------------|-------------------|-----------------------------------|---------------------|
| -10°C          | **0.006861**                 | **0.000099**                | **0.998376**      | **0.009948**                      | **0.6861%**         |
| 0°C            | **0.0064**                   | **0.000100**                | **0.998500**      | **0.010000**                      | **0.64%**           |
| 10°C           | **0.008792**                 | **0.000218**                | **0.997099**      | **0.014764**                      | **0.8792%**         |
| 25°C           | **0.007124**                 | **0.000153**                | **0.998181**      | **0.012368**                      | **0.7124%**         |

---

### 📉 Visual Results

#### 🔍 -10°C (Zoomed-In Instabilities)
![SOC at -10°C (Zoomed-In)](results/actual_vs_predicted_soc_at_-10Celsius_zoomed_plot.png)

#### 🌡️ 0°C
![SOC at 0°C](results/actual_vs_predicted_soc_at_0Celsius_plot.png)

#### 🌡️ 10°C
![SOC at 10°C](results/actual_vs_predicted_soc_at_10Celsius_plot.png)

#### 🌡️ 25°C
![SOC at 25°C](results/actual_vs_predicted_soc_at_25Celsius_plot.png)

---

### 🔍 Key Insights
1. **Error Reduction**:
   - The **ConvLSTM** outperformed previous models (**Temporal CNN** and **LSTM**) that had an average error rate of **1.4%**.
   - The **ConvLSTM** achieved an **average error of 0.7%**, representing a substantial improvement.

2. **Challenges at -10°C**:
   - **Observation**: At **-10°C**, fluctuations appear during low SOC regions, as shown in the zoomed-in plot.
   - **Analysis**:
     - Likely caused by **limited training data** at extreme temperatures.
     - Sensor noise or inaccuracies at low temperatures might have affected predictions.
   - **Recommendations**:
     - Enhance the training dataset with more low-temperature samples.
     - Explore **temperature-specific fine-tuning** or **data augmentation** for extreme conditions.

3. **Stability Across Other Temperatures**:
   - The model exhibited consistent accuracy at **0°C, 10°C, and 25°C**, ensuring smooth SOC predictions during charging/discharging cycles.

4. **Hybrid Architecture Advantage**:
   - Combines **spatial feature extraction** (via convolutional layers) with **temporal learning** (via LSTMs) to address complex SOC dynamics.

---

### 🔍 Residual Analysis Across Temperatures

Residual plots for SOC estimation reveal the performance of the ConvLSTM model across different temperatures:

#### 📊 Observations:
1. **-10°C:**
   - Residuals show high variability and larger deviations.
   - Indicates model struggles due to non-linear battery behavior in extreme cold.

2. **0°C:**
   - Moderate fluctuations, with residuals mostly centered around 0.
   - Some spikes indicate occasional prediction errors.

3. **10°C:**
   - Stable residuals with minor deviations, showing consistent model performance.
   - Rare outliers are observed but do not affect overall accuracy.

4. **25°C:**
   - Residuals are the most stable with minimal fluctuations.
   - Best performance, reflecting predictable battery behavior at standard conditions.

#### ⚡ Insights:
- **Best Performance:** At moderate temperatures (10°C and 25°C), residuals are minimal, highlighting model robustness.
- **Challenges:** At -10°C, higher variability suggests a need for improved handling of extreme conditions.

#### 📉 Residual Plots:
![Residual Analysis Across Temperatures](results/residual_analysis_plot.png)

#### 🔧 Next Steps:
- Incorporate temperature-specific features or data augmentation for extreme conditions.
- Analyze outliers to further enhance model performance.

This analysis emphasizes the need for temperature-aware optimization to ensure reliable SOC predictions under diverse operating conditions.

### 📊 Error Analysis: ConvLSTM SOC Estimation

The **ConvLSTM** model demonstrated significant improvements in predicting the **State of Charge (SOC)**, with tight error bounds across various temperatures. Below is the breakdown of **maximum** and **minimum errors** observed during testing:

---

#### 🔥 **Error Summary by Temperature**  

| 🌡️ Temperature | 🚩 Max Error  | 📉 Min Error  |
|-----------------|--------------|---------------|
| **-10°C**       | 0.0673       | -0.0459       |
| **0°C**         | 0.0501       | -0.0358       |
| **10°C**        | 0.0394       | -0.0525       |
| **25°C**        | 0.0441       | -0.0434       |

---

#### 🔍 **Insights**
- **Tight Prediction Bounds**: Across all temperatures, the maximum error remained within **6.73%**, showcasing the model's robustness.
- **High Accuracy at Moderate Temperatures**: At **10°C** and **25°C**, the errors were notably lower, indicating reliable predictions under standard conditions.
- **Challenges at Extreme Temperatures**: Slightly higher errors at **-10°C** highlight the need for additional optimization in extreme conditions, potentially due to sensor variability or non-linear temperature effects.

---

### 🛠️ **Is This Error Acceptable?**
The observed maximum error of **6.73%** is within acceptable limits for SOC estimation in most applications. Accurate SOC prediction is critical for maintaining battery performance and safety, and this level of precision ensures reliable system operation. However, further optimization at extreme temperatures (e.g., -10°C) can improve robustness, particularly for high-performance or safety-critical use cases.

---

### ✅ Conclusion

The **ConvLSTM** architecture has demonstrated its potential as a reliable and accurate solution for **SOC estimation**, achieving consistent performance across various temperature conditions. Its hybrid nature combines the strengths of **LSTMs** for temporal dependencies and **CNNs** for spatial feature extraction, enabling it to outperform standalone Temporal CNN and LSTM models. 

<p align="center">
  <img src="https://img.icons8.com/dusk/64/checkmark.png" alt="success" width="50">
  <b>ConvLSTM: Pushing the boundaries of SOC estimation accuracy! 🚀</b>
</p>

---

### 🌟 **Benefits**:
- **Improved Accuracy**: With an average error of **0.7%**, ConvLSTM sets a new benchmark for SOC estimation models.
- **Generalization**: Handles varying temperature conditions effectively, making it robust for real-world deployment.
- **Hybrid Design**: Combines temporal and spatial feature extraction, improving prediction smoothness and reliability.

---

### ⚠️ **Challenges and Weaknesses**:
- **Extreme Temperatures**: While robust at moderate temperatures, performance slightly declines under extreme conditions like **-10°C**, where fluctuations are more noticeable.
- **Computational Overhead**: ConvLSTM models are computationally heavier compared to simpler architectures, requiring optimized deployment strategies like quantization.
- **Hyperparameter Tuning**: Fine-tuning the architecture (e.g., kernel size, dropout rates) required significant effort to achieve the best results.
- **Edge Deployment**: While quantization and latency testing helped, deploying on low-resource devices posed challenges due to the model's complexity.

---

Despite these challenges, the **ConvLSTM model** proves to be a strong candidate for deployment in **Battery Management Systems**, offering a promising balance of accuracy, robustness, and real-world applicability. Future improvements could focus on optimizing performance at extreme temperatures and reducing computational demands for edge applications.

The results affirm the capability of ConvLSTM to deliver accurate, reliable, and actionable insights for next-generation battery systems.

---

## 🚀 Usage

### 1. Clone the Repository
   ```bash
   git clone https://github.com/yasirusama61/SOC_Estimation_ConvLSTM.git
   cd SOC_Estimation_ConvLSTM
   ```
### 2. Set Up the Environment

   Install the required dependencies using the provided requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```
### 3. Prepare the Data
   Place your input data in the data/ directory following the project structure.

### 4. Train the Model
   Run the training script to train the ConvLSTM model on your dataset:
   ```bash
   python scripts/train_model.py
   ```
### 5. Evaluate and Visualize
   Evaluate the trained model on test datasets and generate residual plots for analysis:
   ```bash
   python scripts/evaluate_model.py
   ```
### 6. View Results
   Find saved plots and metrics in the results/ directory for a detailed performance overview.

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software as per the terms of the license.

**Key Points of the MIT License:**
- Permission to use, copy, modify, and distribute the software for personal and commercial purposes.
- No warranty is provided, and the software is provided "as is."

For more details, refer to the [LICENSE](LICENSE) file included in this repository.

---
