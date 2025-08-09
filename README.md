# 🌬️ Air Quality Prediction using Artificial Neural Network (NumPy Only)

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN) from scratch** using **only NumPy** to predict the **Air Quality Index (AQI)** based on pollutant concentration levels.  
No deep learning frameworks (TensorFlow, PyTorch, etc.) are used — all **forward propagation, backpropagation, and weight updates** are coded manually.

The dataset contains **hourly air quality measurements** from multiple cities, including:

- **CO**
- **NO₂**
- **SO₂**
- **O₃**
- **PM2.5**
- **PM10**
- **AQI (target variable)**

---

## 📂 Dataset

**File:** `Air_Quality.csv`  
**Size:** 52,560 rows × 9 columns

| Column | Description                             |
| ------ | --------------------------------------- |
| Date   | Timestamp of the observation            |
| City   | Name of the city                        |
| CO     | Carbon Monoxide concentration           |
| NO₂    | Nitrogen Dioxide concentration          |
| SO₂    | Sulphur Dioxide concentration           |
| O₃     | Ozone concentration                     |
| PM2.5  | Fine particulate matter concentration   |
| PM10   | Coarse particulate matter concentration |
| AQI    | Air Quality Index (target)              |

---

## 🧠 Model Architecture

- **Input Layer:** Encoded pollutant and city features
- **Hidden Layers:** Fully connected layers with **ReLU** activation
- **Output Layer:** Single neuron for AQI prediction
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Gradient Descent (manual implementation)

---

## ⚙️ Implementation Steps

1. **Data Preprocessing**

   - Drop/encode non-numeric columns (`Date`, `City`)
   - Normalize numerical features
   - Train-test split

2. **ANN from Scratch (NumPy)**

   - Initialize weights & biases
   - Forward propagation
   - Loss calculation (MSE)
   - Backpropagation (manual gradient computation)
   - Parameter update with Gradient Descent

3. **Model Training**

   - Run for defined number of epochs
   - Track loss over time

4. **Evaluation**
   - Predict AQI values on test set
   - Compare predicted vs. actual values

---

## 📊 Expected Output

- **Loss curve** showing training progress
- **Predicted vs Actual AQI** comparison
- **Performance metrics**: MSE, MAE, R² score

---

## 🛠 Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib** (for visualization)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/air-quality-ann-numpy.git
cd air-quality-ann-numpy

# Install dependencies
pip install -r requirements.txt

# Run the model
python train_ann.py
```
