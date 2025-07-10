# 🎓 Student Dropout Prediction Project

A machine learning project that predicts whether a student will drop out, graduate, or remain enrolled in an undergraduate program, based on socio-economic, academic, and demographic data collected at the time of enrollment.

---

## 📌 Project Objective

The goal of this project is to **predict student dropout** using various features including:

- Academic background
- Socioeconomic factors
- Demographic data
- Early academic performance

This model can assist educational institutions in early intervention and resource planning.

---

## 🧠 Model Details

- **Algorithm**: Multi-Layer Perceptron (MLP)
- **Frameworks**: Scikit-learn, Pandas, NumPy
- **Model Specs**:
  - `hidden_layer_sizes=(80, 10)`
  - `activation='relu'`
  - `solver='adam'`
  - `learning_rate='adaptive'`

The final model and preprocessing pipelines were serialized using `joblib` for reuse and deployment.

---

## 🧼 Data Preprocessing

The dataset underwent several preprocessing steps:

- **Encoding**: 
  - Nominal categorical features encoded using One-Hot (dummies) method.
  - Ordinal categorical features encoded with ordinal mappings.
- **Scaling**: Numerical features scaled using `StandardScaler`.
- **Outlier Removal**: Outliers in key numerical features were removed based on thresholds or statistical techniques.
- **Feature Selection**: Irrelevant / low-variance features were dropped.

---

## 📁 Directory Structure
```
Student's-Drop_Out_predictor/
├── Data/                    # Dataset directory
│   └── data.csv             # Source dataset
|
├── model/                   # Contains all model-related files
│   ├── model.pkl            # Trained MLP model
│   ├── scaler.pkl           # Scaler for numerical features
│   ├── nominal_encoder.pkl  # Encoder for nominal features
│   └── ordinal_encoder.pkl  # Encoder for ordinal features
│
├── predict.py               # Script to load model & make predictions
├── project_notebook.ipynb   # EDA, preprocessing & training steps
├── .gitignore
├── README.md
└── requirements.txt         # required packages and libraries for running project locally

```

---

## 🛠 How to Use

1. **Clone the Repository**

```bash
git clone https://github.com/youssef6316/student-dropout-prediction.git
cd Student-s-drop-out-predictor
```
2.  **Install Dependencies**
```bash
pip install -r requirements.txt
```
3. Make Predictions
```bash
python predict.py
```