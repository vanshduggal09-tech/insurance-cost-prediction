# 💰 Insurance Cost Prediction

A machine learning project that predicts medical insurance expenses based on
demographic and lifestyle-related features.

## 📌 Project Overview

This project uses machine learning to predict insurance expenses from
features such as age, BMI, smoking status, sex, number of children, and region.

The project includes data preprocessing, feature selection, Ridge Regression,
model evaluation, and a Streamlit web application for making predictions.

## 🚀 Features

- Data preprocessing and cleaning
- Feature selection
- StandardScaler preprocessing
- Ridge Regression with regularization
- Model evaluation using:
  - R²
  - Adjusted R²
  - MAE
  - RMSE
- Interactive Streamlit UI
- Real-time insurance cost prediction

## 🧠 Model Features

The final model uses:

- `age`
- `bmi`
- `isfemale`
- `issmoker`
- `children`
- `issoutheast`

Categorical variables are converted into numerical features before prediction.

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib / Seaborn

## 📊 Model

The final model is based on **Ridge Regression** with StandardScaler
preprocessing.

The preprocessing and model are combined using a Scikit-learn Pipeline,
which ensures that new inputs are transformed consistently before prediction.

## 🖥️ Streamlit Application

The project includes an interactive web interface where users can enter:

- Age
- BMI
- Sex
- Smoking status
- Number of children
- Region

The application then predicts the estimated insurance cost.


--- Project Structure

insurance-project/
│
├── app.py
├── insurance_model.pkl
├── insurance.csv
├── 01.ipynb
└── README.md