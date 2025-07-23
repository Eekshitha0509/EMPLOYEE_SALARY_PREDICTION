# EMPLOYEE_SALARY_PREDICTION
Summer internship 

# 💼 Employee Salary Prediction Web App

A machine learning-powered web application that predicts whether an employee earns more than ₹50K based on key demographic and employment details. Built using Streamlit for the frontend and trained with scikit-learn.

---

## 📊 Features

- Predicts employee income category (≤ ₹50K or > ₹50K)
- Interactive sliders and dropdowns to input data
- Clean and responsive UI using Streamlit
- Confidence indicator and visual feedback
- Trained using RandomForestClassifier with Label Encoding
- End-to-end pipeline integration and model deployment

---

## 🛠️ Tech Stack

| Technology     | Description                             |
|----------------|-----------------------------------------|
| **Python**     | Core programming language               |
| **Pandas**     | Data manipulation and cleaning          |
| **Matplotlib** | Data visualization (box plots, etc.)    |
| **Scikit-learn** | Model training and evaluation         |
| **Joblib**     | Model and encoder serialization         |
| **Streamlit**  | Interactive frontend for model serving  |

---

## 📂 Project Structure

employee-salary-prediction/
│
├── salary_model.pkl # Trained ML model

├── le_gender.pkl # Gender encoder

├── le_workclass.pkl # Workclass encoder

├── le_occupation.pkl # Occupation encoder

├── app.py # Streamlit web app

├── salary_prediction.py # Model training script

├── adult 3.csv # Dataset

