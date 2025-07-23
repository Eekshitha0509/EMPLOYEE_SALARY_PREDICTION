import streamlit as st
import numpy as np
import joblib

# Load the model and encoders
model = joblib.load("salary_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_workclass = joblib.load("le_workclass.pkl")
le_occupation = joblib.load("le_occupation.pkl")

# Set page configuration
st.set_page_config(page_title="Salary Prediction Pro", page_icon="💼", layout="wide")

# --- Sidebar Branding ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/briefcase.png", width=80)
    st.title("Salary Predictor")
    st.markdown("🚀 Predict if an employee earns more than ₹50K based on key attributes.")
    st.markdown("---")
    st.markdown("👩‍💻 Built by [Eekshitha Namala](https://www.linkedin.com/in/eekshitha-namala-2b363626b/)")
    st.markdown("🌐 [GitHub](https://github.com/Eekshitha0509)")
    st.markdown("✉️ eekshithanamala3@gmail.com")

# --- Header ---
st.markdown(
    "<h1 style='text-align: center; color:#007BFF;'>💰 Employee Salary Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color:gray;'>A smart way to estimate employee income category using Machine Learning 📊</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Input Section ---
st.markdown("### 🧾 Fill in Employee Details Below")

col1, col2 = st.columns([1, 1])

with col1:
    age = st.slider("🎂 Age", 17, 75, 30)
    educational_num = st.slider("🎓 Education Level (1=Preschool, 16=Doctorate)", 1, 16, 10)
    hours_per_week = st.slider("⏱️ Hours Worked per Week", 1, 100, 40)
    gender = st.selectbox("🧑 Gender", le_gender.classes_)

with col2:
    workclass = st.selectbox("🏢 Workclass", le_workclass.classes_)
    occupation = st.selectbox("💼 Occupation", le_occupation.classes_)
    capital_gain = st.number_input("📈 Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("📉 Capital Loss", min_value=0, value=0)

# --- Encode Inputs ---
gender_val = le_gender.transform([gender])[0]
workclass_val = le_workclass.transform([workclass])[0]
occupation_val = le_occupation.transform([occupation])[0]

# Maintain exact order used in training
features = np.array([[age, workclass_val, educational_num, occupation_val, gender_val, capital_gain, capital_loss, hours_per_week]])

# --- Prediction ---
st.markdown("### 🔎 Prediction Result")

if st.button("💡 Predict"):
    prediction = model.predict(features)[0]
    result = "> ₹50K" if prediction == 1 else "≤ ₹50K"
    proba = model.predict_proba(features)[0][prediction] if hasattr(model, "predict_proba") else None

    # Styled response
    st.markdown("---")
    st.subheader("📊 Result Summary")
    if prediction == 1:
        st.success("🟢 The employee is likely to earn **more than ₹50K**.")
        st.metric("Prediction", result, delta="+ High Income", delta_color="normal")
        st.balloons()
    else:
        st.warning("🔴 The employee is likely to earn **₹50K or less**.")
        st.metric("Prediction", result, delta="- Low Income", delta_color="inverse")

    if proba:
        st.progress(min(int(proba * 100), 100))
        st.caption(f"Confidence: {proba:.2%}")

    st.markdown("#### 💼 Details Entered:")
    st.markdown(f"""
    - **Age**: {age}
    - **Education Level**: {educational_num}
    - **Gender**: {gender}
    - **Workclass**: {workclass}
    - **Occupation**: {occupation}
    - **Capital Gain**: ₹{capital_gain}
    - **Capital Loss**: ₹{capital_loss}
    - **Hours/Week**: {hours_per_week}
    """)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Made with ❤️ using Streamlit · © 2025 Eekshitha Namala</div>",
    unsafe_allow_html=True
)
