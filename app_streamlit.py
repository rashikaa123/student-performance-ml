import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("🎓 Student Performance Predictor")

# Load dataset safely
try:
    data = pd.read_csv("student_performance.csv")
except:
    st.warning("Dataset not found. Using sample data...")
    data = pd.DataFrame({
        'weekly_self_study_hours': [10, 5, 15],
        'attendance_percentage': [80, 50, 90],
        'total_score': [75, 40, 85],
        'grade': ['B', 'F', 'A']
    })

# Clean columns
data.columns = data.columns.str.strip().str.lower()

# Convert grade → Pass/Fail
data['result'] = data['grade'].map({
    'A': 1, 'B': 1, 'C': 1,
    'D': 0, 'F': 0
})

# Train model
X = data[['weekly_self_study_hours', 'attendance_percentage', 'total_score']]
y = data['result']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Inputs
study = st.number_input("Weekly Study Hours", 0.0, 50.0)
attendance = st.number_input("Attendance %", 0.0, 100.0)
score = st.number_input("Total Score", 0.0, 100.0)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[study, attendance, score]])
    result = "Pass ✅" if prediction[0] == 1 else "Fail ❌"
    st.success(f"Result: {result}")