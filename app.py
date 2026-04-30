from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

try:
    data = pd.read_csv("student_performance.csv")
except:
    print("Dataset not found. Using sample data...")
    data = pd.DataFrame({
        'weekly_self_study_hours': [10, 5, 15],
        'attendance_percentage': [80, 50, 90],
        'total_score': [75, 40, 85],
        'grade': ['B', 'F', 'A']
    })

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Convert grade → Pass/Fail
data['result'] = data['grade'].map({
    'A': 1,
    'B': 1,
    'C': 1,
    'D': 0,
    'F': 0
})

# Features
X = data[['weekly_self_study_hours', 'attendance_percentage', 'total_score']]
y = data['result']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study = float(request.form['study'])
    attendance = float(request.form['attendance'])
    score = float(request.form['score'])

    prediction = model.predict([[study, attendance, score]])
    result = "Pass" if prediction[0] == 1 else "Fail"

    return render_template('index.html', prediction_text="Result: " + result)

if __name__ == "__main__":
    app.run(debug=True)