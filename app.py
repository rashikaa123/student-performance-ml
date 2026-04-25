from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("student_performance.csv")

# Clean column names (removes spaces issues)
data.columns = data.columns.str.strip().str.lower()

# Print columns (for checking)
print("Columns in dataset:", data.columns)

# Convert grade → Pass/Fail
data['result'] = data['grade'].map({
    'A': 1,
    'B': 1,
    'C': 1,
    'D': 0,
    'F': 0
})

# Select features (safe names based on your dataset)
X = data[['weekly_self_study_hours', 'attendance_percentage', 'total_score']]
y = data['result']

# Train model
model = LogisticRegression()
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