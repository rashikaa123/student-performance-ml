import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Smart Student Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------- BACKGROUND IMAGE ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.75);
            padding: 2rem;
            border-radius: 15px;
        }}

        h1 {{
            text-align: center;
            color: white;
        }}

        .stButton>button {{
            background-color: #4f46e5;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }}
        </style>
    """, unsafe_allow_html=True)

# 👉 make sure bg.jpg is in same folder
set_bg("bg.jpg")

# ---------- TITLE ----------
st.markdown("<h1>🎓 Smart Student Performance Predictor</h1>", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
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

# ---------- INPUT SECTION ----------
st.subheader("📊 Enter Student Details")

# Name input
name = st.text_input("👤 Student Name")

# Inputs in columns
col1, col2, col3 = st.columns(3)

with col1:
    study = st.number_input("📚 Study Hours", 0.0, 50.0, 5.0)

with col2:
    attendance = st.number_input("📅 Attendance %", 0.0, 100.0, 50.0)

with col3:
    score = st.number_input("📝 Score", 0.0, 100.0, 50.0)

# ---------- PREDICTION ----------
if st.button("🔍 Predict Result"):

    if name.strip() == "":
        st.warning("⚠ Please enter student name")
    else:
        prediction = model.predict([[study, attendance, score]])

        if prediction[0] == 1:
            st.success(f"🎉 {name}, you PASSED!")
            st.balloons()

            st.markdown("""
            ### 🌟 Excellent Work!
            - Keep your consistency 💪  
            - Aim for higher scores 🚀  
            - You’re doing great 👏  
            """)

        else:
            st.error(f"❌ {name}, you need improvement")

            st.markdown("""
            ### 📚 Improvement Tips:
            - Study more regularly ⏳  
            - Maintain attendance 📅  
            - Revise concepts 🧠  
            - Practice questions 📝  
            """)

        # Progress bar (visual)
        st.progress(min(int(score), 100))

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("✨ Built with Streamlit | ML Project")