import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Dashboard", layout="wide")

st.title("🎓 Student Performance Dashboard")

# Load data
df = pd.read_csv("final_student_realistic_v3.csv")

# -------------------------
# 🧹 DATA CLEANING
# -------------------------
for col in ['sessional1', 'sessional2', 'sessional3']:
    df[col] = df[col].clip(0, 30)

df['attendance'] = df['attendance'].clip(0, 100)
df['avg'] = df[['sessional1','sessional2','sessional3']].mean(axis=1)

# -------------------------
# 🔍 SEARCH / SELECT STUDENT
# -------------------------
st.subheader("🔍 Select Student")

student_name = st.selectbox("Choose Student", df['name'].unique())

student = df[df['name'] == student_name].iloc[0]

# -------------------------
# 📊 STUDENT INFO
# -------------------------
st.subheader("📋 Student Details")

col1, col2, col3 = st.columns(3)

col1.metric("Roll No", student['roll'])
col2.metric("Attendance (%)", student['attendance'])
col3.metric("Average Marks", round(student['avg'], 2))

# -------------------------
# 📊 INDIVIDUAL GRAPH
# -------------------------
st.subheader("📊 Sessional Performance")

marks_df = pd.DataFrame({
    "Sessionals": ["S1", "S2", "S3"],
    "Marks": [student['sessional1'], student['sessional2'], student['sessional3']]
})

st.bar_chart(marks_df.set_index("Sessionals"))

# -------------------------
# 📈 PERFORMANCE TREND
# -------------------------
st.subheader("📈 Performance Trend")

st.line_chart(marks_df.set_index("Sessionals"))

# -------------------------
# 📉 ATTENDANCE VS AVG (only highlight student)
# -------------------------
st.subheader("📉 Attendance vs Performance")

chart_df = df[['attendance','avg']]
st.scatter_chart(chart_df)

st.success(f"📍 Selected Student Avg Marks: {round(student['avg'],2)}")

# -------------------------
# 🔮 PREDICTION
# -------------------------
st.subheader("🔮 Predicted End Semester Marks")

att_scaled = student['attendance'] / 100 * 30

prediction = (
    0.2 * att_scaled +
    0.25 * student['sessional1'] +
    0.25 * student['sessional2'] +
    0.30 * student['sessional3']
)

prediction = max(0, min(prediction, 100))

st.metric("Predicted Marks", round(prediction, 2))

# -------------------------
# 🏆 RANKING SYSTEM
# -------------------------
st.subheader("🏆 Top 5 Students")

df['predicted'] = (
    0.2 * (df['attendance']/100*30) +
    0.25 * df['sessional1'] +
    0.25 * df['sessional2'] +
    0.30 * df['sessional3']
)

top_students = df.sort_values(by='predicted', ascending=False).head(5)

st.table(top_students[['name','roll','predicted']])
