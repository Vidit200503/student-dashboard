import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Dashboard", layout="wide")

st.title("🎓 Student Performance Dashboard")

# -------------------------
# 📂 LOAD DATA (FINAL FILE NAME)
# -------------------------
df = pd.read_csv("final_student_200.csv")

# -------------------------
# 🧹 DATA CLEANING
# -------------------------
for col in ['sessional1', 'sessional2', 'sessional3']:
    df[col] = df[col].clip(0, 30)

df['attendance'] = df['attendance'].clip(0, 100)

# Average
df['avg'] = df[['sessional1','sessional2','sessional3']].mean(axis=1)

# -------------------------
# 📋 DATASET (TOP SECTION)
# -------------------------
st.subheader("📋 Student Dataset")
st.dataframe(df)

# -------------------------
# 🔍 SELECT STUDENT
# -------------------------
st.subheader("🔍 Select Student")

student_name = st.selectbox("Choose Student", df['name'])

student = df[df['name'] == student_name].iloc[0]

# -------------------------
# 📊 STUDENT INFO
# -------------------------
st.subheader("📊 Student Details")

col1, col2, col3 = st.columns(3)

col1.metric("Roll No", student['roll'])
col2.metric("Attendance (%)", student['attendance'])
col3.metric("Average Marks", round(student['avg'], 2))

# -------------------------
# 📊 INDIVIDUAL GRAPH
# -------------------------
st.subheader("📊 Sessional Performance")

marks_df = pd.DataFrame({
    "Sessional": ["S1", "S2", "S3"],
    "Marks": [student['sessional1'], student['sessional2'], student['sessional3']]
})

st.bar_chart(marks_df.set_index("Sessional"))

# -------------------------
# 📈 TREND
# -------------------------
st.subheader("📈 Performance Trend")
st.line_chart(marks_df.set_index("Sessional"))

# -------------------------
# 🔮 CORRECT PREDICTION (OUT OF 100)
# -------------------------
st.subheader("🔮 Predicted End Semester Marks")

# scale sessionals to 100
s1_scaled = (student['sessional1'] / 30) * 100
s2_scaled = (student['sessional2'] / 30) * 100
s3_scaled = (student['sessional3'] / 30) * 100

prediction = (
    0.2 * student['attendance'] +
    0.25 * s1_scaled +
    0.25 * s2_scaled +
    0.30 * s3_scaled
)

prediction = max(0, min(prediction, 100))

st.metric("Predicted Marks (out of 100)", round(prediction, 2))

# -------------------------
# 🏆 TOP STUDENTS
# -------------------------
st.subheader("🏆 Top Performers")

df['predicted'] = (
    0.2 * df['attendance'] +
    0.25 * (df['sessional1']/30*100) +
    0.25 * (df['sessional2']/30*100) +
    0.30 * (df['sessional3']/30*100)
)

top_students = df.sort_values(by='predicted', ascending=False).head(5)

st.table(top_students[['name','roll','predicted']])
