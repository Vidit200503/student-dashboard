import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("final_student_200.csv")

st.title("🎓 Student Performance Dashboard")

# Show data
st.subheader("Student Data")
st.dataframe(df)

# Metrics
st.subheader("Overview")
st.metric("Total Students", len(df))
st.metric("Avg Attendance", round(df['attendance'].mean(), 2))

avg_marks = df[['sessional1','sessional2','sessional3']].mean().mean()
st.metric("Avg Marks", round(avg_marks, 2))

# Bar Chart
st.subheader("Sessional Comparison")
st.bar_chart(df.set_index('name')[['sessional1','sessional2','sessional3']])

# Trend
st.subheader("Performance Trend")
st.line_chart(df[['sessional1','sessional2','sessional3']].mean())

# Scatter
st.subheader("Attendance vs Performance")
df['avg'] = df[['sessional1','sessional2','sessional3']].mean(axis=1)
st.scatter_chart(df[['attendance','avg']])

# Prediction
st.subheader("Predicted EndSem")

df['predicted'] = (
    0.2 * df['attendance'] +
    0.25 * df['sessional1'] +
    0.25 * df['sessional2'] +
    0.30 * df['sessional3']
)

st.dataframe(df[['name','roll','predicted']])