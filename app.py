import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

CSV_PATH = "final_student_200.csv"

# ---------- Helpers ----------
def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=["name", "roll", "attendance", "sessional1", "sessional2", "sessional3"])

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower().str.strip()

    required_cols = ["name", "roll", "attendance", "sessional1", "sessional2", "sessional3"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required_cols].copy()
    return df


def prepare_training_data(df):
    work_df = df.copy()
    work_df["endsem"] = (
        0.20 * work_df["attendance"]
        + 0.25 * (work_df["sessional1"] / 30) * 100
        + 0.25 * (work_df["sessional2"] / 30) * 100
        + 0.30 * (work_df["sessional3"] / 30) * 100
    )

    np.random.seed(42)
    work_df["endsem"] += np.random.randint(-5, 6, size=len(work_df))
    work_df["endsem"] = work_df["endsem"].clip(0, 100)
    return work_df


def train_model(df):
    train_df = prepare_training_data(df)
    X = train_df[["attendance", "sessional1", "sessional2", "sessional3"]]
    y = train_df["endsem"]

    model = LinearRegression()
    accuracy = None

    if len(train_df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_test_class = ["PASS" if val >= 40 else "FAIL" for val in y_test]
        y_pred_class = ["PASS" if val >= 40 else "FAIL" for val in y_pred]
        accuracy = accuracy_score(y_test_class, y_pred_class) * 100
    else:
        model.fit(X, y)

    return model, accuracy


def get_grade(prediction):
    if prediction >= 90:
        return "A+"
    if prediction >= 80:
        return "A"
    if prediction >= 70:
        return "B"
    if prediction >= 60:
        return "C"
    if prediction >= 50:
        return "D"
    if prediction >= 40:
        return "E"
    return "F"


def get_attendance_status(attendance):
    if attendance < 65:
        return "Poor"
    if attendance <= 75:
        return "Good"
    return "Excellent"


def get_trend(s1, s2, s3):
    if s1 < s2 < s3:
        return "Improving"
    if s1 > s2 > s3:
        return "Declining"
    return "Stable"


def predict_student(model, attendance, s1, s2, s3):
    features = pd.DataFrame(
        [[attendance, s1, s2, s3]],
        columns=["attendance", "sessional1", "sessional2", "sessional3"],
    )
    prediction = float(model.predict(features)[0])
    return max(0.0, min(100.0, prediction))


def save_new_entry(df, new_row):
    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    updated_df.to_csv(CSV_PATH, index=False)
    return updated_df


# ---------- UI ----------
st.title("🎓 Student Performance Dashboard")
st.caption("Manual entry add karo, prediction lo, aur same CSV/database me save karo.")

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "pending_row" not in st.session_state:
    st.session_state.pending_row = None

try:
    df = load_data()
    model, accuracy = train_model(df)
except Exception as e:
    st.error(f"Data load/train error: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df))
c2.metric("Highest Roll", int(df["roll"].max()) if not df.empty else 0)
c3.metric("Model Accuracy (PASS/FAIL)", f"{accuracy:.2f}%" if accuracy is not None else "N/A")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("➕ Add New Student Entry")

    with st.form("student_form", clear_on_submit=False):
        name = st.text_input("Student Name")
        roll = st.number_input("Roll Number", min_value=1, step=1)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
        s1 = st.number_input("Sessional 1 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        s2 = st.number_input("Sessional 2 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        s3 = st.number_input("Sessional 3 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)

        predict_btn = st.form_submit_button("Predict Result")

    if predict_btn:
        clean_name = name.strip()

        if not clean_name:
            st.warning("Please enter student name.")
        elif int(roll) in df["roll"].astype(int).tolist():
            st.warning("This roll number already exists. Use a new roll number or add edit feature.")
        else:
            pred = predict_student(model, attendance, s1, s2, s3)
            avg = round((s1 + s2 + s3) / 3, 2)
            result = {
                "name": clean_name,
                "roll": int(roll),
                "attendance": float(attendance),
                "sessional1": float(s1),
                "sessional2": float(s2),
                "sessional3": float(s3),
                "predicted_endsem": round(pred, 2),
                "status": "PASS" if pred >= 40 else "FAIL",
                "grade": get_grade(pred),
                "trend": get_trend(s1, s2, s3),
                "attendance_status": get_attendance_status(attendance),
                "average_sessional": avg,
            }
            st.session_state.last_prediction = result
            st.session_state.pending_row = {
                "name": clean_name,
                "roll": int(roll),
                "attendance": float(attendance),
                "sessional1": float(s1),
                "sessional2": float(s2),
                "sessional3": float(s3),
            }

with right:
    st.subheader("📊 Prediction Output")

    if st.session_state.last_prediction:
        result = st.session_state.last_prediction

        r1, r2 = st.columns(2)
        r1.metric("Predicted EndSem", result["predicted_endsem"])
        r2.metric("Status", result["status"])

        st.write(f"**Name:** {result['name']}")
        st.write(f"**Roll:** {result['roll']}")
        st.write(f"**Average Sessional:** {result['average_sessional']}")
        st.write(f"**Attendance Status:** {result['attendance_status']}")
        st.write(f"**Grade:** {result['grade']}")
        st.write(f"**Trend:** {result['trend']}")

        if st.button("💾 Save This Entry to CSV"):
            try:
                updated_df = save_new_entry(df, st.session_state.pending_row)
                st.success("New student entry saved successfully in final_student_200.csv")
                st.session_state.pending_row = None
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Save failed: {e}")
    else:
        st.info("Form fill karke Predict Result dabao. Yahin prediction show hoga aur save option milega.")

st.divider()

st.subheader("🔍 Search Existing Student")
search_roll = st.number_input("Enter roll number to search", min_value=1, step=1, key="search_roll")

if st.button("Search Student"):
    student = df[df["roll"].astype(int) == int(search_roll)]
    if student.empty:
        st.warning("Student not found.")
    else:
        student = student.iloc[0]
        pred = predict_student(
            model,
            float(student["attendance"]),
            float(student["sessional1"]),
            float(student["sessional2"]),
            float(student["sessional3"]),
        )
        avg = round((student["sessional1"] + student["sessional2"] + student["sessional3"]) / 3, 2)

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted EndSem", round(pred, 2))
        col2.metric("Status", "PASS" if pred >= 40 else "FAIL")
        col3.metric("Grade", get_grade(pred))

        st.write(f"**Name:** {student['name']}")
        st.write(f"**Attendance:** {student['attendance']}%")
        st.write(f"**Sessionals:** {student['sessional1']}, {student['sessional2']}, {student['sessional3']}")
        st.write(f"**Average:** {avg}")
        st.write(f"**Attendance Status:** {get_attendance_status(float(student['attendance']))}")
        st.write(f"**Trend:** {get_trend(float(student['sessional1']), float(student['sessional2']), float(student['sessional3']))}")

st.divider()
st.subheader("📁 Current Dataset Preview")
st.dataframe(df.tail(15), use_container_width=True)
