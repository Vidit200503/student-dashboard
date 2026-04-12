import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

CSV_PATH = "final_student_200.csv"
BASE_COLUMNS = ["name", "roll", "attendance", "sessional1", "sessional2", "sessional3"]
OUTPUT_COLUMNS = [
    "predicted_endsem",
    "status",
    "grade",
    "trend",
    "attendance_status",
    "average_sessional",
]
ALL_COLUMNS = BASE_COLUMNS + OUTPUT_COLUMNS


def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=ALL_COLUMNS)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower().str.strip()

    for column in ALL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df = df[ALL_COLUMNS].copy()
    if not df.empty:
        df["roll"] = pd.to_numeric(df["roll"], errors="coerce")
        df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
        df["sessional1"] = pd.to_numeric(df["sessional1"], errors="coerce")
        df["sessional2"] = pd.to_numeric(df["sessional2"], errors="coerce")
        df["sessional3"] = pd.to_numeric(df["sessional3"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def prepare_training_data(df):
    work_df = df[BASE_COLUMNS].dropna().copy()
    if work_df.empty:
        return work_df

    work_df["endsem"] = (
        0.20 * work_df["attendance"]
        + 0.25 * (work_df["sessional1"] / 30.0) * 100.0
        + 0.25 * (work_df["sessional2"] / 30.0) * 100.0
        + 0.30 * (work_df["sessional3"] / 30.0) * 100.0
    )

    np.random.seed(42)
    work_df["endsem"] += np.random.randint(-5, 6, size=len(work_df))
    work_df["endsem"] = work_df["endsem"].clip(0, 100)
    return work_df


@st.cache_resource(show_spinner=False)
def train_model_cached(data_signature):
    df = load_data()
    train_df = prepare_training_data(df)

    model = LinearRegression()
    accuracy = None

    X = train_df[["attendance", "sessional1", "sessional2", "sessional3"]]
    y = train_df["endsem"]

    if len(train_df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_test_class = ["PASS" if value >= 40 else "FAIL" for value in y_test]
        y_pred_class = ["PASS" if value >= 40 else "FAIL" for value in y_pred]
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
    return round(max(0.0, min(100.0, prediction)), 2)


def build_result_row(model, name, roll, attendance, s1, s2, s3):
    predicted_endsem = predict_student(model, attendance, s1, s2, s3)
    average_sessional = round((s1 + s2 + s3) / 3.0, 2)
    status = "PASS" if predicted_endsem >= 40 else "FAIL"
    grade = get_grade(predicted_endsem)
    trend = get_trend(s1, s2, s3)
    attendance_status = get_attendance_status(attendance)

    return {
        "name": name.strip(),
        "roll": int(roll),
        "attendance": float(attendance),
        "sessional1": float(s1),
        "sessional2": float(s2),
        "sessional3": float(s3),
        "predicted_endsem": predicted_endsem,
        "status": status,
        "grade": grade,
        "trend": trend,
        "attendance_status": attendance_status,
        "average_sessional": average_sessional,
    }


def save_entry(df, row_data):
    updated_df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    updated_df = updated_df.sort_values("roll").reset_index(drop=True)
    updated_df.to_csv(CSV_PATH, index=False)
    return updated_df


if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "pending_row" not in st.session_state:
    st.session_state.pending_row = None

st.title("Student Performance Dashboard")
st.caption("Add student records, generate predictions, and save new entries directly to the dataset.")

try:
    df = load_data()
    signature = (
        len(df),
        str(df[BASE_COLUMNS].fillna("").astype(str).agg("|".join, axis=1).sum()) if not df.empty else "empty",
    )
    model, accuracy = train_model_cached(signature)
except Exception as error:
    st.error(f"Application error: {error}")
    st.stop()

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Records", int(len(df)))
metric_col2.metric("Current Highest Roll", int(df["roll"].max()) if not df.empty else 0)
metric_col3.metric("Model Accuracy", f"{accuracy:.2f}%" if accuracy is not None else "N/A")

st.divider()
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Add New Student")

    with st.form("student_entry_form", clear_on_submit=False):
        name = st.text_input("Student Name", placeholder="Enter full name")
        roll = st.number_input("Roll Number", min_value=1, step=1, value=201)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
        s1 = st.number_input("Sessional 1 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        s2 = st.number_input("Sessional 2 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        s3 = st.number_input("Sessional 3 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        predict_clicked = st.form_submit_button("Generate Prediction")

    if predict_clicked:
        clean_name = name.strip()
        existing_rolls = set(df["roll"].dropna().astype(int).tolist())

        if not clean_name:
            st.warning("Please enter a student name.")
        elif int(roll) in existing_rolls:
            st.warning("This roll number already exists. Please enter a unique roll number.")
        else:
            result_row = build_result_row(model, clean_name, roll, attendance, s1, s2, s3)
            st.session_state.last_prediction = result_row
            st.session_state.pending_row = result_row
            st.success("Prediction generated successfully.")

with right_col:
    st.subheader("Prediction Result")

    if st.session_state.last_prediction is None:
        st.info("Enter student details and click Generate Prediction to view the result here.")
    else:
        result = st.session_state.last_prediction
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Predicted EndSem", result["predicted_endsem"])
        summary_col2.metric("Status", result["status"])
        summary_col3.metric("Grade", result["grade"])

        st.write(f"**Name:** {result['name']}")
        st.write(f"**Roll Number:** {result['roll']}")
        st.write(f"**Attendance:** {result['attendance']}%")
        st.write(f"**Sessional Scores:** {result['sessional1']}, {result['sessional2']}, {result['sessional3']}")
        st.write(f"**Average Sessional:** {result['average_sessional']}")
        st.write(f"**Attendance Status:** {result['attendance_status']}")
        st.write(f"**Trend:** {result['trend']}")

        if st.button("Save Entry to Dataset"):
            try:
                df = save_entry(df, st.session_state.pending_row)
                st.session_state.pending_row = None
                st.cache_data.clear()
                train_model_cached.clear()
                st.success("Entry saved successfully.")
                st.rerun()
            except Exception as error:
                st.error(f"Save failed: {error}")

st.divider()
st.subheader("Search Existing Student")
search_col1, search_col2 = st.columns([1, 3])
with search_col1:
    search_roll = st.number_input("Roll Number", min_value=1, step=1, key="search_roll")
with search_col2:
    search_clicked = st.button("Search Record")

if search_clicked:
    student = df[df["roll"].fillna(-1).astype(int) == int(search_roll)]
    if student.empty:
        st.warning("No student record found for this roll number.")
    else:
        student = student.iloc[0]
        if pd.isna(student["predicted_endsem"]):
            predicted_endsem = predict_student(
                model,
                float(student["attendance"]),
                float(student["sessional1"]),
                float(student["sessional2"]),
                float(student["sessional3"]),
            )
            status = "PASS" if predicted_endsem >= 40 else "FAIL"
            grade = get_grade(predicted_endsem)
            trend = get_trend(float(student["sessional1"]), float(student["sessional2"]), float(student["sessional3"]))
            attendance_status = get_attendance_status(float(student["attendance"]))
            average_sessional = round((float(student["sessional1"]) + float(student["sessional2"]) + float(student["sessional3"])) / 3.0, 2)
        else:
            predicted_endsem = student["predicted_endsem"]
            status = student["status"]
            grade = student["grade"]
            trend = student["trend"]
            attendance_status = student["attendance_status"]
            average_sessional = student["average_sessional"]

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        detail_col1.metric("Predicted EndSem", predicted_endsem)
        detail_col2.metric("Status", status)
        detail_col3.metric("Grade", grade)

        st.write(f"**Name:** {student['name']}")
        st.write(f"**Attendance:** {student['attendance']}%")
        st.write(f"**Sessional Scores:** {student['sessional1']}, {student['sessional2']}, {student['sessional3']}")
        st.write(f"**Average Sessional:** {average_sessional}")
        st.write(f"**Attendance Status:** {attendance_status}")
        st.write(f"**Trend:** {trend}")

st.divider()
st.subheader("Dataset Records")

view_col1, view_col2 = st.columns([1, 1])
with view_col1:
    show_full_dataset = st.checkbox("Show complete dataset", value=True)
with view_col2:
    name_filter = st.text_input("Filter by name", placeholder="Type a name to filter records")

display_df = df.copy()
if name_filter.strip():
    display_df = display_df[display_df["name"].astype(str).str.contains(name_filter.strip(), case=False, na=False)]

st.write(f"Showing {len(display_df)} record(s) out of {len(df)} total record(s).")

if show_full_dataset:
    st.dataframe(display_df, use_container_width=True, height=600)
else:
    st.dataframe(display_df.head(25), use_container_width=True, height=600)

csv_download = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Visible Records as CSV",
    data=csv_download,
    file_name="student_records_view.csv",
    mime="text/csv",
)
