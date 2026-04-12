import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
        numeric_columns = ["roll", "attendance", "sessional1", "sessional2", "sessional3"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
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



def save_entry(df, row_data, replace_existing=False):
    working_df = df.copy()
    roll_value = int(row_data["roll"])
    existing_mask = working_df["roll"].fillna(-1).astype(int) == roll_value

    if replace_existing and existing_mask.any():
        working_df = working_df.loc[~existing_mask].copy()

    updated_df = pd.concat([working_df, pd.DataFrame([row_data])], ignore_index=True)
    updated_df = updated_df.sort_values("roll").reset_index(drop=True)
    updated_df.to_csv(CSV_PATH, index=False)
    return updated_df



def save_prediction_to_existing_record(df, roll, prediction_data):
    updated_df = df.copy()
    roll_mask = updated_df["roll"].fillna(-1).astype(int) == int(roll)

    if not roll_mask.any():
        raise ValueError("The selected roll number was not found in the dataset.")

    for column in OUTPUT_COLUMNS:
        updated_df.loc[roll_mask, column] = prediction_data[column]

    updated_df = updated_df.sort_values("roll").reset_index(drop=True)
    updated_df.to_csv(CSV_PATH, index=False)
    return updated_df


if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "pending_row" not in st.session_state:
    st.session_state.pending_row = None
if "duplicate_roll_found" not in st.session_state:
    st.session_state.duplicate_roll_found = False
if "existing_row_preview" not in st.session_state:
    st.session_state.existing_row_preview = None
if "searched_student_result" not in st.session_state:
    st.session_state.searched_student_result = None

st.title("Student Performance Dashboard")
st.caption("Add student records, generate predictions, and save records directly to the dataset.")

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

    default_roll = int(df["roll"].max()) + 1 if not df.empty and pd.notna(df["roll"].max()) else 1

    with st.form("student_entry_form", clear_on_submit=False):
        name = st.text_input("Student Name", placeholder="Enter full name")
        roll = st.number_input("Roll Number", min_value=1, step=1, value=default_roll)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
        s1 = st.number_input("Sessional 1 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        s2 = st.number_input("Sessional 2 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        s3 = st.number_input("Sessional 3 (out of 30)", min_value=0.0, max_value=30.0, step=1.0)
        predict_clicked = st.form_submit_button("Generate Prediction")

    if predict_clicked:
        clean_name = name.strip()

        if not clean_name:
            st.warning("Please enter a student name.")
        else:
            result_row = build_result_row(model, clean_name, roll, attendance, s1, s2, s3)
            st.session_state.last_prediction = result_row
            st.session_state.pending_row = result_row

            duplicate_mask = df["roll"].fillna(-1).astype(int) == int(roll)
            st.session_state.duplicate_roll_found = bool(duplicate_mask.any())
            st.session_state.existing_row_preview = (
                df.loc[duplicate_mask].iloc[0].to_dict() if duplicate_mask.any() else None
            )

            if st.session_state.duplicate_roll_found:
                st.warning(
                    "This roll number already exists. You can replace the existing record with the new values. If you only want to generate and save prediction for an existing record, use the search section below."
                )
            else:
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

        replace_existing = False
        if st.session_state.duplicate_roll_found:
            st.warning("A record with this roll number is already present in the dataset.")
            if st.session_state.existing_row_preview is not None:
                existing = st.session_state.existing_row_preview
                st.write("**Existing Record**")
                st.write(f"Name: {existing.get('name', '')}")
                st.write(f"Attendance: {existing.get('attendance', '')}%")
                st.write(
                    f"Sessional Scores: {existing.get('sessional1', '')}, {existing.get('sessional2', '')}, {existing.get('sessional3', '')}"
                )
            replace_existing = st.checkbox(
                "Roll number already exists. Replace the existing record with this new entry?",
                value=False,
                key="replace_existing_checkbox",
            )

        button_label = "Replace Existing Entry" if st.session_state.duplicate_roll_found else "Save Entry to Dataset"

        if st.button(button_label):
            if st.session_state.duplicate_roll_found and not replace_existing:
                st.info("Replacement was not selected. Existing record has not been changed.")
            else:
                try:
                    df = save_entry(
                        df,
                        st.session_state.pending_row,
                        replace_existing=st.session_state.duplicate_roll_found,
                    )
                    st.session_state.pending_row = None
                    st.session_state.last_prediction = None
                    st.session_state.duplicate_roll_found = False
                    st.session_state.existing_row_preview = None
                    st.cache_data.clear()
                    train_model_cached.clear()
                    st.success("Dataset updated successfully.")
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
    student_df = df[df["roll"].fillna(-1).astype(int) == int(search_roll)]
    if student_df.empty:
        st.session_state.searched_student_result = None
        st.warning("No student record found for this roll number.")
    else:
        student = student_df.iloc[0]
        searched_result = build_result_row(
            model,
            str(student["name"]),
            int(student["roll"]),
            float(student["attendance"]),
            float(student["sessional1"]),
            float(student["sessional2"]),
            float(student["sessional3"]),
        )
        st.session_state.searched_student_result = searched_result

if st.session_state.searched_student_result is not None:
    student_result = st.session_state.searched_student_result

    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("Predicted EndSem", student_result["predicted_endsem"])
    detail_col2.metric("Status", student_result["status"])
    detail_col3.metric("Grade", student_result["grade"])

    st.write(f"**Name:** {student_result['name']}")
    st.write(f"**Roll Number:** {student_result['roll']}")
    st.write(f"**Attendance:** {student_result['attendance']}%")
    st.write(
        f"**Sessional Scores:** {student_result['sessional1']}, {student_result['sessional2']}, {student_result['sessional3']}"
    )
    st.write(f"**Average Sessional:** {student_result['average_sessional']}")
    st.write(f"**Attendance Status:** {student_result['attendance_status']}")
    st.write(f"**Trend:** {student_result['trend']}")

    existing_student_row = df[df["roll"].fillna(-1).astype(int) == int(student_result["roll"])].iloc[0]
    prediction_already_saved = pd.notna(existing_student_row["predicted_endsem"])
    prediction_button_label = (
        "Update Saved Prediction" if prediction_already_saved else "Save Prediction to Existing Record"
    )

    if st.button(prediction_button_label):
        try:
            df = save_prediction_to_existing_record(df, student_result["roll"], student_result)
            st.cache_data.clear()
            train_model_cached.clear()
            st.success("Prediction saved to the existing record successfully.")
            st.rerun()
        except Exception as error:
            st.error(f"Prediction save failed: {error}")

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
