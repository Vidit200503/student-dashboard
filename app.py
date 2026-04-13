import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

CSV_FILE = "final_student_200.csv"


# -----------------------------
# Data handling
# -----------------------------
def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=[
            "name",
            "roll",
            "attendance",
            "sessional1",
            "sessional2",
            "sessional3",
            "predicted_endsem",
            "status",
            "grade",
            "trend",
            "attendance_status",
            "average_sessional",
            "record_action",
            "last_updated",
            "source_mode"
        ])

    df.columns = df.columns.str.lower().str.strip()

    required_columns = [
        "name",
        "roll",
        "attendance",
        "sessional1",
        "sessional2",
        "sessional3",
        "predicted_endsem",
        "status",
        "grade",
        "trend",
        "attendance_status",
        "average_sessional",
        "record_action",
        "last_updated",
        "source_mode"
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    numeric_cols = [
        "roll",
        "attendance",
        "sessional1",
        "sessional2",
        "sessional3",
        "predicted_endsem",
        "average_sessional"
    ]

    text_cols = [
        "name",
        "status",
        "grade",
        "trend",
        "attendance_status",
        "record_action",
        "last_updated",
        "source_mode"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in text_cols:
        df[col] = df[col].astype("object")

    return df


def save_data(df):
    save_df = df.copy()

    numeric_cols = [
        "roll",
        "attendance",
        "sessional1",
        "sessional2",
        "sessional3",
        "predicted_endsem",
        "average_sessional"
    ]

    for col in numeric_cols:
        if col in save_df.columns:
            save_df[col] = pd.to_numeric(save_df[col], errors="coerce")

    save_df.to_csv(CSV_FILE, index=False)


# -----------------------------
# Model preparation
# -----------------------------
def prepare_training_data(df):
    train_df = df.copy()

    numeric_cols = ["attendance", "sessional1", "sessional2", "sessional3"]
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)

    if "endsem" not in train_df.columns:
        train_df["endsem"] = (
            0.2 * train_df["attendance"] +
            0.25 * (train_df["sessional1"] / 30) * 100 +
            0.25 * (train_df["sessional2"] / 30) * 100 +
            0.30 * (train_df["sessional3"] / 30) * 100
        )
        np.random.seed(42)
        train_df["endsem"] += np.random.randint(-5, 6, size=len(train_df))
        train_df["endsem"] = train_df["endsem"].clip(0, 100)

    return train_df


def train_model(df):
    train_df = prepare_training_data(df)

    X = train_df[["attendance", "sessional1", "sessional2", "sessional3"]]
    y = train_df["endsem"]

    model = LinearRegression()
    model.fit(X, y)
    return model


# -----------------------------
# Prediction helpers
# -----------------------------
def get_grade(prediction):
    if prediction >= 90:
        return "A+"
    elif prediction >= 80:
        return "A"
    elif prediction >= 70:
        return "B"
    elif prediction >= 60:
        return "C"
    elif prediction >= 50:
        return "D"
    elif prediction >= 40:
        return "E"
    return "F"


def get_status(prediction):
    return "PASS" if prediction >= 40 else "FAIL"


def get_attendance_status(attendance):
    if attendance < 65:
        return "Poor"
    elif attendance <= 75:
        return "Good"
    return "Best"


def get_trend(s1, s2, s3):
    if s1 < s2 < s3:
        return "Improving"
    elif s1 > s2 > s3:
        return "Declining"
    return "Stable"


def predict_result(model, attendance, s1, s2, s3):
    input_df = pd.DataFrame([{
        "attendance": float(attendance),
        "sessional1": float(s1),
        "sessional2": float(s2),
        "sessional3": float(s3)
    }])

    prediction = float(model.predict(input_df)[0])
    prediction = max(0, min(100, prediction))

    average_sessional = round((float(s1) + float(s2) + float(s3)) / 3, 2)

    return {
        "predicted_endsem": round(prediction, 2),
        "status": get_status(prediction),
        "grade": get_grade(prediction),
        "trend": get_trend(float(s1), float(s2), float(s3)),
        "attendance_status": get_attendance_status(float(attendance)),
        "average_sessional": average_sessional
    }


def build_student_record(name, roll, attendance, s1, s2, s3, result, action, source_mode):
    return {
        "name": str(name).strip(),
        "roll": int(roll),
        "attendance": float(attendance),
        "sessional1": float(s1),
        "sessional2": float(s2),
        "sessional3": float(s3),
        "predicted_endsem": float(result["predicted_endsem"]),
        "status": str(result["status"]),
        "grade": str(result["grade"]),
        "trend": str(result["trend"]),
        "attendance_status": str(result["attendance_status"]),
        "average_sessional": float(result["average_sessional"]),
        "record_action": str(action),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_mode": str(source_mode)
    }


# -----------------------------
# Download helpers
# -----------------------------
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Updated_Student_Data")
    return output.getvalue()


# -----------------------------
# UI helpers
# -----------------------------
def show_prediction_section(result):
    st.markdown("### Model Prediction")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted EndSem", result["predicted_endsem"])
    c2.metric("Status", result["status"])
    c3.metric("Grade", result["grade"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Average Sessional", result["average_sessional"])
    c5.metric("Attendance Status", result["attendance_status"])
    c6.metric("Trend", result["trend"])

    st.info(
        f"Model predicts {result['predicted_endsem']} marks in EndSem with "
        f"status {result['status']}, grade {result['grade']}, and trend {result['trend']}."
    )


# -----------------------------
# App start
# -----------------------------
df = load_data()
model = train_model(df)

st.title("Student Performance Prediction Dashboard")
st.caption("Add, replace, update, view, predict, and download student records directly from the dashboard.")

tab1, tab2, tab3 = st.tabs([
    "Add / Update Student",
    "Search Existing Student",
    "View Updated Dataset"
])

# -----------------------------
# Tab 1: Add / Update Student
# -----------------------------
with tab1:
    st.subheader("Add New Student or Replace Existing Record")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Student Name")
        roll = st.number_input("Roll Number", min_value=1, step=1)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=0.1)

    with col2:
        s1 = st.number_input("Sessional 1", min_value=0.0, max_value=30.0, step=0.1)
        s2 = st.number_input("Sessional 2", min_value=0.0, max_value=30.0, step=0.1)
        s3 = st.number_input("Sessional 3", min_value=0.0, max_value=30.0, step=0.1)

    if st.button("Generate Prediction", use_container_width=True):
        if not name.strip():
            st.error("Please enter student name.")
        else:
            result = predict_result(model, attendance, s1, s2, s3)
            st.session_state["new_prediction"] = {
                "name": name.strip(),
                "roll": int(roll),
                "attendance": float(attendance),
                "sessional1": float(s1),
                "sessional2": float(s2),
                "sessional3": float(s3),
                **result
            }

    if "new_prediction" in st.session_state:
        student = st.session_state["new_prediction"]

        result = {
            "predicted_endsem": student["predicted_endsem"],
            "status": student["status"],
            "grade": student["grade"],
            "trend": student["trend"],
            "attendance_status": student["attendance_status"],
            "average_sessional": student["average_sessional"]
        }

        show_prediction_section(result)

        df["roll"] = pd.to_numeric(df["roll"], errors="coerce")
        existing = df[df["roll"] == int(student["roll"])]

        if not existing.empty:
            st.warning("This roll number already exists.")
            st.write("### Existing Record")
            st.dataframe(existing, use_container_width=True)

            replace_existing = st.checkbox(
                "Replace the existing record with this new entry",
                key="replace_existing_checkbox"
            )

            if st.button("Save / Replace Record", use_container_width=True):
                if replace_existing:
                    updated_student = build_student_record(
                        student["name"],
                        student["roll"],
                        student["attendance"],
                        student["sessional1"],
                        student["sessional2"],
                        student["sessional3"],
                        result,
                        "Replaced Existing Record",
                        "Manual Entry"
                    )

                    existing_index = existing.index[0]

                    text_cols = [
                        "name",
                        "status",
                        "grade",
                        "trend",
                        "attendance_status",
                        "record_action",
                        "last_updated",
                        "source_mode"
                    ]
                    for col in text_cols:
                        if col in df.columns:
                            df[col] = df[col].astype("object")

                    for col, value in updated_student.items():
                        df.at[existing_index, col] = value

                    save_data(df)

                    st.success("Existing record replaced successfully at the same position.")
                    st.session_state.pop("new_prediction", None)
                    st.rerun()
                else:
                    st.info("Tick the checkbox to replace the existing record.")
        else:
            if st.button("Save New Record", use_container_width=True):
                new_student = build_student_record(
                    student["name"],
                    student["roll"],
                    student["attendance"],
                    student["sessional1"],
                    student["sessional2"],
                    student["sessional3"],
                    result,
                    "New Entry",
                    "Manual Entry"
                )

                df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
                save_data(df)

                st.success("New record saved successfully.")
                st.session_state.pop("new_prediction", None)
                st.rerun()

# -----------------------------
# Tab 2: Search Existing Student
# -----------------------------
with tab2:
    st.subheader("Search Existing Student and View Model Prediction")

    search_roll = st.number_input(
        "Enter Roll Number",
        min_value=1,
        step=1,
        key="search_roll"
    )

    if st.button("Search Student", use_container_width=True):
        st.session_state["search_triggered"] = True

    if st.session_state.get("search_triggered", False):
        df["roll"] = pd.to_numeric(df["roll"], errors="coerce")
        found = df[df["roll"] == int(search_roll)]

        if found.empty:
            st.error("No student found for this roll number.")
        else:
            st.write("### Existing Student Record")
            st.dataframe(found, use_container_width=True)

            row = found.iloc[0]
            result = predict_result(
                model,
                float(row["attendance"]),
                float(row["sessional1"]),
                float(row["sessional2"]),
                float(row["sessional3"])
            )

            show_prediction_section(result)

            if st.button("Save Prediction to Existing Record", use_container_width=True):
                idx = found.index[0]

                text_cols = [
                    "name",
                    "status",
                    "grade",
                    "trend",
                    "attendance_status",
                    "record_action",
                    "last_updated",
                    "source_mode"
                ]
                for col in text_cols:
                    if col in df.columns:
                        df[col] = df[col].astype("object")

                df.at[idx, "predicted_endsem"] = result["predicted_endsem"]
                df.at[idx, "status"] = result["status"]
                df.at[idx, "grade"] = result["grade"]
                df.at[idx, "trend"] = result["trend"]
                df.at[idx, "attendance_status"] = result["attendance_status"]
                df.at[idx, "average_sessional"] = result["average_sessional"]
                df.at[idx, "record_action"] = "Prediction Updated"
                df.at[idx, "last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.at[idx, "source_mode"] = "Search Existing Student"

                save_data(df)
                st.success("Prediction saved to the existing record successfully.")
                st.session_state["search_triggered"] = False
                st.rerun()

# -----------------------------
# Tab 3: View Updated Dataset
# -----------------------------
with tab3:
    st.subheader("Updated Student Dataset")

    df_display = df.copy()

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        if "record_action" in df_display.columns:
            action_values = [x for x in df_display["record_action"].dropna().astype(str).unique()]
            action_options = ["All"] + sorted(action_values)
            selected_action = st.selectbox("Filter by Record Action", action_options)
            if selected_action != "All":
                df_display = df_display[df_display["record_action"] == selected_action]

    with filter_col2:
        search_name = st.text_input("Filter by Name")
        if search_name.strip():
            df_display = df_display[
                df_display["name"].astype(str).str.lower().str.contains(search_name.strip().lower(), na=False)
            ]

    st.write(f"**Total Records:** {len(df_display)}")
    st.dataframe(df_display, use_container_width=True)

    if "predicted_endsem" in df_display.columns:
        st.markdown("### Prediction Overview")
        p1, p2, p3 = st.columns(3)
        p1.metric("Students with Prediction", int(df_display["predicted_endsem"].notna().sum()))
        p2.metric("Pass Predictions", int((df_display["status"] == "PASS").sum()) if "status" in df_display.columns else 0)
        p3.metric("Fail Predictions", int((df_display["status"] == "FAIL").sum()) if "status" in df_display.columns else 0)

    csv_data = convert_df_to_csv(df_display)
    excel_data = convert_df_to_excel(df_display)

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Download Updated Dataset as CSV",
            data=csv_data,
            file_name="updated_final_student_200.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:
        st.download_button(
            label="Download Updated Dataset as Excel",
            data=excel_data,
            file_name="updated_final_student_200.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
