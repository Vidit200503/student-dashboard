import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

CSV_FILE = "final_student_200.csv"


def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=[
            "name", "roll", "attendance", "sessional1", "sessional2", "sessional3"
        ])

    df.columns = df.columns.str.lower().str.strip()
    return df


def save_data(df):
    df.to_csv(CSV_FILE, index=False)


def prepare_training_data(df):
    train_df = df.copy()

    required_cols = ["attendance", "sessional1", "sessional2", "sessional3"]
    for col in required_cols:
        if col not in train_df.columns:
            train_df[col] = 0

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
        "attendance": attendance,
        "sessional1": s1,
        "sessional2": s2,
        "sessional3": s3
    }])

    prediction = float(model.predict(input_df)[0])
    prediction = max(0, min(100, prediction))

    average_sessional = round((s1 + s2 + s3) / 3, 2)

    return {
        "predicted_endsem": round(prediction, 2),
        "status": get_status(prediction),
        "grade": get_grade(prediction),
        "trend": get_trend(s1, s2, s3),
        "attendance_status": get_attendance_status(attendance),
        "average_sessional": average_sessional
    }


def build_student_record(name, roll, attendance, s1, s2, s3, result, action, source_mode):
    return {
        "name": name.strip(),
        "roll": int(roll),
        "attendance": float(attendance),
        "sessional1": float(s1),
        "sessional2": float(s2),
        "sessional3": float(s3),
        "predicted_endsem": result["predicted_endsem"],
        "status": result["status"],
        "grade": result["grade"],
        "trend": result["trend"],
        "attendance_status": result["attendance_status"],
        "average_sessional": result["average_sessional"],
        "record_action": action,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_mode": source_mode
    }


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Updated_Student_Data")
    return output.getvalue()


df = load_data()
model = train_model(df)

st.title("Student Performance Prediction Dashboard")
st.caption("Add, replace, update, view, and download student records directly from the dashboard.")

tab1, tab2, tab3 = st.tabs([
    "Add / Update Student",
    "Search Existing Student",
    "View Updated Dataset"
])

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

        st.markdown("### Prediction Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted EndSem", student["predicted_endsem"])
        m2.metric("Status", student["status"])
        m3.metric("Grade", student["grade"])

        st.write(f"**Trend:** {student['trend']}")
        st.write(f"**Attendance Status:** {student['attendance_status']}")
        st.write(f"**Average Sessional:** {student['average_sessional']}")

        existing = df[df["roll"] == student["roll"]]

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
                        {
                            "predicted_endsem": student["predicted_endsem"],
                            "status": student["status"],
                            "grade": student["grade"],
                            "trend": student["trend"],
                            "attendance_status": student["attendance_status"],
                            "average_sessional": student["average_sessional"]
                        },
                        "Replaced Existing Record",
                        "Manual Entry"
                    )

                    df = df[df["roll"] != student["roll"]]
                    df = pd.concat([df, pd.DataFrame([updated_student])], ignore_index=True)
                    save_data(df)

                    st.success("Existing record replaced successfully.")
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
                    {
                        "predicted_endsem": student["predicted_endsem"],
                        "status": student["status"],
                        "grade": student["grade"],
                        "trend": student["trend"],
                        "attendance_status": student["attendance_status"],
                        "average_sessional": student["average_sessional"]
                    },
                    "New Entry",
                    "Manual Entry"
                )

                df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
                save_data(df)

                st.success("New record saved successfully.")
                st.session_state.pop("new_prediction", None)
                st.rerun()

with tab2:
    st.subheader("Search Existing Student and Save Updated Prediction")

    search_roll = st.number_input(
        "Enter Roll Number",
        min_value=1,
        step=1,
        key="search_roll"
    )

    if st.button("Search Student", use_container_width=True):
        st.session_state["search_triggered"] = True

    if st.session_state.get("search_triggered", False):
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

            st.write("### Predicted Result")
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted EndSem", result["predicted_endsem"])
            p2.metric("Status", result["status"])
            p3.metric("Grade", result["grade"])

            st.write(f"**Trend:** {result['trend']}")
            st.write(f"**Attendance Status:** {result['attendance_status']}")
            st.write(f"**Average Sessional:** {result['average_sessional']}")

            if st.button("Save Prediction to Existing Record", use_container_width=True):
                idx = found.index[0]

                df.loc[idx, "predicted_endsem"] = result["predicted_endsem"]
                df.loc[idx, "status"] = result["status"]
                df.loc[idx, "grade"] = result["grade"]
                df.loc[idx, "trend"] = result["trend"]
                df.loc[idx, "attendance_status"] = result["attendance_status"]
                df.loc[idx, "average_sessional"] = result["average_sessional"]
                df.loc[idx, "record_action"] = "Prediction Updated"
                df.loc[idx, "last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.loc[idx, "source_mode"] = "Search Existing Student"

                save_data(df)
                st.success("Prediction saved to the existing record successfully.")
                st.session_state["search_triggered"] = False
                st.rerun()

with tab3:
    st.subheader("Updated Student Dataset")

    df_display = df.copy()

    if "last_updated" in df_display.columns:
        df_display["last_updated_sort"] = pd.to_datetime(
            df_display["last_updated"],
            errors="coerce"
        )
        df_display = df_display.sort_values(
            by="last_updated_sort",
            ascending=False,
            na_position="last"
        ).drop(columns=["last_updated_sort"])

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        if "record_action" in df_display.columns:
            action_options = ["All"] + sorted(
                [x for x in df_display["record_action"].dropna().astype(str).unique()]
            )
            selected_action = st.selectbox("Filter by Record Action", action_options)
            if selected_action != "All":
                df_display = df_display[df_display["record_action"] == selected_action]

    with filter_col2:
        if "source_mode" in df_display.columns:
            source_options = ["All"] + sorted(
                [x for x in df_display["source_mode"].dropna().astype(str).unique()]
            )
            selected_source = st.selectbox("Filter by Source Mode", source_options)
            if selected_source != "All":
                df_display = df_display[df_display["source_mode"] == selected_source]

    st.write(f"**Total Records:** {len(df_display)}")
    st.dataframe(df_display, use_container_width=True)

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
