import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

CSV_FILE = "final_student_200.csv"


# -------------------------
# Data functions
# -------------------------
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

    avg = round((s1 + s2 + s3) / 3, 2)
    status = get_status(prediction)
    grade = get_grade(prediction)
    attendance_status = get_attendance_status(attendance)
    trend = get_trend(s1, s2, s3)

    return {
        "predicted_endsem": round(prediction, 2),
        "status": status,
        "grade": grade,
        "trend": trend,
        "attendance_status": attendance_status,
        "average_sessional": avg
    }


# -------------------------
# App
# -------------------------
st.title("Student Performance Prediction Dashboard")
st.caption("Manual entry, direct CSV update, and prediction saving in one dashboard.")

df = load_data()
model = train_model(df)

tab1, tab2, tab3 = st.tabs(["Add / Update Student", "Search Student", "Dataset"])

with tab1:
    st.subheader("Add New Student or Update Existing Student")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Student Name")
        roll = st.number_input("Roll Number", min_value=1, step=1)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=0.1)

    with col2:
        s1 = st.number_input("Sessional 1", min_value=0.0, max_value=30.0, step=0.1)
        s2 = st.number_input("Sessional 2", min_value=0.0, max_value=30.0, step=0.1)
        s3 = st.number_input("Sessional 3", min_value=0.0, max_value=30.0, step=0.1)

    if st.button("Generate Prediction"):
        if not name.strip():
            st.error("Please enter student name.")
        else:
            result = predict_result(model, attendance, s1, s2, s3)

            st.session_state["generated_student"] = {
                "name": name.strip(),
                "roll": int(roll),
                "attendance": attendance,
                "sessional1": s1,
                "sessional2": s2,
                "sessional3": s3,
                **result
            }

    if "generated_student" in st.session_state:
        student = st.session_state["generated_student"]

        st.markdown("### Prediction Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted EndSem", student["predicted_endsem"])
        c2.metric("Status", student["status"])
        c3.metric("Grade", student["grade"])

        st.write(f"**Trend:** {student['trend']}")
        st.write(f"**Attendance Status:** {student['attendance_status']}")
        st.write(f"**Average Sessional:** {student['average_sessional']}")

        existing = df[df["roll"] == student["roll"]]

        if not existing.empty:
            st.warning("This roll number already exists in the CSV file.")
            st.dataframe(existing, use_container_width=True)

            replace_existing = st.checkbox("Replace the existing record with this new entry")

            if st.button("Save to CSV"):
                if replace_existing:
                    df = df[df["roll"] != student["roll"]]

                    new_row = pd.DataFrame([student])
                    df = pd.concat([df, new_row], ignore_index=True)
                    save_data(df)

                    st.success("Existing record replaced and CSV updated successfully.")
                    st.rerun()
                else:
                    st.info("Tick the replace checkbox if you want to overwrite the old entry.")
        else:
            if st.button("Save to CSV"):
                new_row = pd.DataFrame([student])
                df = pd.concat([df, new_row], ignore_index=True)
                save_data(df)

                st.success("New student record saved directly to CSV.")
                st.rerun()

with tab2:
    st.subheader("Search Existing Student")

    search_roll = st.number_input("Enter Roll Number to Search", min_value=1, step=1, key="search_roll")

    if st.button("Search Student"):
        found = df[df["roll"] == int(search_roll)]

        if found.empty:
            st.error("No student found for this roll number.")
        else:
            st.dataframe(found, use_container_width=True)

            row = found.iloc[0]
            result = predict_result(
                model,
                float(row["attendance"]),
                float(row["sessional1"]),
                float(row["sessional2"]),
                float(row["sessional3"])
            )

            st.markdown("### Predicted Result")
            a1, a2, a3 = st.columns(3)
            a1.metric("Predicted EndSem", result["predicted_endsem"])
            a2.metric("Status", result["status"])
            a3.metric("Grade", result["grade"])

            st.write(f"**Trend:** {result['trend']}")
            st.write(f"**Attendance Status:** {result['attendance_status']}")
            st.write(f"**Average Sessional:** {result['average_sessional']}")

            if st.button("Save Prediction to Existing Record"):
                idx = found.index[0]

                df.loc[idx, "predicted_endsem"] = result["predicted_endsem"]
                df.loc[idx, "status"] = result["status"]
                df.loc[idx, "grade"] = result["grade"]
                df.loc[idx, "trend"] = result["trend"]
                df.loc[idx, "attendance_status"] = result["attendance_status"]
                df.loc[idx, "average_sessional"] = result["average_sessional"]

                save_data(df)
                st.success("Prediction saved directly to the same CSV file.")
                st.rerun()

with tab3:
    st.subheader("Current Dataset")
    st.write(f"Total records: **{len(df)}**")
    st.dataframe(df, use_container_width=True)
