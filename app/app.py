import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64

model_path = os.path.join("models", "xgboost_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide", page_icon="")
st.title("Employee Attrition Prediction & HR Analytics App")

st.markdown("""
Welcome to the **Employee Attrition Prediction App**!
This system uses a machine learning model to analyze employee data and predict who is likely to **resign**.

Use this app to explore data, generate predictions, and analyze HR trends.
""")

with st.sidebar:
    st.header("About the App")
    st.write("Built with Streamlit, XGBoost, and Plotly. Designed to help HR professionals take data-driven action.")
    st.markdown("**Created by Sidhaanth Kapoor**")
    st.markdown("---")
    st.image("figures/HR_Logo.png", width=200)
    st.markdown("---")
    st.subheader("Model Details")
    st.code("models/xgboost_model.pkl")

st.header("Upload Employee Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with employee data:", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.header("Visual Analytics")
    selected_col = st.selectbox("Select numeric column to visualize:", df.select_dtypes(include=np.number).columns)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.header("Batch Prediction")
    if st.button("Run Predictions on Uploaded Data"):
        input_features = [
            "MonthlyIncome", "Age", "TotalWorkingYears",
            "DailyRate", "MonthlyRate", "OverTime", "DistanceFromHome", "HourlyRate"
        ]

        df_predict = df.copy()
        if "OverTime" in df_predict.columns:
            df_predict["OverTime"] = df_predict["OverTime"].map({"Yes": 1, "No": 0})

        predictions = model.predict(df_predict[input_features])
        probabilities = model.predict_proba(df_predict[input_features])[:, 1]

        df_predict["Prediction"] = predictions
        df_predict["Resignation Probability"] = probabilities

        st.subheader("Prediction Results")
        st.dataframe(df_predict[["Prediction", "Resignation Probability"]].head())

        csv = df_predict.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="attrition_predictions.csv">📥 Download Results</a>'
        st.markdown(href, unsafe_allow_html=True)

st.header("Single Employee Prediction")
with st.form(key="input_form"):
    st.subheader("Enter Employee Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=30)
        total_working_years = st.slider("Total Working Years", 0, 40, 5)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000, step=500)
        daily_rate = st.slider("Daily Rate ($)", 100, 1500, 800)
        overtime = st.selectbox("OverTime", ["Yes", "No"])

    with col2:
        hourly_rate = st.slider("Hourly Rate ($)", 10, 200, 60)
        monthly_rate = st.slider("Monthly Rate ($)", 1000, 20000, 10000)
        distance_from_home = st.slider("Distance From Home (miles)", 0, 60, 10)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    overtime_binary = 1 if overtime == "Yes" else 0
    input_data = pd.DataFrame([{
        "MonthlyIncome": monthly_income,
        "Age": age,
        "TotalWorkingYears": total_working_years,
        "DailyRate": daily_rate,
        "MonthlyRate": monthly_rate,
        "OverTime": overtime_binary,
        "DistanceFromHome": distance_from_home,
        "HourlyRate": hourly_rate
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.success("Prediction Completed!")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("This employee is **likely to resign.**")
    else:
        st.success("This employee is **likely to stay.**")

    st.metric(label="Probability of Resignation", value=f"{probability:.2%}")

    with st.expander("View Input Data"):
        st.dataframe(input_data)

    with st.expander("What Do These Features Mean?"):
        st.markdown("""
        - **Monthly Income**: Salary per month.
        - **Age**: Current age of the employee.
        - **Total Working Years**: Total years of experience.
        - **Daily Rate**: Daily wage.
        - **Monthly Rate**: Base monthly compensation.
        - **Hourly Rate**: Hourly wage.
        - **OverTime**: Whether the employee regularly works overtime.
        - **Distance From Home**: Commute distance in miles.
        """)

st.markdown("---")
st.markdown("<center><small>Made using Streamlit • Model: XGBoost • Created by Sidhaanth Kapoor</small></center>", unsafe_allow_html=True)
