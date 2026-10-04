"""
Project: Employee Salary Prediction System - Web Application
Author: Machine Learning Engineer
Date: 2026
"""

import os
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Employee Salary Prediction System")
st.write("Enter the employee's professional metrics below to estimate their ideal base salary using Machine Learning.")

MODEL_PATH = "salary_predictor_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found! Please run 'train_model.py' first to generate the model before launching the app.")
else:
    st.subheader("📋 Employee Metrics")
    
    with st.form("prediction_form"):
        years_exp = st.number_input(
            "Years of Experience", 
            min_value=0.0, 
            max_value=40.0, 
            value=3.5, 
            step=0.5,
            help="Total professional corporate experience in years."
        )
        
        edu_level = st.selectbox(
            "Highest Education Level",
            options=["Bachelor", "Master", "PhD"],
            index=0
        )
        
        skill_rating = st.slider(
            "Technical Skill Rating",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Rate the core technical capabilities from 1.0 (Beginner) to 5.0 (Expert)."
        )
        
        submit_btn = st.form_submit_button("Calculate Estimated Salary")
    
    if submit_btn:
        input_df = pd.DataFrame([{
            'Years_Experience': years_exp,
            'Education_Level': edu_level,
            'Skill_Rating': skill_rating
        }])
        
        prediction = model.predict(input_df)
        
        st.success("🎉 Prediction Completed Successfully!")
        st.metric(
            label="Estimated Base Salary (Annual)",
            value=f"${prediction:,.2f}"
        )
        st.info(f"💡 This estimation is highly optimized for a **{edu_level}** degree holder possessing a technical index rating of **{skill_rating}/5.0**.")