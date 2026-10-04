"""
Project: Employee Salary Prediction System
Author: Machine Learning Engineer
Date: 2026
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found!")
    df = pd.read_csv(file_path)
    print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
    return df

def build_pipeline():
    categorical_features = ['Education_Level']
    numerical_features = ['Years_Experience', 'Skill_Rating']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def main():
    data_path = "salary_data.csv"
    df = load_data(data_path)

    X = df[['Years_Experience', 'Education_Level', 'Skill_Rating']]
    y = df['Base_Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training the Machine Learning model...")
    model_pipeline = build_pipeline()
    model_pipeline.fit(X_train, y_train)
    print("[SUCCESS] Model trained successfully!")

    y_pred = model_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*30 + " MODEL PERFORMANCE " + "="*30)
    print(f"Mean Absolute Error (MAE) : ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R-squared (Accuracy Score)   : {r2 * 100:.2f}%")
    print("="*71 + "\n")

    model_filename = "salary_predictor_model.pkl"
    joblib.dump(model_pipeline, model_filename)
    print(f"[SUCCESS] Model saved as '{model_filename}'")

if __name__ == '__main__':
    main()
