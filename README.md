# Employee Salary Prediction System

A professional, production-ready Machine Learning system built with Python to predict employee base salaries based on their years of experience, education level, and skill ratings.

## 🚀 Project Overview
This project implements an end-to-end Machine Learning pipeline using a `RandomForestRegressor`. It handles data pre-processing (scaling numerical data and one-hot encoding categorical data) smoothly through an integrated Scikit-Learn `Pipeline`.

## 📁 Repository Structure
* `salary_data.csv` - The training dataset containing employee metrics.
* `train_model.py` - Production training script that validates data, trains the pipeline, and serializes the model.
* `predict.py` - User-interactive inference script that loads the saved model for real-time predictions.
* `requirements.txt` - Python project dependencies.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/employee-salary-prediction.git
   cd employee-salary-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 How to Run

1. **Train and Save the Model:**
   ```bash
   python train_model.py
   ```
   This will train the Random Forest model and save it as `salary_predictor_model.pkl`.

2. **Run Real-Time Prediction Dashboard (CLI):**
   ```bash
   python predict.py
   ```
