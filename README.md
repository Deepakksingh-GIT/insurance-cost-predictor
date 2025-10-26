# insurance-cost-predictor
End-to-end Python project to predict individual medical insurance costs using Machine Learning. Includes EDA, data preprocessing, regression models, MLflow tracking, and an interactive Streamlit web app.

Project Overview
Dataset info
Features
EDA insights summary
Model training
Streamlit app usage
Setup instructions
Author info

# Clone repository
git clone https://github.com/<username>/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction

# Create virtual environment
python -m venv venv
# Activate venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run MLflow UI (optional)
mlflow ui

# Train model
python src/train.py

# Launch Streamlit app
streamlit run app.py
