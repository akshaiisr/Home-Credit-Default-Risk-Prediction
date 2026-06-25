# Home Credit Default Risk Prediction

Predicts the probability that a loan applicant will default, using the
Home Credit dataset. Built as an end-to-end data science project:
data analysis, model training, and a live web app.

## Business Problem
Lenders lose money when borrowers default. This project predicts default
risk before a loan is approved, helping the business make safer decisions.

## Dataset
Home Credit `application_train.csv` — 307,511 applicants, 122 columns.
Default rate is 8% (imbalanced).

## Approach
- Cleaned data: dropped columns >50% empty, converted DAYS_BIRTH to age.
- Baseline: Logistic Regression — AUC 0.7455
- Main model: LightGBM (handles imbalance via scale_pos_weight) — AUC 0.7568
- Top features: EXT_SOURCE_3, EXT_SOURCE_2, AMT_CREDIT

## Project Structure
- `streamlit_app.py` — web app for live predictions
- `requirements.txt` — Python libraries
- `model.pkl`, `columns.pkl`, `cat_maps.pkl` — trained model files
- `notebooks/` — EDA and model training

## Live App
Streamlit link — (https://home-credit-default-risk-prediction-dycgn3eghwv3q9nykkceqv.streamlit.app/)

## Demo Video
[Video link — add after recording]

## How to Run
1. Install: `pip install -r requirements.txt`
2. Run: `streamlit run streamlit_app.py`
