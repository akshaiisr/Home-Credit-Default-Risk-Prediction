import streamlit as st
import pandas as pd
import joblib

# load saved files
model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')

st.title("Home Credit — Default Risk Predictor")
st.write("Enter applicant details to estimate default probability.")

# collect the key inputs
ext2 = st.slider("External Score 2 (0-1)", 0.0, 1.0, 0.5)
ext3 = st.slider("External Score 3 (0-1)", 0.0, 1.0, 0.5)
credit = st.number_input("Loan amount (AMT_CREDIT)", value=600000)
annuity = st.number_input("Annuity (yearly repayment)", value=27000)
goods = st.number_input("Goods price", value=540000)
income = st.number_input("Annual income", value=170000)
age = st.slider("Age", 20, 70, 40)
employed_years = st.slider("Years employed", 0, 40, 5)

if st.button("Predict default risk"):
    # start every feature at 0, then fill the ones we collected
    row = pd.DataFrame([[0]*len(columns)], columns=columns)
    row['EXT_SOURCE_2'] = ext2
    row['EXT_SOURCE_3'] = ext3
    row['AMT_CREDIT'] = credit
    row['AMT_ANNUITY'] = annuity
    row['AMT_GOODS_PRICE'] = goods
    row['AMT_INCOME_TOTAL'] = income
    row['AGE'] = age
    row['DAYS_BIRTH'] = -age * 365
    row['DAYS_EMPLOYED'] = -employed_years * 365

    prob = model.predict_proba(row)[0][1]
    st.subheader(f"Default risk: {prob*100:.1f}%")
    if prob > 0.5:
        st.error("High risk — likely to default")
    else:
        st.success("Low risk — likely to repay")
