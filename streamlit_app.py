import streamlit as st
import pandas as pd
import joblib

# Load the tuned model and the typical/default values used for fields
# that are not entered manually in the web form.
model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')
defaults = joblib.load('defaults.pkl')

# Threshold selected from the validation ROC curve (Youden's J statistic).
DECISION_THRESHOLD = 0.463

st.set_page_config(page_title='Home Credit Default Risk Predictor', page_icon='📊')

st.title('Home Credit — Default Risk Predictor')
st.write(
    'Enter the key applicant details below. The model estimates the probability '
    'of default using the tuned LightGBM model.'
)

ext2 = st.slider('External Score 2 (0–1)', 0.0, 1.0, 0.50, 0.01)
ext3 = st.slider('External Score 3 (0–1)', 0.0, 1.0, 0.50, 0.01)
credit = st.number_input('Loan amount (AMT_CREDIT)', min_value=0.0, value=600000.0, step=1000.0)
annuity = st.number_input('Annuity (yearly repayment)', min_value=0.0, value=27000.0, step=500.0)
goods = st.number_input('Goods price', min_value=0.0, value=540000.0, step=1000.0)
income = st.number_input('Annual income', min_value=0.0, value=170000.0, step=1000.0)
age = st.slider('Age', 20, 70, 40)
employed_years = st.slider('Years employed', 0, 40, 5)

if st.button('Predict default risk'):
    # Start from realistic typical values learned from the training data rather
    # than setting every unused feature to zero.
    row = pd.DataFrame([defaults]).reindex(columns=columns)

    # Replace the fields collected in the interface with the applicant values.
    row.loc[0, 'EXT_SOURCE_2'] = ext2
    row.loc[0, 'EXT_SOURCE_3'] = ext3
    row.loc[0, 'AMT_CREDIT'] = credit
    row.loc[0, 'AMT_ANNUITY'] = annuity
    row.loc[0, 'AMT_GOODS_PRICE'] = goods
    row.loc[0, 'AMT_INCOME_TOTAL'] = income
    row.loc[0, 'AGE'] = age
    row.loc[0, 'DAYS_BIRTH'] = -age * 365
    row.loc[0, 'DAYS_EMPLOYED'] = -employed_years * 365

    probability = model.predict_proba(row)[0, 1]

    st.subheader(f'Default risk: {probability * 100:.1f}%')
    st.progress(min(float(probability), 1.0))

    if probability >= DECISION_THRESHOLD:
        st.error('Higher risk — the predicted risk is above the validation-based decision threshold.')
    else:
        st.success('Lower risk — the predicted risk is below the validation-based decision threshold.')

    st.caption(
        f'Decision threshold: {DECISION_THRESHOLD:.3f}. '
        'This is a model estimate for demonstration and should not be used as the sole basis for a real lending decision.'
    )
