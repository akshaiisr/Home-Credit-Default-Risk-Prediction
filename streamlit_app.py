import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Home Credit Default Risk Predictor",
    page_icon="🏦",
    layout="centered"
)


# ---------------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------------

# This makes the app work correctly both locally and
# when deployed from GitHub / Streamlit Community Cloud.
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "model1.pkl")
    columns = joblib.load(BASE_DIR / "columns1.pkl")
    defaults = joblib.load(BASE_DIR / "defaults1.pkl")
    cat_maps = joblib.load(BASE_DIR / "cat_maps1.pkl")

    return model, columns, defaults, cat_maps


try:
    model, columns, defaults, cat_maps = load_model_files()

except Exception as e:
    st.error("The model files could not be loaded.")
    st.write("Please make sure the following files are in the repository:")
    st.code(
        """
model1.pkl
columns1.pkl
defaults1.pkl
cat_maps1.pkl
        """
    )
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# DECISION THRESHOLD
# ---------------------------------------------------------

# Threshold selected from validation rather than
# automatically assuming 0.50.
DECISION_THRESHOLD = 0.463


# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------

st.title("🏦 Home Credit — Default Risk Predictor")

st.write(
    """
    This application estimates the probability that a loan applicant
    may default using a trained machine-learning model.

    Enter the main applicant and loan information below.
    """
)

st.divider()


# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------

st.subheader("Applicant Information")


ext2 = st.slider(
    "External Credit Score 2",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01,
    help="External credit score. Higher values generally indicate lower credit risk."
)


ext3 = st.slider(
    "External Credit Score 3",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01,
    help="Another external credit score used by the model."
)


income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=170000.0,
    step=5000.0
)


age = st.slider(
    "Age",
    min_value=20,
    max_value=70,
    value=40
)


employed_years = st.slider(
    "Years Employed",
    min_value=0,
    max_value=40,
    value=5
)


st.subheader("Loan Information")


credit = st.number_input(
    "Loan Amount (AMT_CREDIT)",
    min_value=0.0,
    value=600000.0,
    step=10000.0
)


annuity = st.number_input(
    "Annuity / Repayment Amount",
    min_value=0.0,
    value=27000.0,
    step=1000.0
)


goods = st.number_input(
    "Goods Price",
    min_value=0.0,
    value=540000.0,
    step=10000.0
)


st.divider()


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button(
    "Predict Default Risk",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # Start from realistic training-data defaults
    # -----------------------------------------------------
    #
    # The previous version created every unused feature as 0.
    # That could produce unrealistic applicant records.
    #
    # defaults1.pkl contains typical values calculated from
    # the training data.
    # -----------------------------------------------------

    row = pd.DataFrame([defaults])

    # Ensure exactly the same feature order used during training
    row = row.reindex(columns=columns)


    # -----------------------------------------------------
    # Replace default values with user-entered information
    # -----------------------------------------------------

    if "EXT_SOURCE_2" in row.columns:
        row.loc[0, "EXT_SOURCE_2"] = ext2

    if "EXT_SOURCE_3" in row.columns:
        row.loc[0, "EXT_SOURCE_3"] = ext3

    if "AMT_CREDIT" in row.columns:
        row.loc[0, "AMT_CREDIT"] = credit

    if "AMT_ANNUITY" in row.columns:
        row.loc[0, "AMT_ANNUITY"] = annuity

    if "AMT_GOODS_PRICE" in row.columns:
        row.loc[0, "AMT_GOODS_PRICE"] = goods

    if "AMT_INCOME_TOTAL" in row.columns:
        row.loc[0, "AMT_INCOME_TOTAL"] = income

    if "AGE" in row.columns:
        row.loc[0, "AGE"] = age

    # Original Home Credit dataset stores age as negative days
    if "DAYS_BIRTH" in row.columns:
        row.loc[0, "DAYS_BIRTH"] = -(age * 365)

    # Employment duration is also stored as negative days
    if "DAYS_EMPLOYED" in row.columns:
        row.loc[0, "DAYS_EMPLOYED"] = -(employed_years * 365)


    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    try:

        probability = model.predict_proba(row)[0][1]

        risk_percentage = probability * 100


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.subheader("Prediction Result")

        st.metric(
            label="Estimated Default Probability",
            value=f"{risk_percentage:.1f}%"
        )


        # Risk progress indicator
        st.progress(
            min(max(float(probability), 0.0), 1.0)
        )


        # -------------------------------------------------
        # RISK CLASSIFICATION
        # -------------------------------------------------

        if probability >= DECISION_THRESHOLD:

            st.error(
                f"""
                ⚠️ Higher Default Risk

                The predicted probability of default is
                **{risk_percentage:.1f}%**.

                This is above the model's decision threshold
                of **{DECISION_THRESHOLD * 100:.1f}%**.
                """
            )

        else:

            st.success(
                f"""
                ✅ Lower Default Risk

                The predicted probability of default is
                **{risk_percentage:.1f}%**.

                This is below the model's decision threshold
                of **{DECISION_THRESHOLD * 100:.1f}%**.
                """
            )


        # -------------------------------------------------
        # SHOW INPUT SUMMARY
        # -------------------------------------------------

        with st.expander("View Applicant Input Summary"):

            input_summary = pd.DataFrame(
                {
                    "Variable": [
                        "External Score 2",
                        "External Score 3",
                        "Annual Income",
                        "Loan Amount",
                        "Annuity",
                        "Goods Price",
                        "Age",
                        "Years Employed"
                    ],

                    "Value": [
                        round(ext2, 2),
                        round(ext3, 2),
                        f"{income:,.0f}",
                        f"{credit:,.0f}",
                        f"{annuity:,.0f}",
                        f"{goods:,.0f}",
                        age,
                        employed_years
                    ]
                }
            )

            st.dataframe(
                input_summary,
                hide_index=True,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.divider()

with st.expander("About the Model"):

    st.write(
        """
        The model was developed using the Home Credit Default Risk
        dataset.

        The dataset contains information about more than 300,000
        loan applicants.

        The target variable represents:

        **0 — Applicant repaid the loan**

        **1 — Applicant defaulted**

        Because the dataset is highly imbalanced, model performance
        was evaluated primarily using ROC-AUC rather than accuracy.

        Logistic Regression was used as a baseline model and
        LightGBM was used as the more advanced model.

        Hyperparameter tuning was performed before selecting the
        final model.
        """
    )


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.caption(
    """
    Educational project only. The predicted probability is a
    machine-learning estimate and should not be used as the sole
    basis for a real lending or credit decision.
    """
)
