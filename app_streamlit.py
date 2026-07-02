"""
app_streamlit.py
Streamlit inference app for the Credit Score classification model.
"""

import sys
import os
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(
    page_title="Credit Score Predictor",
    layout="centered"
)

st.markdown("""
<style>

.stApp{
    background:#F5F6F8;
}

.block-container{
    max-width:850px;
    padding-top:3rem;
    padding-bottom:3rem;
}

h1{
    color:#1B3A6B;
    font-weight:700;
    text-align:center;
}

.subtitle{
    color:#64748B;
    margin-bottom:2rem;
    text-align:center;
}

[data-testid="stForm"]{

    background:rgba(255,255,255,0.75);

    backdrop-filter:blur(10px);

    border:1px solid rgba(209,221,244,0.8);

    border-radius:18px;

    padding:2rem;

    margin-top:1rem;

    box-shadow:0 8px 24px rgba(27,58,107,0.08);
}

/* Customer Information */

h3{
    color:#1B3A6B;
}

/* LABEL INPUT */

[data-testid="stWidgetLabel"]{

    color:#1B3A6B !important;

    font-weight:600 !important;
}

[data-testid="stWidgetLabel"] p{

    color:#1B3A6B !important;

    font-weight:600 !important;

    font-size:14px !important;
}

/* INPUT BOX */

.stNumberInput input,

.stTextInput input{

    border-radius:10px;
}

/* BUTTON */

div[data-testid="stFormSubmitButton"] button{

    width:100%;

    border:none;

    border-radius:12px;

    background:#1B3A6B;

    color:white;

    font-weight:600;

    height:48px;

    transition:0.2s;
}

div[data-testid="stFormSubmitButton"] button:hover{

    background:#142C52;

    color:white;
}

/* RESULT CARD */

.result-card{

    background:rgba(255,255,255,0.75);

    backdrop-filter:blur(10px);

    border:1px solid rgba(209,221,244,0.8);

    border-radius:18px;

    padding:1.5rem;

    box-shadow:0 8px 24px rgba(27,58,107,0.08);

    text-align:center;

    margin-top:1rem;
}

.result-good{

    color:#1B7A4D;

    font-size:32px;

    font-weight:700;
}

.result-standard{

    color:#B58A00;

    font-size:32px;

    font-weight:700;
}

.result-poor{

    color:#B5342B;

    font-size:32px;

    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    return model, preprocessor

best_model, preprocessor = load_artifacts()

st.title("Credit Score Predictor")

st.markdown(
"""
<p class="subtitle">
Enter customer financial information to predict their credit score category.
</p>
""",
unsafe_allow_html=True
)

with st.form("credit_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", min_value=18, max_value=100, value=30)

        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)

        monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0)

        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=3)

        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, value=4)

        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)

        num_of_loan = st.number_input("Number of Loans", min_value=0, value=2)

        delay_from_due_date = st.number_input("Delay from Due Date (days)", value=10)

        num_of_delayed_payment = st.number_input(
            "Number of Delayed Payments",
            min_value=0,
            value=5
        )

    with col2:

        changed_credit_limit = st.number_input(
            "Changed Credit Limit",
            value=10.0
        )

        num_credit_inquiries = st.number_input(
            "Number of Credit Inquiries",
            min_value=0,
            value=4
        )

        outstanding_debt = st.number_input(
            "Outstanding Debt",
            min_value=0.0,
            value=1000.0
        )

        credit_utilization_ratio = st.number_input(
            "Credit Utilization Ratio (%)",
            min_value=0.0,
            value=30.0
        )

        credit_history_age = st.text_input(
            "Credit History Age",
            value="22 Years and 3 Months"
        )

        total_emi_per_month = st.number_input(
            "Total EMI per Month",
            min_value=0.0,
            value=150.0
        )

        amount_invested_monthly = st.number_input(
            "Amount Invested Monthly",
            min_value=0.0,
            value=100.0
        )

        monthly_balance = st.number_input(
            "Monthly Balance",
            value=350.0
        )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        month = st.selectbox(
            "Month",
            ["April","August","February","January",
             "July","June","March","May"]
        )

        occupation = st.selectbox(
            "Occupation",
            ["Accountant","Architect","Developer","Doctor",
             "Engineer","Entrepreneur","Journalist",
             "Lawyer","Manager","Mechanic",
             "Media_Manager","Musician",
             "Scientist","Teacher","Unknown",
             "Writer"]
        )

        credit_mix = st.selectbox(
            "Credit Mix",
            ["Bad","Good","Standard","Unknown"]
        )

    with col4:

        payment_of_min_amount = st.selectbox(
            "Payment of Min Amount",
            ["No","Unknown","Yes"]
        )

        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "High_spent_Large_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Small_value_payments",
                "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Small_value_payments",
                "Unknown"
            ]
        )

        type_of_loan = st.text_input(
            "Type of Loan",
            value="Mortgage Loan, Personal Loan"
        )

    predict_clicked = st.form_submit_button(
        "Predict Credit Score"
    )
    
if predict_clicked:
    try:
        raw_input = pd.DataFrame([{
            "Age": age,
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_inhand_salary,
            "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card,
            "Interest_Rate": interest_rate,
            "Num_of_Loan": num_of_loan,
            "Delay_from_due_date": delay_from_due_date,
            "Num_of_Delayed_Payment": num_of_delayed_payment,
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Credit_Inquiries": num_credit_inquiries,
            "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Credit_History_Age": credit_history_age,
            "Total_EMI_per_month": total_emi_per_month,
            "Amount_invested_monthly": amount_invested_monthly,
            "Monthly_Balance": monthly_balance,
            "Month": month,
            "Occupation": occupation,
            "Credit_Mix": credit_mix,
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Payment_Behaviour": payment_behaviour,
            "Type_of_Loan": type_of_loan
        }])

        X_input = preprocessor.transform_for_inference(raw_input)
        prediction = best_model.predict(X_input)[0]

        result_class = {
            "Good": "result-good",
            "Standard": "result-standard",
            "Poor": "result-poor"
        }.get(prediction, "result-standard")

        st.markdown(
            f"""
            <div class="result-card">

            <h3>Prediction Result</h3>

            <p class="{result_class}">
            {prediction}
            </p>

            </div>
            """,
            unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")