from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("models/best_churn_model.joblib")
FEATURE_IMPORTANCE_PATH = Path("outputs/feature_importance.csv")


st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)


def risk_label(probability):
    if probability >= 0.70:
        return "High churn risk"
    if probability >= 0.40:
        return "Medium churn risk"
    return "Low churn risk"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Customer Churn Prediction Dashboard")

st.write(
    "This dashboard predicts customer churn using customer behavior, billing, and account data."
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Run: python src/generate_data.py")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python src/train_model.py")
    st.stop()


df = load_data()
model = load_model()


tab1, tab2, tab3 = st.tabs(["Overview", "Model Insights", "Predict Churn"])


with tab1:
    st.header("Dataset Overview")

    total_customers = len(df)
    churned_customers = int(df["churn"].sum())
    churn_rate = df["churn"].mean() * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Churn Distribution")
    churn_counts = df["churn"].map({0: "Not Churned", 1: "Churned"}).value_counts()
    st.bar_chart(churn_counts)


with tab2:
    st.header("Model Insights")

    if FEATURE_IMPORTANCE_PATH.exists():
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

        st.subheader("Top Churn Drivers")

        top_features = importance_df.head(10).copy()
        top_features["feature"] = top_features["feature"].str.replace(
            "numeric__", "", regex=False
        )
        top_features["feature"] = top_features["feature"].str.replace(
            "categorical__", "", regex=False
        )

        st.dataframe(top_features, use_container_width=True)

        chart_data = top_features.set_index("feature")["importance"]
        st.bar_chart(chart_data)
    else:
        st.warning("Feature importance file not found. Run: python src/train_model.py")

    col1, col2 = st.columns(2)

    with col1:
        if Path("outputs/confusion_matrix.png").exists():
            st.image("outputs/confusion_matrix.png", caption="Confusion Matrix")

    with col2:
        if Path("outputs/roc_curve.png").exists():
            st.image("outputs/roc_curve.png", caption="ROC Curve")


with tab3:
    st.header("Predict Churn for a New Customer")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure_months = st.slider("Tenure Months", 1, 72, 5)
        monthly_charges = st.number_input("Monthly Charges", 20.0, 150.0, 105.5)
        total_charges = st.number_input("Total Charges", 20.0, 10000.0, 527.5)
        products_used = st.slider("Products Used", 1, 5, 1)

    with col2:
        contract_type = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No internet"],
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        )
        auto_pay = st.selectbox("Auto Pay", ["Yes", "No"])

    with col3:
        support_tickets = st.slider("Support Tickets", 0, 10, 4)
        late_payments = st.slider("Late Payments", 0, 10, 2)
        complaints_last_6_months = st.slider("Complaints Last 6 Months", 0, 10, 3)
        avg_session_minutes = st.slider("Average Session Minutes", 3.0, 120.0, 12.5)
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])

    new_customer = pd.DataFrame(
        [
            {
                "tenure_months": tenure_months,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "contract_type": contract_type,
                "internet_service": internet_service,
                "payment_method": payment_method,
                "support_tickets": support_tickets,
                "late_payments": late_payments,
                "complaints_last_6_months": complaints_last_6_months,
                "products_used": products_used,
                "avg_session_minutes": avg_session_minutes,
                "auto_pay": auto_pay,
                "senior_citizen": senior_citizen,
            }
        ]
    )

    if st.button("Predict Churn"):
        churn_probability = model.predict_proba(new_customer)[:, 1][0]
        prediction = model.predict(new_customer)[0]
        risk = risk_label(churn_probability)

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Prediction",
            "Will Churn" if prediction == 1 else "Will Not Churn",
        )
        result_col2.metric("Churn Probability", f"{churn_probability:.2%}")
        result_col3.metric("Risk Level", risk)

        if churn_probability >= 0.70:
            st.error("This customer has a high risk of churn.")
        elif churn_probability >= 0.40:
            st.warning("This customer has a medium risk of churn.")
        else:
            st.success("This customer has a low risk of churn.")