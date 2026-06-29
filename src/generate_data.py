from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
DATA_PATH = Path("data/customer_churn.csv")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    n_customers = 2500

    customer_id = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]

    tenure_months = rng.integers(1, 73, n_customers)

    monthly_charges = np.round(rng.normal(75, 22, n_customers), 2)
    monthly_charges = np.clip(monthly_charges, 20, 150)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_customers,
        p=[0.55, 0.25, 0.20],
    )

    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No internet"],
        size=n_customers,
        p=[0.48, 0.37, 0.15],
    )

    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_customers,
        p=[0.38, 0.28, 0.24, 0.10],
    )

    support_tickets = rng.poisson(1.2, n_customers)
    late_payments = rng.poisson(0.45, n_customers)
    complaints_last_6_months = rng.poisson(0.55, n_customers)

    products_used = rng.integers(1, 6, n_customers)

    avg_session_minutes = np.round(rng.normal(38, 15, n_customers), 1)
    avg_session_minutes = np.clip(avg_session_minutes, 3, 120)

    auto_pay = rng.choice(["Yes", "No"], size=n_customers, p=[0.58, 0.42])
    senior_citizen = rng.choice(["Yes", "No"], size=n_customers, p=[0.18, 0.82])

    total_charges = np.round(
        tenure_months * monthly_charges + rng.normal(0, 120, n_customers),
        2,
    )
    total_charges = np.clip(total_charges, 20, None)

    churn_risk = (
        -2.3
        - 0.035 * tenure_months
        + 0.018 * monthly_charges
        + 0.38 * support_tickets
        + 0.45 * late_payments
        + 0.55 * complaints_last_6_months
        - 0.22 * products_used
        - 0.015 * avg_session_minutes
        + np.where(contract_type == "Month-to-month", 1.05, 0)
        + np.where(contract_type == "Two year", -0.65, 0)
        + np.where(internet_service == "Fiber optic", 0.35, 0)
        + np.where(payment_method == "Electronic check", 0.50, 0)
        + np.where(auto_pay == "No", 0.35, 0)
        + np.where(senior_citizen == "Yes", 0.20, 0)
    )

    churn_probability = sigmoid(churn_risk)
    churn = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
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
            "churn": churn,
        }
    )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Churn rate: {df['churn'].mean():.2%}")


if __name__ == "__main__":
    main()