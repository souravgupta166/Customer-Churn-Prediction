from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/best_churn_model.joblib")


def risk_label(probability):
    if probability >= 0.70:
        return "High churn risk"
    if probability >= 0.40:
        return "Medium churn risk"
    return "Low churn risk"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run this first: python src/train_model.py"
        )

    model = joblib.load(MODEL_PATH)

    new_customer = pd.DataFrame(
        [
            {
                "tenure_months": 5,
                "monthly_charges": 105.50,
                "total_charges": 527.50,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "support_tickets": 4,
                "late_payments": 2,
                "complaints_last_6_months": 3,
                "products_used": 1,
                "avg_session_minutes": 12.5,
                "auto_pay": "No",
                "senior_citizen": "No",
            }
        ]
    )

    churn_probability = model.predict_proba(new_customer)[:, 1][0]
    prediction = model.predict(new_customer)[0]

    print("Customer Churn Prediction")
    print("-" * 35)
    print(f"Prediction: {'Will churn' if prediction == 1 else 'Will not churn'}")
    print(f"Churn probability: {churn_probability:.2%}")
    print(f"Risk level: {risk_label(churn_probability)}")


if __name__ == "__main__":
    main()