# Customer Churn Prediction Dashboard

A machine learning project that predicts customer churn using behavioral, transactional, and account-level customer data. The project includes a full ML workflow from data generation and preprocessing to model training, evaluation, feature importance analysis, and live dashboard deployment.

## Live Dashboard

View the deployed Streamlit app here:

[Customer Churn Prediction Dashboard](https://customer-churn-prediction-ckigmdkovc3rud3tuyxkpf.streamlit.app/)

## Project Overview

Customer churn is one of the most important business problems for subscription-based and service-driven companies. Retaining existing customers is often more cost-effective than acquiring new ones, so identifying customers who are likely to leave can help businesses take timely action.

This project uses machine learning classification models to predict whether a customer is likely to churn. It also provides churn probability, risk level, key churn drivers, and business recommendations through an interactive Streamlit dashboard.

## Key Features

* Built an end-to-end churn prediction pipeline using Python and Scikit-learn
* Generated a synthetic customer churn dataset with 2,500 records
* Performed data preprocessing for numerical and categorical features
* Trained and compared Random Forest and Gradient Boosting classifiers
* Evaluated models using precision, recall, F1-score, accuracy, and ROC-AUC
* Selected the best model based on churn F1-score
* Saved the trained model using Joblib
* Built an interactive dashboard using Streamlit
* Deployed the dashboard live using Streamlit Community Cloud

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit
* GitHub
* Streamlit Community Cloud

## Machine Learning Workflow

The project follows a complete machine learning workflow:

1. Data generation
2. Data loading and validation
3. Exploratory data analysis
4. Feature preprocessing
5. Train-test split
6. Model training
7. Model comparison
8. Model evaluation
9. Feature importance analysis
10. Model saving
11. Live prediction dashboard deployment

## Dataset

The dataset contains customer-level features related to account behavior, payment patterns, service usage, and engagement.

### Features Used

* Tenure in months
* Monthly charges
* Total charges
* Contract type
* Internet service type
* Payment method
* Support tickets
* Late payments
* Complaints in the last 6 months
* Products used
* Average session minutes
* Auto-pay status
* Senior citizen status

### Target Variable

`churn`

* `0`: Customer did not churn
* `1`: Customer churned

## Models Used

Two classification models were trained and compared:

* Random Forest Classifier
* Gradient Boosting Classifier

The final model was selected based on churn F1-score because churn prediction requires a balance between correctly identifying churners and avoiding too many false positives.

## Evaluation Metrics

The models were evaluated using:

* Precision
* Recall
* F1-score
* Accuracy
* ROC-AUC

## Business Insights

The model identified several important churn drivers, including:

* Short customer tenure
* High monthly charges
* Month-to-month contracts
* High number of support tickets
* Customer complaints
* Low product usage
* Late payments

These insights can help customer success, marketing, and retention teams take proactive actions such as personalized offers, support follow-ups, loyalty benefits, and contract upgrade campaigns.

## Dashboard Features

The Streamlit dashboard includes:

* Customer dataset overview
* Churn rate summary
* Churn distribution chart
* Feature importance visualization
* Confusion matrix
* ROC curve
* Interactive customer churn prediction form
* Churn probability score
* Risk level classification
* Business recommendations

## Project Structure

```text
Customer Churn Prediction/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   └── best_churn_model.joblib
│
├── outputs/
│   ├── model_metrics.json
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── src/
    ├── generate_data.py
    ├── train_model.py
    └── predict.py
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/souravgupta166/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Generate the dataset

```bash
python src/generate_data.py
```

### 6. Train the model

```bash
python src/train_model.py
```

### 7. Run prediction script

```bash
python src/predict.py
```

### 8. Run the Streamlit dashboard

```bash
streamlit run app.py
```

## Example Prediction Output

```text
Customer Churn Prediction
-----------------------------------
Prediction: Will churn
Churn probability: 83.55%
Risk level: High churn risk
```

## Business Use Case

This project demonstrates how machine learning can support customer retention strategies. By identifying high-risk customers early, businesses can take targeted actions to reduce churn, improve customer satisfaction, and increase lifetime value.

## Future Improvements

* Use a real-world telecom or SaaS churn dataset
* Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV
* Add SHAP explainability for individual predictions
* Store prediction history in a database
* Add user authentication for dashboard access
* Improve dashboard styling and visual design
* Add downloadable prediction reports

## Author

Sourav Gupta

GitHub: [souravgupta166](https://github.com/souravgupta166)
