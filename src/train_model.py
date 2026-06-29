from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("models/best_churn_model.joblib")
OUTPUTS_DIR = Path("outputs")
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run this first: python src/generate_data.py"
        )

    df = pd.read_csv(DATA_PATH)
    return df


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0,
    )

    return preprocessor


def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    results = {
        "model": model_name,
        "precision_churn": report["1"]["precision"],
        "recall_churn": report["1"]["recall"],
        "f1_churn": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
        "roc_auc": roc_auc,
    }

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC-AUC: {roc_auc:.4f}")

    return results, y_pred, y_probability


def save_feature_importance(best_model):
    preprocessor = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    top_features = importance_df.head(15).sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 15 Churn Prediction Features")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    print(f"\nFeature importance saved to: {FEATURE_IMPORTANCE_PATH}")
    print("\nTop 10 churn drivers:")
    print(importance_df.head(10).to_string(index=False))


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("Dataset preview:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nChurn distribution:")
    print(df["churn"].value_counts(normalize=True).rename("percentage"))

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    all_results = []
    trained_models = {}
    prediction_store = {}

    for model_name, classifier in models.items():
        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        results, y_pred, y_probability = evaluate_model(
            model_name, pipeline, X_test, y_test
        )

        all_results.append(results)
        trained_models[model_name] = pipeline
        prediction_store[model_name] = {
            "y_pred": y_pred,
            "y_probability": y_probability,
        }

    results_df = pd.DataFrame(all_results).sort_values(
        "f1_churn", ascending=False
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 70)
    print("Model comparison")
    print("=" * 70)
    print(results_df.to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    best_predictions = prediction_store[best_model_name]
    best_y_pred = best_predictions["y_pred"]
    best_y_probability = best_predictions["y_probability"]

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, best_y_pred),
        display_labels=["Not churned", "Churned"],
    ).plot()

    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, best_y_probability)
    plt.title(f"ROC Curve - {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    save_feature_importance(best_model)

    print("\nCharts saved in the outputs folder.")


if __name__ == "__main__":
    main()