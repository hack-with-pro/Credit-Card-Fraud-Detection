import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced fraud detection workflow with feature engineering and reporting."
    )
    parser.add_argument(
        "--dataset",
        default="creditcard.csv",
        help="Path to the fraud dataset CSV. Default: creditcard.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where reports and plots will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of highest-risk transactions to export.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    required_columns = {"Class", "Amount"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    if "Time" in featured.columns:
        featured["Hour"] = ((featured["Time"] // 3600) % 24).astype(int)
        featured["Hour_Sin"] = np.sin(2 * np.pi * featured["Hour"] / 24.0)
        featured["Hour_Cos"] = np.cos(2 * np.pi * featured["Hour"] / 24.0)

        day_period = pd.cut(
            featured["Hour"],
            bins=[-1, 5, 11, 17, 23],
            labels=["night", "morning", "afternoon", "evening"],
        )
        period_dummies = pd.get_dummies(day_period, prefix="Period", dtype=int)
        featured = pd.concat([featured, period_dummies], axis=1)

    featured["Log_Amount"] = np.log1p(featured["Amount"])
    featured["Sqrt_Amount"] = np.sqrt(np.clip(featured["Amount"], a_min=0, a_max=None))
    featured["High_Value_Flag"] = (
        featured["Amount"] >= featured["Amount"].quantile(0.95)
    ).astype(int)
    featured["Extreme_Value_Flag"] = (
        featured["Amount"] >= featured["Amount"].quantile(0.99)
    ).astype(int)
    featured["Amount_Percentile"] = featured["Amount"].rank(pct=True)

    amount_mean = featured["Amount"].mean()
    amount_std = featured["Amount"].std() + 1e-9
    featured["Amount_ZScore"] = (featured["Amount"] - amount_mean) / amount_std

    v_columns = [column for column in featured.columns if column.startswith("V")]
    if v_columns:
        v_frame = featured[v_columns]
        featured["V_Mean"] = v_frame.mean(axis=1)
        featured["V_Std"] = v_frame.std(axis=1)
        featured["V_Abs_Max"] = v_frame.abs().max(axis=1)
        featured["V_Positive_Count"] = (v_frame > 0).sum(axis=1)
        featured["V_Negative_Count"] = (v_frame < 0).sum(axis=1)

    return featured


def fraud_scores(estimator, features: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(features)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return estimator.predict(features).astype(float)


def threshold_search(y_true: pd.Series, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.05, 0.95, 37)
    rows = []

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "f1": f1_score(y_true, preds, zero_division=0),
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, preds),
            }
        )

    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(
        ["f1", "balanced_accuracy", "recall"], ascending=False
    ).iloc[0]
    return float(best_row["threshold"]), threshold_df


def benchmark_models(X_train, y_train, X_valid, y_valid):
    candidates = {
        "Balanced Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    }

    fitted = {}
    results = []

    for name, estimator in candidates.items():
        model = clone(estimator)
        model.fit(X_train, y_train)
        fitted[name] = model

        valid_scores = fraud_scores(model, X_valid)
        valid_preds = (valid_scores >= 0.5).astype(int)

        results.append(
            {
                "model": name,
                "validation_pr_auc": average_precision_score(y_valid, valid_scores),
                "validation_roc_auc": roc_auc_score(y_valid, valid_scores),
                "validation_f1_at_0_50": f1_score(y_valid, valid_preds, zero_division=0),
                "validation_recall_at_0_50": recall_score(
                    y_valid, valid_preds, zero_division=0
                ),
                "validation_precision_at_0_50": precision_score(
                    y_valid, valid_preds, zero_division=0
                ),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    )
    winner_name = results_df.iloc[0]["model"]
    return winner_name, fitted[winner_name], results_df, fitted


def save_feature_importance_plot(model, feature_names, output_dir: Path):
    if not hasattr(model, "feature_importances_"):
        return

    top_features = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"][::-1], top_features["importance"][::-1], color="#1f77b4")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    top_features.to_csv(output_dir / "top_feature_importance.csv", index=False)


def save_score_distribution_plot(scores: np.ndarray, labels: pd.Series, output_dir: Path):
    plt.figure(figsize=(9, 5))
    plt.hist(scores[labels.values == 0], bins=40, alpha=0.7, label="Legitimate", color="#4c78a8")
    plt.hist(scores[labels.values == 1], bins=40, alpha=0.7, label="Fraud", color="#e45756")
    plt.title("Fraud Score Distribution")
    plt.xlabel("Predicted Fraud Score")
    plt.ylabel("Transaction Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close()


def create_business_report(
    dataset_path: Path,
    output_dir: Path,
    best_model_name: str,
    best_threshold: float,
    metrics: dict,
    top_alerts: pd.DataFrame,
):
    summary_lines = [
        "Enhanced Credit Card Fraud Detection Report",
        "=" * 45,
        f"Dataset: {dataset_path}",
        f"Best model: {best_model_name}",
        f"Decision threshold: {best_threshold:.3f}",
        "",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"PR-AUC: {metrics['pr_auc']:.4f}",
        f"F1 score: {metrics['f1']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        "",
        f"Top-alert precision@{len(top_alerts)}: {metrics['precision_at_k']:.4f}",
        f"Top-alert recall@{len(top_alerts)}: {metrics['recall_at_k']:.4f}",
        "",
        "Generated files:",
        "- model_benchmark.csv",
        "- threshold_search.csv",
        "- high_risk_transactions.csv",
        "- feature_importance.png",
        "- score_distribution.png",
        "- confusion_matrix.csv",
    ]

    (output_dir / "business_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(dataset_path)
    featured_df = build_features(raw_df)

    X = featured_df.drop(columns=["Class"])
    y = featured_df["Class"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    best_model_name, best_model, benchmark_df, fitted_models = benchmark_models(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
    benchmark_df.to_csv(output_dir / "model_benchmark.csv", index=False)

    validation_scores = fraud_scores(best_model, X_valid)
    best_threshold, threshold_df = threshold_search(y_valid, validation_scores)
    threshold_df.to_csv(output_dir / "threshold_search.csv", index=False)

    test_scores = fraud_scores(best_model, X_test)
    test_predictions = (test_scores >= best_threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, test_scores),
        "pr_auc": average_precision_score(y_test, test_scores),
        "f1": f1_score(y_test, test_predictions, zero_division=0),
        "precision": precision_score(y_test, test_predictions, zero_division=0),
        "recall": recall_score(y_test, test_predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_predictions),
    }

    confusion = pd.DataFrame(
        confusion_matrix(y_test, test_predictions),
        index=["actual_legitimate", "actual_fraud"],
        columns=["pred_legitimate", "pred_fraud"],
    )
    confusion.to_csv(output_dir / "confusion_matrix.csv")

    alert_frame = X_test.copy()
    alert_frame["Actual_Class"] = y_test.values
    alert_frame["Fraud_Score"] = test_scores
    alert_frame["Risk_Tier"] = pd.cut(
        alert_frame["Fraud_Score"],
        bins=[-0.01, 0.2, 0.5, 0.8, 1.0],
        labels=["low", "medium", "high", "critical"],
    )
    top_alerts = alert_frame.sort_values("Fraud_Score", ascending=False).head(args.top_k)
    top_alerts.to_csv(output_dir / "high_risk_transactions.csv", index=False)

    fraud_hits = int(top_alerts["Actual_Class"].sum())
    metrics["precision_at_k"] = fraud_hits / max(len(top_alerts), 1)
    metrics["recall_at_k"] = fraud_hits / max(int(y_test.sum()), 1)

    save_feature_importance_plot(best_model, X.columns, output_dir)
    save_score_distribution_plot(test_scores, y_test, output_dir)
    create_business_report(
        dataset_path=dataset_path,
        output_dir=output_dir,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
        metrics=metrics,
        top_alerts=top_alerts,
    )

    print("Enhanced workflow completed.")
    print(f"Dataset rows: {len(raw_df):,}")
    print(f"Original features: {raw_df.shape[1] - 1}")
    print(f"Engineered features used: {X.shape[1]}")
    print(f"Best model: {best_model_name}")
    print(f"Chosen threshold: {best_threshold:.3f}")
    print(
        "Test metrics -> "
        f"ROC-AUC: {metrics['roc_auc']:.4f}, "
        f"PR-AUC: {metrics['pr_auc']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"Recall: {metrics['recall']:.4f}"
    )
    print(f"Top-{len(top_alerts)} alert precision: {metrics['precision_at_k']:.4f}")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
