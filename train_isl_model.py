"""
Train a Random Forest classifier for Indian Sign Language recognition
using hand landmark coordinates from isl_dataset.csv.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Configuration
# ---------------------------------------------------------------------------
CSV_FILE = "isl_dataset.csv"
MODEL_FILE = "isl_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return feature matrix X and target vector y."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: {os.path.abspath(filepath)}\n"
            "Run collect_isl_data.py first to create isl_dataset.csv."
        )

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(
            f"Dataset file is empty: {os.path.abspath(filepath)}\n"
            "Collect gesture samples before training."
        )

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    feature_columns = [col for col in df.columns if col.startswith(("x", "y", "z"))]
    if not feature_columns:
        raise ValueError(
            "No hand landmark coordinate columns found (expected x0, y0, z0, ...)."
        )

    X = df[feature_columns]
    y = df["label"]

    if X.isnull().any().any():
        raise ValueError("Feature columns contain missing values. Clean the dataset first.")

    if y.isnull().any():
        raise ValueError("Target labels contain missing values. Clean the dataset first.")

    if y.nunique() < 2:
        raise ValueError(
            f"Need at least 2 gesture classes to train; found {y.nunique()}."
        )

    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Split data, train RandomForestClassifier, and print metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """Persist the trained model with joblib."""
    joblib.dump(model, filepath)
    print(f"Model saved to: {os.path.abspath(filepath)}")


def main() -> None:
    try:
        print(f"Loading dataset from: {CSV_FILE}")
        X, y = load_dataset(CSV_FILE)

        print(f"Samples: {len(X)} | Features: {X.shape[1]} | Classes: {y.nunique()}")
        print(f"Labels: {sorted(y.unique().tolist())}")

        model = train_and_evaluate(X, y)
        save_model(model, MODEL_FILE)

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(
            f"Error: Dataset file is empty or invalid: {os.path.abspath(CSV_FILE)}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error during training: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
