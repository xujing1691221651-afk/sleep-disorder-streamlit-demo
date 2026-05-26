from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "sleep_health_dataset.csv"
ARTIFACT_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "sleep_rf_pipeline.joblib"

TARGET_COLUMN = "sleep_disorder_risk"

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "sleep_duration_hrs",
    "sleep_quality_score",
    "caffeine_mg_before_bed",
    "alcohol_units_before_bed",
    "screen_time_before_bed_mins",
    "exercise_day",
    "steps_that_day",
    "stress_score",
    "heart_rate_resting_bpm",
    "shift_work",
]

CATEGORICAL_FEATURES = [
    "gender",
    "occupation",
    "country",
    "chronotype",
    "season",
]

FEATURE_COLUMNS = [
    "age",
    "gender",
    "occupation",
    "bmi",
    "country",
    "sleep_duration_hrs",
    "sleep_quality_score",
    "caffeine_mg_before_bed",
    "alcohol_units_before_bed",
    "screen_time_before_bed_mins",
    "exercise_day",
    "steps_that_day",
    "stress_score",
    "chronotype",
    "heart_rate_resting_bpm",
    "shift_work",
    "season",
]

REPORT_METRICS = {
    "Accuracy": 0.7681,
    "Precision macro": 0.6494,
    "Recall macro": 0.6459,
    "F1 macro": 0.6475,
}

MAX_TRAINING_ROWS = 25000


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) > MAX_TRAINING_ROWS:
        df = df.sample(n=MAX_TRAINING_ROWS, random_state=42).copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision macro": precision,
        "Recall macro": recall,
        "F1 macro": f1,
    }

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "report": report,
        "feature_importance": get_feature_importance(pipeline),
    }


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_artifact(bundle: dict[str, Any], path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET_COLUMN].value_counts().rename_axis("risk_level").reset_index(name="count")
    counts["proportion"] = counts["count"] / counts["count"].sum()
    return counts
