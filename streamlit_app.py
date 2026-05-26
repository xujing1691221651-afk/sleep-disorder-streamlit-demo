from __future__ import annotations

import pandas as pd
import streamlit as st

from modeling import (
    ARTIFACT_DIR,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    REPORT_METRICS,
    class_distribution,
    load_artifact,
    load_dataset,
    save_artifact,
    train_model,
)


st.set_page_config(
    page_title="Sleep Disorder Risk Demo",
    layout="wide",
)


RISK_NOTES = {
    "Healthy": "Low predicted risk. The profile is closer to healthy sleep patterns in the dataset.",
    "Mild": "Some risk signals are present. Lifestyle and sleep hygiene changes may be useful.",
    "Moderate": "The profile shows several risk indicators and should be interpreted with care.",
    "Severe": "The model detects strong risk signals. This is a screening demo, not a diagnosis.",
}


def local_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background: #f7f8fb;
            border: 1px solid #e3e6ef;
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }
        .risk-note {
            border-left: 4px solid #2f6fed;
            background: #f5f8ff;
            padding: 0.9rem 1rem;
            border-radius: 6px;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner="Preparing the tuned Random Forest model...")
def cached_model() -> dict:
    if MODEL_PATH.exists():
        return load_artifact(MODEL_PATH)

    df = cached_dataset()
    bundle = train_model(df)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        save_artifact(bundle, MODEL_PATH)
    except OSError:
        pass
    return bundle


def option_values(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique().tolist())


def median_value(df: pd.DataFrame, column: str) -> float:
    return float(df[column].median())


def build_input_form(df: pd.DataFrame) -> pd.DataFrame:
    left, middle, right = st.columns(3)

    with left:
        age = st.slider("Age", 18, 80, int(median_value(df, "age")))
        gender = st.selectbox("Gender", option_values(df, "gender"))
        occupation = st.selectbox("Occupation", option_values(df, "occupation"))
        country = st.selectbox("Country", option_values(df, "country"))
        bmi = st.slider("BMI", 15.0, 45.0, round(median_value(df, "bmi"), 1), 0.1)
        heart_rate = st.slider(
            "Resting heart rate (bpm)",
            40,
            120,
            int(median_value(df, "heart_rate_resting_bpm")),
        )

    with middle:
        sleep_duration = st.slider(
            "Sleep duration (hours)",
            2.0,
            12.0,
            round(median_value(df, "sleep_duration_hrs"), 2),
            0.1,
        )
        sleep_quality = st.slider(
            "Sleep quality score",
            1.0,
            10.0,
            round(median_value(df, "sleep_quality_score"), 1),
            0.1,
        )
        stress = st.slider("Stress score", 1.0, 10.0, round(median_value(df, "stress_score"), 1), 0.1)
        screen_time = st.slider(
            "Screen time before bed (minutes)",
            0,
            240,
            int(median_value(df, "screen_time_before_bed_mins")),
        )
        chronotype = st.selectbox("Chronotype", option_values(df, "chronotype"))
        season = st.selectbox("Season", option_values(df, "season"))

    with right:
        steps = st.slider("Steps that day", 0, 30000, int(median_value(df, "steps_that_day")), 100)
        caffeine = st.slider(
            "Caffeine before bed (mg)",
            0,
            500,
            int(median_value(df, "caffeine_mg_before_bed")),
            5,
        )
        alcohol = st.slider(
            "Alcohol before bed (units)",
            0.0,
            6.0,
            round(median_value(df, "alcohol_units_before_bed"), 1),
            0.1,
        )
        exercise = st.checkbox("Exercised that day", value=bool(round(median_value(df, "exercise_day"))))
        shift_work = st.checkbox("Shift work", value=bool(round(median_value(df, "shift_work"))))

    values = {
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "bmi": bmi,
        "country": country,
        "sleep_duration_hrs": sleep_duration,
        "sleep_quality_score": sleep_quality,
        "caffeine_mg_before_bed": caffeine,
        "alcohol_units_before_bed": alcohol,
        "screen_time_before_bed_mins": screen_time,
        "exercise_day": int(exercise),
        "steps_that_day": steps,
        "stress_score": stress,
        "chronotype": chronotype,
        "heart_rate_resting_bpm": heart_rate,
        "shift_work": int(shift_work),
        "season": season,
    }

    return pd.DataFrame([values])[FEATURE_COLUMNS]


def show_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    cols = st.columns(4)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(f"{prefix}{name}", f"{value:.4f}")


def main() -> None:
    local_css()

    st.title("Sleep Disorder Risk Prediction Demo")
    st.caption("Final model used in the group report: Tuned Random Forest")

    try:
        df = cached_dataset()
    except Exception as exc:
        st.error(f"Could not load dataset from {DATA_PATH}: {exc}")
        st.stop()

    prediction_tab, evidence_tab, data_tab = st.tabs(["Prediction", "Model Evidence", "Dataset"])

    with prediction_tab:
        st.subheader("Patient-style input profile")
        input_df = build_input_form(df)

        if st.button("Predict risk level", type="primary", use_container_width=True):
            try:
                bundle = cached_model()
                model = bundle["pipeline"]
            except Exception as exc:
                st.error(f"Could not train or load model: {exc}")
                st.stop()

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = model.named_steps["model"].classes_

            result_col, chart_col = st.columns([1, 2])
            with result_col:
                st.metric("Predicted risk", prediction)
                st.markdown(
                    f'<div class="risk-note">{RISK_NOTES.get(prediction, "Prediction generated.")}</div>',
                    unsafe_allow_html=True,
                )
            with chart_col:
                probability_df = pd.DataFrame(
                    {
                        "risk_level": classes,
                        "probability": probabilities,
                    }
                ).set_index("risk_level")
                st.bar_chart(probability_df)

            st.dataframe(input_df, use_container_width=True, hide_index=True)

    with evidence_tab:
        st.subheader("Live model check")
        st.write("The app trains or loads the selected Random Forest model when evidence is requested.")

        if st.button("Load model evidence", use_container_width=True):
            try:
                bundle = cached_model()
            except Exception as exc:
                st.error(f"Could not train or load model: {exc}")
                st.stop()

            show_metrics(bundle["metrics"])

            st.subheader("Top feature importance")
            importance = bundle["feature_importance"].head(10).set_index("feature")
            st.bar_chart(importance)

            report_df = pd.DataFrame(bundle["report"]).transpose()
            st.dataframe(report_df, use_container_width=True)

        st.subheader("Reported final model metrics")
        show_metrics(REPORT_METRICS, prefix="Report ")

    with data_tab:
        st.subheader("Target distribution")
        dist = class_distribution(df)
        st.dataframe(dist, use_container_width=True, hide_index=True)
        st.bar_chart(dist.set_index("risk_level")["count"])

        st.subheader("Dataset preview")
        st.dataframe(df[FEATURE_COLUMNS + ["sleep_disorder_risk"]].head(50), use_container_width=True)


if __name__ == "__main__":
    main()
