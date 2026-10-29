
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# APPLICATION CONFIGURATION

st.set_page_config(
    page_title="CrossWalkGap | Dhaka",
    page_icon="🚸",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parent
    / "Survey-data-on-pedestrian-crosswalk-usage-behavior-in-Dhaka.xlsx"
)

BARRIER_LABELS = {
    "V1: Crosswalk usage is time-consuming": "Time-consuming",
    "V2: Absence of guard rails on median": "No median guard rails",
    "V3: Unsuitability of crosswalk location": "Poor location",
    "V4: Feel unsafe to cross directly without a crosswalk":
        "Safety motivation",
    "V5: Inadequate marking and sign to locate crosswalk":
        "Poor markings/signs",
    "V6: Poor entry access to crosswalk": "Poor entry access",
    "V7: Difficulty crossing road using crosswalk":
        "Crossing difficulty",
    "V8: Inadequate lightning at night near crosswalk":
        "Poor night lighting",
}

BARRIER_COLS = list(BARRIER_LABELS.keys())

FREQUENCY_ORDER = [
    "Very rarely",
    "Once or twice a month",
    "Once or twice a week",
    "Every day",
    "More than once a day",
]

FREQUENCY_SCORE = {
    value: index for index, value in enumerate(FREQUENCY_ORDER)
}

DEMOGRAPHIC_COLS = [
    "Gender",
    "Age (years)",
    "Do you prefer to use crosswalk",
    "How often do you use crosswalk",
]

# DATA LOADING AND VALIDATION
 

@st.cache_data
def load_data(file_path):
    """Load the Excel dataset and validate its structure."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the Excel file beside app.py."
        )

    df = pd.read_excel(path, sheet_name=0)

    # Normalize column names.
    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    required = DEMOGRAPHIC_COLS + BARRIER_COLS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing expected columns:\n"
            + "\n".join(missing)
            + "\n\nColumns found:\n"
            + "\n".join(df.columns.astype(str))
        )

    # Clean text columns.
    for col in DEMOGRAPHIC_COLS:
        df[col] = df[col].astype("string").str.strip()

    # Convert barrier responses into numeric values.
    for col in BARRIER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep recognized frequency responses.
    df = df[
        df["How often do you use crosswalk"].isin(FREQUENCY_ORDER)
    ].copy()

    # Exclude records without barrier ratings for this analysis.
    df = df.dropna(subset=BARRIER_COLS).copy()

    # Define frequent use as weekly or more frequent.
    # This is an operational definition, not an original dataset label.
    df["Frequency score"] = (
        df["How often do you use crosswalk"].map(FREQUENCY_SCORE)
    )

    df["Frequent user (weekly+)"] = (
        df["Frequency score"] >= 2
    ).astype(int)

    df["Mean barrier rating"] = df[BARRIER_COLS].mean(axis=1)

    return df

# 3. MACHINE-LEARNING MODELS

def get_models():
    """Build reproducible model pipelines."""

    return {
        "Logistic regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]),

        "Random forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


def evaluate_models(df):
    """
    Evaluate models using stratified cross-validation.

    Out-of-fold predictions are used for the reported metrics.
    """

    X = df[BARRIER_COLS]
    y = df["Frequent user (weekly+)"]

    if y.nunique() != 2:
        raise ValueError(
            "Both outcome classes are required for classification."
        )

    smallest_class = int(y.value_counts().min())
    n_splits = min(5, smallest_class)

    if n_splits < 2:
        raise ValueError(
            "Not enough observations in each class for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    models = get_models()
    results = []
    predictions = {}

    for name, model in models.items():

        # Each row is predicted by a model that excluded that row
        # from its training fold.
        pred = cross_val_predict(
            model, X, y, cv=cv, method="predict"
        )

        probabilities = cross_val_predict(
            model, X, y, cv=cv, method="predict_proba"
        )[:, 1]

        results.append({
            "Model": name,
            "Balanced accuracy": balanced_accuracy_score(y, pred),
            "Precision": precision_score(
                y, pred, zero_division=0
            ),
            "Recall": recall_score(
                y, pred, zero_division=0
            ),
            "F1": f1_score(
                y, pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y, probabilities),
        })

        predictions[name] = {
            "pred": pred,
            "prob": probabilities,
        }

    return pd.DataFrame(results), predictions, models


# VISUALIZATION HELPER

def make_radar_chart(values, title):
    """Create a radar chart of eight barrier ratings."""

    labels = [BARRIER_LABELS[col] for col in BARRIER_COLS]
    values = list(values)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Barrier ratings",
        )
    )

    fig.update_layout(
        title=title,
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 5],
            }
        },
        height=480,
        showlegend=False,
    )

    return fig


# LOAD THE DATASET

try:
    df = load_data(str(DATA_PATH))

except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


# 6. NAVIGATION

st.sidebar.title("CrossWalkGap")

st.sidebar.caption(
    "Pedestrian crosswalk survey analysis"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Barrier patterns",
        "Group comparisons",
        "Model evaluation",
        "Profile explorer",
        "Data quality and limitations",
    ],
)

st.title("CrossWalkGap: Pedestrian Crosswalk Behavior")

st.write(
    "An exploratory analytics application using survey responses "
    "about crosswalk use and perceived pedestrian barriers in Dhaka, "
    "Bangladesh. Survey associations do not establish causation."
)


# OVERVIEW DASHBOARD

if page == "Overview":

    preferred = (
        df["Do you prefer to use crosswalk"]
        .str.lower()
        .eq("yes")
        .mean()
    )

    frequent = df["Frequent user (weekly+)"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Usable responses", f"{len(df):,}")
    col2.metric("Prefer crosswalk", f"{preferred:.1%}")
    col3.metric("Weekly-or-more users", f"{frequent:.1%}")
    col4.metric(
        "Duplicate rows",
        f"{int(df.duplicated().sum()):,}",
    )

    st.subheader("Crosswalk-use frequency")

    frequency_counts = (
        df["How often do you use crosswalk"]
        .value_counts()
        .reindex(FREQUENCY_ORDER, fill_value=0)
        .rename_axis("Frequency")
        .reset_index(name="Respondents")
    )

    fig = px.bar(
        frequency_counts,
        x="Frequency",
        y="Respondents",
        category_orders={"Frequency": FREQUENCY_ORDER},
        title="Self-reported crosswalk-use frequency",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average perceived barrier ratings")

    averages = (
        df[BARRIER_COLS]
        .mean()
        .rename(index=BARRIER_LABELS)
        .sort_values()
        .rename_axis("Barrier")
        .reset_index(name="Mean rating")
    )

    fig = px.bar(
        averages,
        x="Mean rating",
        y="Barrier",
        orientation="h",
        title="Average rating for each barrier",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Check the original questionnaire to confirm the response "
        "scale before interpreting the numerical averages."
    )


#BARRIER PATTERN ANALYSIS

elif page == "Barrier patterns":

    st.subheader("Barrier ratings by reported usage")

    plot_df = df.copy()

    plot_df["Usage group"] = (
        plot_df["Frequent user (weekly+)"]
        .map({
            0: "Less than weekly",
            1: "Weekly or more",
        })
    )

    melted = plot_df.melt(
        id_vars="Usage group",
        value_vars=BARRIER_COLS,
        var_name="Barrier",
        value_name="Rating",
    )

    melted["Barrier"] = melted["Barrier"].map(BARRIER_LABELS)

    summary = (
        melted.groupby(
            ["Usage group", "Barrier"],
            as_index=False,
        )["Rating"]
        .mean()
    )

    fig = px.bar(
        summary,
        x="Rating",
        y="Barrier",
        color="Usage group",
        barmode="group",
        title="Average barriers by usage group",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True)

    st.info(
        "These are descriptive comparisons. Differences do not show "
        "that a barrier causes a change in crosswalk use."
    )


# DEMOGRAPHIC COMPARISONS

elif page == "Group comparisons":

    group_col = st.selectbox(
        "Compare respondents by",
        [
            "Gender",
            "Age (years)",
            "Do you prefer to use crosswalk",
        ],
    )

    melted = df.melt(
        id_vars=[group_col],
        value_vars=BARRIER_COLS,
        var_name="Barrier",
        value_name="Rating",
    )

    melted["Barrier"] = melted["Barrier"].map(BARRIER_LABELS)

    summary = (
        melted.groupby(
            [group_col, "Barrier"],
            dropna=False,
            as_index=False,
        )
        .agg(
            **{
                "Mean rating": ("Rating", "mean"),
                "Responses": ("Rating", "count"),
            }
        )
    )

    fig = px.bar(
        summary,
        x="Barrier",
        y="Mean rating",
        color=group_col,
        barmode="group",
        title=f"Barrier ratings by {group_col}",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True)

    st.caption(
        "Review group sizes before drawing conclusions. These "
        "comparisons are not adjusted for other variables."
    )


#MODEL EVALUATION

elif page == "Model evaluation":

    st.subheader("Cross-validated classification")

    st.write(
        "Outcome: weekly-or-more crosswalk use, derived from the "
        "reported frequency categories. Predictors: the eight "
        "barrier ratings."
    )

    try:
        results, predictions, models = evaluate_models(df)

    except Exception as error:
        st.error(f"Model evaluation failed: {error}")
        st.stop()

    st.dataframe(
        results.style.format({
            "Balanced accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}",
        }),
        use_container_width=True,
    )

    selected = st.selectbox(
        "Inspect model predictions",
        list(models.keys()),
    )

    y_true = df["Frequent user (weekly+)"].to_numpy()
    y_pred = predictions[selected]["pred"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted: less than weekly", "Predicted: weekly+"],
        y=["Actual: less than weekly", "Actual: weekly+"],
        labels={
            "x": "Predicted class",
            "y": "Actual class",
            "color": "Count",
        },
        title=f"Out-of-fold confusion matrix: {selected}",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.text(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Less than weekly",
                "Weekly or more",
            ],
            zero_division=0,
        )
    )

    st.warning(
        "These are exploratory cross-validation results. The model "
        "predicts a self-reported survey category, not pedestrian "
        "crashes or actual road safety. Model performance does not "
        "demonstrate causation or deployment readiness."
    )


# HYPOTHETICAL PROFILE EXPLORER


elif page == "Profile explorer":

    st.subheader("Explore a hypothetical pedestrian profile")

    st.write(
        "Adjust the ratings to inspect a model estimate. This is "
        "not a simulation of real-world policy effects."
    )

    values = {}
    left, right = st.columns(2)

    for index, col in enumerate(BARRIER_COLS):

        target = left if index < 4 else right

        with target:
            values[col] = st.slider(
                BARRIER_LABELS[col],
                min_value=1,
                max_value=5,
                value=3,
                key=f"barrier_{index}",
            )

    try:
        _, _, models = evaluate_models(df)

        selected = st.selectbox(
            "Select model",
            list(models.keys()),
        )

        # Fit a fresh copy on all available survey rows for this
        # interactive demonstration only.
        model = clone(models[selected])

        X = df[BARRIER_COLS]
        y = df["Frequent user (weekly+)"]

        model.fit(X, y)

        profile = pd.DataFrame(
            [values],
            columns=BARRIER_COLS,
        )

        probability = model.predict_proba(profile)[0, 1]

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated probability of weekly-or-more use",
            f"{probability:.1%}",
        )

        col2.metric(
            "Classification at 0.50 threshold",
            "Weekly or more"
            if probability >= 0.5
            else "Less than weekly",
        )

        st.plotly_chart(
            make_radar_chart(
                [values[col] for col in BARRIER_COLS],
                "Hypothetical barrier profile",
            ),
            use_container_width=True,
        )

        st.caption(
            "This model is fitted on the available dataset for "
            "demonstration. The estimated probability is model-dependent "
            "and is not an estimate of the effect of changing a barrier."
        )

    except Exception as error:
        st.error(f"Profile explorer failed: {error}")


# DATA QUALITY AND LIMITATIONS

else:

    st.subheader("Data quality checks")

    quality = pd.DataFrame({
        "Check": [
            "Rows included in analysis",
            "Exact duplicate rows",
            "Missing barrier ratings",
            "Unrecognized frequency categories",
        ],
        "Result": [
            len(df),
            int(df.duplicated().sum()),
            int(df[BARRIER_COLS].isna().sum().sum()),
            int(
                (~df["How often do you use crosswalk"]
                 .isin(FREQUENCY_ORDER)).sum()
            ),
        ],
    })

    st.dataframe(quality, use_container_width=True)

    st.subheader("Missing values by column")

    st.dataframe(
        df.isna().sum().rename("Missing values").to_frame(),
        use_container_width=True,
    )

    st.subheader("Frequency response counts")

    st.dataframe(
        df["How often do you use crosswalk"]
        .value_counts()
        .reindex(FREQUENCY_ORDER, fill_value=0)
        .rename_axis("Frequency")
        .to_frame("Respondents"),
        use_container_width=True,
    )

    st.subheader("Research limitations")

    st.markdown("""
    - Survey responses are not direct observations of crossings,
      traffic conflicts, injuries, or crashes.
    - The frequent-use outcome is derived from a self-reported category.
    - Associations cannot establish that a barrier causes behavior.
    - Representativeness depends on the original sampling procedure.
    - Confirm the original questionnaire and response scale.
    - A research paper should report uncertainty, sensitivity analyses,
      sampling limitations, and relevant prior literature.
    """)

    st.download_button(
        label="Download analyzed data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="crosswalkgap_analyzed_data.csv",
        mime="text/csv",
    )


# FOOTER
st.sidebar.divider()

st.sidebar.caption(
    "CrossWalkGap research prototype | "
    "Associations are not causal effects."
)
