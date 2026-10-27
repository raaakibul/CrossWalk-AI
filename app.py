import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="CrossWalk AI — Dhaka",
    page_icon="🚸",
    layout="wide"
)

BARRIER_COLS = [
    "V1 Crosswalk usage is time-consuming",
    "V2 Absence of guard rails on median",
    "V3 Unsuitability of crosswalk location",
    "V4 Feel unsafe to cross directly without a crosswalk",
    "V5 Inadequate marking and sign to locate crosswalk",
    "V6 Poor entry access to crosswalk",
    "V7 Difficulty crossing road using crosswalk",
    "V8 Inadequate lightning at night near crosswalk",
]

BARRIER_LABELS = {
    "V1 Crosswalk usage is time-consuming": "Time-consuming",
    "V2 Absence of guard rails on median": "No median guard rails",
    "V3 Unsuitability of crosswalk location": "Poor location",
    "V4 Feel unsafe to cross directly without a crosswalk": "Safety motivation",
    "V5 Inadequate marking and sign to locate crosswalk": "Poor markings/signs",
    "V6 Poor entry access to crosswalk": "Poor entry access",
    "V7 Difficulty crossing road using crosswalk": "Crossing difficulty",
    "V8 Inadequate lightning at night near crosswalk": "Poor night lighting",
}

SHORT_BARRIERS = [BARRIER_LABELS[col] for col in BARRIER_COLS]

REQUIRED_COLS = [
    "Gender",
    "Age years",
    "Do you prefer to use crosswalk",
    "How often do you use crosswalk",
] + BARRIER_COLS


@st.cache_data
def load_data():
    file = os.path.join(
        os.path.dirname(__file__),
        "Survey-data-on-pedestrian-crosswalk-usage-behavior-in-Dhaka.xlsx"
    )

    if not os.path.exists(file):
        st.error(f"Excel file not found:\n\n{file}")
        st.stop()

    raw_df = pd.read_excel(file, sheet_name=0, header=None)

    header_row = None

    for i in range(min(20, len(raw_df))):
        row_values = raw_df.iloc[i].astype(str).str.strip().tolist()

        if "Gender" in row_values:
            header_row = i
            break

    if header_row is None:
        st.error("Could not find the header row containing 'Gender'.")
        st.write(raw_df.head(10))
        st.stop()

    df = pd.read_excel(file, sheet_name=0, header=header_row)

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df[df["Gender"].astype(str).str.strip().str.lower() != "gender"]

    missing = [col for col in REQUIRED_COLS if col not in df.columns]

    if missing:
        st.error("These required columns were not found:\n\n" + "\n".join(missing))
        st.write("Columns pandas actually found:")
        st.write(list(df.columns))
        st.stop()

    for col in [
        "Gender",
        "Age years",
        "Do you prefer to use crosswalk",
        "How often do you use crosswalk"
    ]:
        df[col] = df[col].astype(str).str.strip()

    for col in BARRIER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=BARRIER_COLS)

    barrier_mean = df[
        [
            BARRIER_COLS[0],
            BARRIER_COLS[1],
            BARRIER_COLS[2],
            BARRIER_COLS[4],
            BARRIER_COLS[5],
            BARRIER_COLS[6],
            BARRIER_COLS[7],
        ]
    ].mean(axis=1)

    df["Crosswalk Barrier Index"] = barrier_mean - df[BARRIER_COLS[3]]

    frequency_map = {
        "Very rarely": 0,
        "Once or twice a month": 1,
        "Once or twice a week": 2,
        "Every day": 3,
        "More than once a day": 4,
    }

    df["Usage Score"] = df["How often do you use crosswalk"].map(frequency_map)
    df["Frequent User"] = (df["Usage Score"] >= 2).astype(int)

    return df


@st.cache_resource
def train_models(df):
    X = df[BARRIER_COLS].copy()
    y = df["Frequent User"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Persona Cluster"] = kmeans.fit_predict(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    importance = pd.DataFrame({
        "Barrier": SHORT_BARRIERS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return (
        df,
        kmeans,
        model,
        scaler,
        accuracy,
        classification_report(y_test, y_pred, output_dict=True),
        confusion_matrix(y_test, y_pred),
        importance
    )


def persona_name(row):
    if row["Crosswalk Barrier Index"] >= 1.5:
        return "High-Barrier Avoiders"
    elif row[BARRIER_COLS[3]] >= 4:
        return "Safety-Motivated Users"
    elif row[BARRIER_COLS[7]] >= 4:
        return "Night-Risk Sensitive Pedestrians"
    elif row[BARRIER_COLS[0]] >= 4:
        return "Time-Pressured Pedestrians"
    else:
        return "Convenience-Oriented Users"


def radar_chart(values, title):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=SHORT_BARRIERS,
        fill="toself",
        name="Perception score"
    ))

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(range=[0, 5])),
        height=500
    )

    return fig


df, kmeans, model, scaler, accuracy, report, cm, importance = train_models(load_data())

df["Persona"] = df.apply(persona_name, axis=1)

st.sidebar.title("🚸 CrossWalk AI")
st.sidebar.info(
    "An explainable pedestrian-safety analytics tool for crosswalk "
    "usage behavior in Dhaka, Bangladesh."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Barrier Index",
        "Pedestrian Personas",
        "Prediction Model",
        "Policy Simulator"
    ]
)

if page == "Overview":
    st.title("🚸 CrossWalk AI: Pedestrian Crossing Behavior in Dhaka")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Survey responses", len(df))
    c2.metric("Prefer crosswalk", f"{(df['Do you prefer to use crosswalk'] == 'Yes').mean():.1%}")
    c3.metric("Frequent users", f"{df['Frequent User'].mean():.1%}")
    c4.metric("Model accuracy", f"{accuracy:.1%}")

    st.subheader("Crosswalk usage frequency")
    freq_counts = df["How often do you use crosswalk"].value_counts().reset_index()
    freq_counts.columns = ["Frequency", "Respondents"]

    fig = px.bar(
        freq_counts,
        x="Frequency",
        y="Respondents",
        color="Frequency",
        title="How often respondents use crosswalks"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average barrier perception")
    avg_scores = df[BARRIER_COLS].mean().reset_index()
    avg_scores.columns = ["Barrier", "Average score"]
    avg_scores["Barrier"] = avg_scores["Barrier"].map(BARRIER_LABELS)

    fig = px.bar(
        avg_scores.sort_values("Average score"),
        x="Average score",
        y="Barrier",
        orientation="h",
        title="Which barriers are perceived as most severe?",
        color="Average score",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Barrier Index":
    st.title("📊 Crosswalk Barrier Index")

    fig = px.histogram(
        df,
        x="Crosswalk Barrier Index",
        color="Do you prefer to use crosswalk",
        barmode="overlay",
        nbins=30,
        title="Barrier Index by crosswalk preference"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Barrier profile by gender")
    gender_avg = df.groupby("Gender")[BARRIER_COLS].mean().reset_index()

    gender_melted = gender_avg.melt(
        id_vars="Gender",
        var_name="Barrier",
        value_name="Average score"
    )
    gender_melted["Barrier"] = gender_melted["Barrier"].map(BARRIER_LABELS)

    fig = px.bar(
        gender_melted,
        x="Barrier",
        y="Average score",
        color="Gender",
        barmode="group",
        title="Average barrier scores by gender"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Barrier profile by age group")
    age_avg = df.groupby("Age years")[BARRIER_COLS].mean().reset_index()

    age_melted = age_avg.melt(
        id_vars="Age years",
        var_name="Barrier",
        value_name="Average score"
    )
    age_melted["Barrier"] = age_melted["Barrier"].map(BARRIER_LABELS)

    fig = px.bar(
        age_melted,
        x="Barrier",
        y="Average score",
        color="Age years",
        barmode="group",
        title="Average barrier scores by age group"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Pedestrian Personas":
    st.title("👥 Pedestrian Personas")

    persona_counts = df["Persona"].value_counts().reset_index()
    persona_counts.columns = ["Persona", "Respondents"]

    fig = px.pie(
        persona_counts,
        names="Persona",
        values="Respondents",
        hole=0.45,
        title="Distribution of pedestrian personas"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Persona profiles")
    persona_profile = df.groupby("Persona")[
        BARRIER_COLS + ["Crosswalk Barrier Index", "Frequent User"]
    ].mean().reset_index()

    st.dataframe(
        persona_profile.rename(columns=BARRIER_LABELS),
        use_container_width=True
    )

    selected_persona = st.selectbox(
        "Select a persona",
        df["Persona"].unique()
    )

    persona_data = df[df["Persona"] == selected_persona]
    avg_values = persona_data[BARRIER_COLS].mean().tolist()

    st.plotly_chart(
        radar_chart(
            avg_values,
            f"Barrier profile: {selected_persona}"
        ),
        use_container_width=True
    )

elif page == "Prediction Model":
    st.title("🤖 Explainable Prediction Model")
    st.metric("Test accuracy", f"{accuracy:.1%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature importance")
        fig = px.bar(
            importance,
            x="Importance",
            y="Barrier",
            orientation="h",
            title="Which barriers most affect frequent use?"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confusion matrix")
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["Not frequent", "Frequent"],
            y=["Not frequent", "Frequent"],
            title="Model performance"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

else:
    st.title("🛠️ Policy Simulator")

    col1, col2 = st.columns(2)

    with col1:
        time = st.slider("Crosswalk use is time-consuming", 1, 5, 3)
        rails = st.slider("Absence of median guard rails", 1, 5, 3)
        location = st.slider("Crosswalk location is unsuitable", 1, 5, 3)
        safety = st.slider("Feels unsafe crossing without crosswalk", 1, 5, 4)

    with col2:
        markings = st.slider("Inadequate markings and signs", 1, 5, 3)
        access = st.slider("Poor entry access", 1, 5, 3)
        difficulty = st.slider("Difficulty crossing using crosswalk", 1, 5, 3)
        lighting = st.slider("Inadequate night lighting", 1, 5, 3)

    user_input = pd.DataFrame([[
        time, rails, location, safety, markings, access, difficulty, lighting
    ]], columns=BARRIER_COLS)

    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    barrier_index = (
        np.mean([time, rails, location, markings, access, difficulty, lighting])
        - safety
    )

    st.subheader("Simulation result")

    if prediction == 1:
        st.success(
            f"✅ Predicted: **Frequent crosswalk user** "
            f"with {probability:.1%} probability."
        )
    else:
        st.warning(
            f"⚠️ Predicted: **Not a frequent crosswalk user** "
            f"with {probability:.1%} probability of frequent use."
        )

    c1, c2 = st.columns(2)
    c1.metric("Crosswalk Barrier Index", f"{barrier_index:.2f}")
    c2.metric("Probability of frequent use", f"{probability:.1%}")

    st.plotly_chart(
        radar_chart(
            user_input.values[0].tolist(),
            "Your simulated pedestrian profile"
        ),
        use_container_width=True
    )