
"""
CrossWalkGap — Interactive Data Visualization
Standalone visualization script. Does not modify app.py.

Run:
    python visualize_crosswalkgap.py

Output:
    outputs/crosswalkgap_visualization.html
"""

from pathlib import Path
import re
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

POSSIBLE_FILES = [
    BASE_DIR / "Survey data on pedestrian crosswalk usage behavior in Dhaka, Bangladesh.xlsx",
    BASE_DIR / "Survey-data-on-pedestrian-crosswalk-usage-behavior-in-Dhaka.xlsx",
]

FREQUENCY_ORDER = [
    "Very rarely",
    "Once or twice a month",
    "Once or twice a week",
    "Every day",
    "More than once a day",
]

BARRIER_LABELS = {
    "V1": "Time-consuming",
    "V2": "No median guardrails",
    "V3": "Poor crosswalk location",
    "V4": "Feels unsafe without crosswalk",
    "V5": "Poor markings and signs",
    "V6": "Poor entry access",
    "V7": "Difficulty crossing",
    "V8": "Poor nighttime lighting",
}

CHART_TEMPLATE = "plotly_white"


# ---------------------------------------------------------
# 2. Load and validate the survey
# ---------------------------------------------------------

def normalize_column_name(name):
    """Normalize spacing and capitalization for matching."""
    return re.sub(r"\s+", " ", str(name).strip()).casefold()


def find_input_file():
    for path in POSSIBLE_FILES:
        if path.exists():
            return path

    # Also search for an Excel file in the project directory.
    candidates = list(BASE_DIR.glob("*.xlsx"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Survey Excel file not found. Place the workbook beside "
        "visualize_crosswalkgap.py and update POSSIBLE_FILES if needed."
    )


def load_data():
    path = find_input_file()
    df = pd.read_excel(path)

    # Normalize column names without changing the underlying data.
    df.columns = [
        re.sub(r"\s+", " ", str(col).strip())
        for col in df.columns
    ]

    # Identify the eight barrier columns using their V1 ... V8 prefixes.
    barrier_columns = {}
    for col in df.columns:
        match = re.match(r"^(V[1-8])\b", col, flags=re.IGNORECASE)
        if match:
            barrier_columns[match.group(1).upper()] = col

    missing_barriers = [
        key for key in BARRIER_LABELS if key not in barrier_columns
    ]

    required = [
        "Gender",
        "Age (years)",
        "Do you prefer to use crosswalk",
        "How often do you use crosswalk",
    ]

    normalized_lookup = {
        normalize_column_name(col): col for col in df.columns
    }

    resolved = {}
    for name in required:
        actual = normalized_lookup.get(normalize_column_name(name))
        if actual is not None:
            resolved[name] = actual

    missing_required = [name for name in required if name not in resolved]

    if missing_required or missing_barriers:
        print("\nCould not match all expected columns.")
        print("\nColumns found in the workbook:")
        for col in df.columns:
            print(f"  - {col}")

        if missing_required:
            print("\nMissing required columns:", missing_required)

        if missing_barriers:
            print("\nMissing barrier prefixes:", missing_barriers)

        raise ValueError(
            "Please update the column mapping in load_data() "
            "to match the actual Excel workbook."
        )

    # Rename only the required demographic/outcome columns internally.
    df = df.rename(
        columns={
            resolved["Gender"]: "Gender",
            resolved["Age (years)"]: "Age",
            resolved["Do you prefer to use crosswalk"]: "Preference",
            resolved["How often do you use crosswalk"]: "Frequency",
        }
    )

    for key, actual_column in barrier_columns.items():
        df[actual_column] = pd.to_numeric(
            df[actual_column], errors="coerce"
        )

    return df, barrier_columns, path


# ---------------------------------------------------------
# 3. Common chart styling
# ---------------------------------------------------------

def style_chart(fig, title, height=480):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=35, r=30, t=85, b=45),
        font=dict(family="Arial, sans-serif", size=13),
        hoverlabel=dict(namelength=-1),
        legend_title_text="",
    )
    return fig


# ---------------------------------------------------------
# 4. Create individual visualizations
# ---------------------------------------------------------

def make_frequency_chart(df):
    counts = (
        df["Frequency"]
        .astype(str)
        .value_counts()
        .rename_axis("Frequency")
        .reset_index(name="Responses")
    )

    counts["Frequency"] = pd.Categorical(
        counts["Frequency"],
        categories=FREQUENCY_ORDER,
        ordered=True,
    )
    counts = counts.sort_values("Frequency")

    fig = px.bar(
        counts,
        x="Frequency",
        y="Responses",
        text="Responses",
        color="Responses",
        color_continuous_scale="Teal",
        title="How often do respondents use crosswalks?",
        labels={"Frequency": "Reported usage frequency"},
    )

    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=-20)
    return style_chart(fig, "Crosswalk usage frequency")


def make_barrier_chart(df, barrier_columns):
    records = []

    for key, column in barrier_columns.items():
        scores = pd.to_numeric(df[column], errors="coerce").dropna()

        records.append({
            "Barrier": f"{key} · {BARRIER_LABELS[key]}",
            "Mean score": scores.mean(),
            "Responses": len(scores),
            "SD": scores.std(),
        })

    summary = pd.DataFrame(records).sort_values("Mean score")

    fig = px.bar(
        summary,
        x="Mean score",
        y="Barrier",
        orientation="h",
        error_x="SD",
        color="Mean score",
        color_continuous_scale="Teal",
        hover_data=["Responses"],
        labels={"Mean score": "Average rating (1–5)"},
    )

    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(range=[0, 5])
    return style_chart(
        fig,
        "Which perceived barriers receive the highest ratings?",
        height=560,
    )


def make_heatmap(df, barrier_columns):
    working = df.copy()

    # Use only rows with a known frequency category.
    working["Frequency"] = working["Frequency"].astype(str).str.strip()
    working = working[
        working["Frequency"].isin(FREQUENCY_ORDER)
    ]

    rows = []
    for frequency in FREQUENCY_ORDER:
        group = working[working["Frequency"] == frequency]

        if group.empty:
            continue

        for key, column in barrier_columns.items():
            values = pd.to_numeric(group[column], errors="coerce")

            rows.append({
                "Frequency": frequency,
                "Barrier": f"{key} · {BARRIER_LABELS[key]}",
                "Mean rating": values.mean(),
                "Responses": int(values.notna().sum()),
            })

    summary = pd.DataFrame(rows)

    if summary.empty:
        return style_chart(
            go.Figure(),
            "Barrier ratings by reported usage frequency",
        )

    pivot = summary.pivot(
        index="Barrier",
        columns="Frequency",
        values="Mean rating",
    )

    # Preserve the questionnaire's intended frequency ordering.
    ordered_columns = [
        category for category in FREQUENCY_ORDER
        if category in pivot.columns
    ]
    pivot = pivot[ordered_columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="YlGnBu",
            zmin=1,
            zmax=5,
            colorbar=dict(title="Mean<br>rating"),
            hovertemplate=(
                "Frequency: %{x}<br>"
                "Barrier: %{y}<br>"
                "Mean rating: %{z:.2f}<extra></extra>"
            ),
        )
    )

    return style_chart(
        fig,
        "Barrier ratings across reported usage-frequency groups",
        height=560,
    )


def make_demographic_chart(df, barrier_columns, group_column, title):
    work = df.copy()
    work[group_column] = work[group_column].fillna("Not reported").astype(str)

    # Convert the barrier ratings to long format for plotting.
    long_df = work.melt(
        id_vars=[group_column],
        value_vars=list(barrier_columns.values()),
        var_name="Original barrier",
        value_name="Rating",
    )

    reverse_labels = {
        col: f"{key} · {BARRIER_LABELS[key]}"
        for key, col in barrier_columns.items()
    }
    long_df["Barrier"] = long_df["Original barrier"].map(reverse_labels)
    long_df["Rating"] = pd.to_numeric(long_df["Rating"], errors="coerce")

    summary = (
        long_df.groupby([group_column, "Barrier"], observed=True)
        .agg(
            Mean_rating=("Rating", "mean"),
            Responses=("Rating", "count"),
        )
        .reset_index()
    )

    fig = px.bar(
        summary,
        x="Barrier",
        y="Mean_rating",
        color=group_column,
        barmode="group",
        hover_data=["Responses"],
        labels={
            "Mean_rating": "Average rating (1–5)",
            group_column: group_column,
        },
    )

    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(range=[0, 5])
    return style_chart(fig, title, height=600)


def make_correlation_chart(df, barrier_columns):
    barrier_df = df[list(barrier_columns.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    barrier_df = barrier_df.rename(
        columns={
            col: f"{key} · {BARRIER_LABELS[key]}"
            for key, col in barrier_columns.items()
        }
    )

    corr = barrier_df.corr(method="spearman", min_periods=3)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="Spearman<br>correlation"),
            text=corr.round(2).values,
            texttemplate="%{text}",
            hovertemplate=(
                "%{y}<br>with %{x}<br>"
                "Correlation: %{z:.2f}<extra></extra>"
            ),
        )
    )

    return style_chart(
        fig,
        "Relationships between perceived barrier ratings",
        height=650,
    )


def make_preference_chart(df):
    values = (
        df["Preference"]
        .fillna("Not reported")
        .astype(str)
        .value_counts()
        .rename_axis("Preference")
        .reset_index(name="Responses")
    )

    fig = px.pie(
        values,
        names="Preference",
        values="Responses",
        hole=0.58,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>Responses: %{value}<extra></extra>",
    )

    return style_chart(fig, "Do respondents prefer to use crosswalks?")


# ---------------------------------------------------------
# 5. Assemble a standalone HTML report
# ---------------------------------------------------------

def create_report(df, barrier_columns, source_path):
    charts = [
        make_frequency_chart(df),
        make_barrier_chart(df, barrier_columns),
        make_heatmap(df, barrier_columns),
        make_demographic_chart(
            df, barrier_columns, "Gender",
            "Average barrier ratings by gender",
        ),
        make_demographic_chart(
            df, barrier_columns, "Age",
            "Average barrier ratings by age group",
        ),
        make_correlation_chart(df, barrier_columns),
        make_preference_chart(df),
    ]

    response_count = len(df)
    preference_yes = (
        df["Preference"].astype(str).str.strip().str.casefold() == "yes"
    ).sum()

    # Include only clearly computed descriptive statistics.
    barrier_values = df[list(barrier_columns.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    mean_rating = barrier_values.stack().mean()

    # Embed Plotly.js once so the report works offline after creation.
    chart_html = []
    for i, fig in enumerate(charts):
        chart_html.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs="cdn" if i == 0 else False,
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": f"crosswalkgap_chart_{i + 1}",
                        "scale": 2,
                    },
                },
            )
        )

    cards = f"""
    <section class="metrics">
      <article class="metric">
        <span class="metric-label">Survey responses</span>
        <strong>{response_count:,}</strong>
      </article>
      <article class="metric">
        <span class="metric-label">Prefer crosswalks</span>
        <strong>{preference_yes:,}</strong>
        <small>{preference_yes / response_count:.1%} of all responses</small>
      </article>
      <article class="metric">
        <span class="metric-label">Barrier items</span>
        <strong>{len(barrier_columns)}</strong>
      </article>
      <article class="metric">
        <span class="metric-label">Overall mean barrier rating</span>
        <strong>{mean_rating:.2f}<small> / 5</small></strong>
      </article>
    </section>
    """

    sections = [
        ("01", "Reported usage frequency", chart_html[0]),
        ("02", "Perceived barrier ranking", chart_html[1]),
        ("03", "Barrier ratings by frequency", chart_html[2]),
        ("04", "Gender comparison", chart_html[3]),
        ("05", "Age-group comparison", chart_html[4]),
        ("06", "Barrier relationship matrix", chart_html[5]),
        ("07", "Crosswalk preference", chart_html[6]),
    ]

    section_html = "\n".join(
        f"""
        <section class="chart-card">
          <div class="chart-heading">
            <span class="chart-number">{number}</span>
            <h2>{title}</h2>
          </div>
          {chart}
        </section>
        """
        for number, title, chart in sections
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CrossWalkGap | Dhaka Pedestrian Survey</title>
  <style>
    :root {{
      --ink: #172b3a;
      --muted: #617486;
      --surface: #ffffff;
      --background: #f2f6f8;
      --accent: #0f766e;
      --line: #dce5e9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--background);
      color: var(--ink);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }}
    .hero {{
      padding: 48px max(24px, calc((100% - 1240px) / 2));
      color: white;
      background: linear-gradient(120deg, #102a43, #0f766e);
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: .15em;
      font-size: 12px;
      font-weight: 700;
      opacity: .85;
    }}
    h1 {{
      margin: 12px 0;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1.1;
    }}
    .hero p {{
      max-width: 720px;
      line-height: 1.7;
      color: #e1f3f1;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 22px 60px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 26px;
    }}
    .metric, .chart-card, .note {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 4px 18px rgba(18, 45, 60, .04);
    }}
    .metric {{
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .metric-label {{
      font-size: 13px;
      color: var(--muted);
    }}
    .metric strong {{
      font-size: 30px;
      color: var(--accent);
    }}
    .metric small {{
      font-size: 13px;
      color: var(--muted);
    }}
    .chart-card {{
      padding: 12px 16px 18px;
      margin: 20px 0;
      overflow: hidden;
    }}
    .chart-heading {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 12px 0;
    }}
    .chart-heading h2 {{
      font-size: 19px;
      margin: 0;
    }}
    .chart-number {{
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 10px;
      background: #d8f3ee;
      color: #0f766e;
      font-weight: 800;
      font-size: 12px;
    }}
    .note {{
      padding: 22px 26px;
      line-height: 1.7;
      color: var(--muted);
    }}
    .note strong {{ color: var(--ink); }}
    footer {{
      padding: 24px;
      text-align: center;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 800px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .hero {{ padding: 34px 22px; }}
    }}
    @media (max-width: 450px) {{
      .metrics {{ grid-template-columns: 1fr 1fr; gap: 8px; }}
      .metric {{ padding: 14px; }}
      .metric strong {{ font-size: 24px; }}
      main {{ padding: 16px 10px 35px; }}
      .chart-card {{ padding: 8px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="eyebrow">Research data explorer · Dhaka, Bangladesh</div>
    <h1>CrossWalkGap</h1>
    <p>
      An interactive exploration of pedestrian crosswalk preferences,
      self-reported usage frequency, and perceived barriers to crosswalk use.
    </p>
  </header>
  <main>
    {cards}
    {section_html}
    <section class="note">
      <strong>How to interpret this report</strong><br>
      These charts describe survey responses. Barrier ratings and reported
      usage frequency are observational and self-reported. Group differences
      and correlations do not establish causation or measure actual crash risk.
      Check the questionnaire, sampling method, category coding, and small
      subgroup counts before making policy or publication claims.
      <br><br>
      <strong>Source workbook:</strong> {source_path.name}
    </section>
  </main>
  <footer>
    CrossWalkGap · Standalone visualization · Generated from the supplied survey
  </footer>
</body>
</html>
"""

    output_path = OUTPUT_DIR / "crosswalkgap_visualization.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"\nLoaded workbook: {source_path}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    print(f"Interactive report saved to: {output_path}")
    print("Open the HTML file in your browser to explore the charts.")


# ---------------------------------------------------------
# 6. Run
# ---------------------------------------------------------

def main():
    try:
        df, barrier_columns, source_path = load_data()
        create_report(df, barrier_columns, source_path)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
