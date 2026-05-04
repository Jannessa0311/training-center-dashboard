import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


CSV_FILE = Path("Training Center Survey-Grid view.csv")

COLUMN_RENAMES = {
    "Id": "id",
    "Do you feel that you retained the information you received": "retained",
    "Name": "respondent_name",
    "Date": "date",
    "Name of your company": "company",
    "Who was your trainer?": "trainer",
    "What Product did you train on?": "product",
    'If "No", can you pelase give a brief description of why?': "retention_reason",
    "How would you rate your trainer": "trainer_rating",
    "How would you rate the trainings you received and the changes?": "training_rating",
    "Did you feel confident to take the safety quiz at the end of training?": "quiz_confidence",
    "What could we do different or add to make it better? Remember we are always building and looking for ways to enhance the training so please feel free to answer?": "improvement_feedback",
    "Overall, how was the training for you and do you feel the changes are making an impact?": "overall_feedback",
}

TEXT_COLUMNS = [
    "respondent_name",
    "company",
    "trainer",
    "product",
    "retention_reason",
    "quiz_confidence",
    "improvement_feedback",
    "overall_feedback",
]

EXCLUDED_TRAINER_PATTERNS = ["david", "unknown"]

THEME_KEYWORDS = {
    "Hands-on / Practical Training": [
        "hands-on",
        "hands on",
        "practice",
        "physical",
        "actual equipment",
        "troubleshoot",
        "troubleshooting",
        "component",
        "replacement",
        "demo",
        "field experience",
        "inverter",
    ],
    "Training Time / Pace": [
        "more time",
        "longer",
        "rushed",
        "pace",
        "too fast",
        "slow down",
    ],
    "Materials / Manuals": [
        "manual",
        "hard copy",
        "print",
        "printed",
        "schematic",
        "slides",
        "documents",
        "material",
        "reference",
    ],
    "Quiz / Test Alignment": [
        "quiz",
        "test",
        "questions",
        "exam",
        "safety quiz",
    ],
    "Class Level / Grouping": [
        "experience level",
        "different classes",
        "advanced",
        "beginner",
        "class size",
        "group",
        "level",
    ],
    "Audio / Communication Support": [
        "microphone",
        "audio",
        "translator",
        "translation",
        "language",
    ],
}

ACTION_RECOMMENDATIONS = {
    "Hands-on / Practical Training": "Add equipment walkthroughs, component replacement demos, troubleshooting simulations, and practical field scenarios.",
    "Training Time / Pace": "Review class duration and pacing, and consider splitting dense content into additional sessions.",
    "Materials / Manuals": "Provide digital or printed manuals, schematics, and post-training reference materials.",
    "Quiz / Test Alignment": "Align quiz questions with covered training content and add quiz preparation checkpoints.",
    "Class Level / Grouping": "Segment classes by experience level so beginner and advanced trainees receive the right depth.",
    "Audio / Communication Support": "Improve microphone, audio, and translation support for clearer instruction.",
}


st.set_page_config(
    page_title="Trainer Improvement Dashboard",
    layout="wide",
)


def normalize_column_name(column_name: str) -> str:
    """Create a safe fallback name for unexpected source columns."""
    column_name = column_name.strip().lower()
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    return column_name.strip("_")


def yes_no_to_flag(value) -> float:
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().lower()
    if normalized.startswith("yes"):
        return 1
    if normalized.startswith("no"):
        return 0
    if normalized in {"somewhat", "partial", "partially", "maybe"}:
        return 0.5
    return pd.NA


def to_numeric_rating(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def classify_feedback(comment: str) -> list[str]:
    normalized = str(comment or "").strip().lower()
    no_comment_values = {"", "nan", "none", "n/a", "na", "no", "nothing", "no comment"}

    if normalized in no_comment_values:
        return ["No Comment"]

    matched_themes = [
        theme
        for theme, keywords in THEME_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]

    return matched_themes or ["General Positive / Other"]


@st.cache_data
def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={column: COLUMN_RENAMES.get(column, normalize_column_name(column)) for column in df.columns})

    required_defaults = {
        "date": pd.NaT,
        "trainer": "Unknown Trainer",
        "product": "Unknown Product",
        "company": "Unknown Company",
        "retained": pd.NA,
        "quiz_confidence": pd.NA,
        "trainer_rating": pd.NA,
        "training_rating": pd.NA,
        "improvement_feedback": "",
        "overall_feedback": "",
        "retention_reason": "",
    }
    for column, default_value in required_defaults.items():
        if column not in df.columns:
            df[column] = default_value

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].isna() | (df["date"] >= pd.Timestamp("2024-01-01"))].copy()

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    df["trainer"] = df["trainer"].replace("", "Unknown Trainer")
    df["product"] = df["product"].replace("", "Unknown Product")
    df["company"] = df["company"].replace("", "Unknown Company")
    df["improvement_feedback"] = df["improvement_feedback"].replace("", "No comment")
    df["overall_feedback"] = df["overall_feedback"].replace("", "No comment")

    excluded_trainers = df["trainer"].str.lower().str.contains(
        "|".join(EXCLUDED_TRAINER_PATTERNS),
        na=True,
    )
    df = df[~excluded_trainers].copy()

    df["trainer_rating"] = to_numeric_rating(df["trainer_rating"])
    df["training_rating"] = to_numeric_rating(df["training_rating"])
    df["retention_flag"] = df["retained"].apply(yes_no_to_flag).astype("Float64")
    df["quiz_confidence_flag"] = df["quiz_confidence"].apply(yes_no_to_flag).astype("Float64")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df.loc[df["date"].isna(), "month"] = "Unknown Date"
    df["feedback_themes"] = df["improvement_feedback"].apply(classify_feedback)

    return df


def format_percent(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.0%}"


def format_number(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def kpi_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def build_trainer_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("trainer", dropna=False)
        .agg(
            response_count=("trainer", "size"),
            avg_trainer_rating=("trainer_rating", "mean"),
            avg_training_rating=("training_rating", "mean"),
            retention_rate=("retention_flag", "mean"),
            quiz_confidence_rate=("quiz_confidence_flag", "mean"),
        )
        .reset_index()
        .sort_values(["response_count", "avg_trainer_rating"], ascending=[False, False])
    )
    return summary


def display_summary_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("No trainer summary is available for the current filters.")
        return

    table = summary.rename(
        columns={
            "trainer": "Trainer",
            "response_count": "Response Count",
            "avg_trainer_rating": "Average Trainer Rating",
            "avg_training_rating": "Average Training Rating",
            "retention_rate": "Retention Rate",
            "quiz_confidence_rate": "Quiz Confidence Rate",
        }
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Average Trainer Rating": st.column_config.NumberColumn(format="%.2f"),
            "Average Training Rating": st.column_config.NumberColumn(format="%.2f"),
            "Retention Rate": st.column_config.NumberColumn(format="%.0%%"),
            "Quiz Confidence Rate": st.column_config.NumberColumn(format="%.0%%"),
        },
    )


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, labels: dict, text_auto: str | bool = ".2f") -> None:
    if df.empty or y not in df.columns:
        st.info(f"No data available for {title.lower()}.")
        return

    chart = px.bar(df, x=x, y=y, title=title, text_auto=text_auto, labels=labels)
    chart.update_layout(title_x=0, yaxis_title=labels.get(y, y), xaxis_title=labels.get(x, x))
    st.plotly_chart(chart, use_container_width=True)


def get_theme_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["trainer", "improvement_feedback", "feedback_theme"])

    theme_rows = df[["trainer", "improvement_feedback", "feedback_themes"]].explode("feedback_themes")
    return theme_rows.rename(columns={"feedback_themes": "feedback_theme"})


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Dashboard Filters")

    trainers = sorted(df["trainer"].dropna().unique())
    products = sorted(df["product"].dropna().unique())
    companies = sorted(df["company"].dropna().unique())

    selected_trainers = st.sidebar.multiselect("Trainer", trainers, default=trainers)
    selected_products = st.sidebar.multiselect("Product", products, default=products)
    selected_companies = st.sidebar.multiselect("Company", companies, default=companies)

    dated_df = df.dropna(subset=["date"])
    if dated_df.empty:
        selected_date_range = None
        st.sidebar.info("No valid dates are available for date filtering.")
    else:
        min_date = dated_df["date"].min().date()
        max_date = dated_df["date"].max().date()
        selected_date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    filtered_df = df[
        df["trainer"].isin(selected_trainers)
        & df["product"].isin(selected_products)
        & df["company"].isin(selected_companies)
    ].copy()

    if selected_date_range and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        date_mask = filtered_df["date"].isna() | (
            filtered_df["date"].dt.date.between(start_date, end_date)
        )
        filtered_df = filtered_df[date_mask].copy()

    st.sidebar.caption(f"{len(filtered_df):,} of {len(df):,} responses selected")
    return filtered_df


def show_executive_overview(df: pd.DataFrame) -> None:
    st.header("Executive Overview")
    st.write(
        "This dashboard converts raw customer survey feedback into trainer-level performance tracking, "
        "trend visibility, and practical program improvement insights for leadership review."
    )

    kpi_columns = st.columns(5)
    with kpi_columns[0]:
        kpi_card("Total Survey Responses", f"{len(df):,}")
    with kpi_columns[1]:
        kpi_card("Average Trainer Rating", format_number(df["trainer_rating"].mean()))
    with kpi_columns[2]:
        kpi_card("Average Training Rating", format_number(df["training_rating"].mean()))
    with kpi_columns[3]:
        kpi_card("Retention Rate", format_percent(df["retention_flag"].mean()))
    with kpi_columns[4]:
        kpi_card("Safety Quiz Confidence Rate", format_percent(df["quiz_confidence_flag"].mean()))

    st.subheader("Trainer Summary")
    display_summary_table(build_trainer_summary(df))


def show_trainer_performance(df: pd.DataFrame) -> None:
    st.header("Trainer Performance")
    st.write("Trainer results are shown with response count so leaders can weigh performance against sample size.")

    summary = build_trainer_summary(df)
    if summary.empty:
        st.info("No trainer performance data is available for the current filters.")
        return

    labels = {
        "trainer": "Trainer",
        "avg_trainer_rating": "Average Trainer Rating",
        "avg_training_rating": "Average Training Rating",
        "retention_rate": "Retention Rate",
        "quiz_confidence_rate": "Safety Quiz Confidence Rate",
        "response_count": "Survey Response Count",
    }

    col1, col2 = st.columns(2)
    with col1:
        bar_chart(summary, "trainer", "avg_trainer_rating", "Average Trainer Rating by Trainer", labels)
        bar_chart(summary, "trainer", "retention_rate", "Retention Rate by Trainer", labels, text_auto=".0%")
        bar_chart(summary, "trainer", "response_count", "Survey Response Count by Trainer", labels, text_auto=True)
    with col2:
        bar_chart(summary, "trainer", "avg_training_rating", "Average Training Rating by Trainer", labels)
        bar_chart(summary, "trainer", "quiz_confidence_rate", "Safety Quiz Confidence Rate by Trainer", labels, text_auto=".0%")


def show_trainer_evolution(df: pd.DataFrame) -> None:
    st.header("Trainer Evolution Over Time")
    st.write(
        "Monthly trends can be used to evaluate whether training changes led to better survey outcomes over time."
    )

    valid_trend_df = df[df["month"] != "Unknown Date"].copy()
    if valid_trend_df.empty:
        st.info("No valid dated responses are available for trend analysis.")
        return

    metric_options = {
        "Average Trainer Rating": ("trainer_rating", "mean"),
        "Average Training Rating": ("training_rating", "mean"),
        "Retention Rate": ("retention_flag", "mean"),
        "Quiz Confidence Rate": ("quiz_confidence_flag", "mean"),
        "Response Count": ("trainer", "size"),
    }
    selected_metric = st.selectbox("Select metric", list(metric_options.keys()))
    value_column, aggregation = metric_options[selected_metric]

    trend = (
        valid_trend_df.groupby(["month", "trainer"], as_index=False)
        .agg(metric_value=(value_column, aggregation))
        .sort_values("month")
    )

    chart = px.line(
        trend,
        x="month",
        y="metric_value",
        color="trainer",
        markers=True,
        title=f"{selected_metric} Trend by Trainer",
        labels={"month": "Month", "metric_value": selected_metric, "trainer": "Trainer"},
    )
    if "Rate" in selected_metric:
        chart.update_yaxes(tickformat=".0%")
    chart.update_layout(title_x=0)
    st.plotly_chart(chart, use_container_width=True)


def show_feedback_theme_analysis(df: pd.DataFrame) -> None:
    st.header("Feedback Theme Analysis")
    st.write("Open-ended improvement comments are classified into practical themes so recurring needs are visible.")

    theme_rows = get_theme_rows(df)
    if theme_rows.empty:
        st.info("No feedback themes are available for the current filters.")
        return

    theme_counts = (
        theme_rows.groupby("feedback_theme")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    theme_by_trainer = (
        theme_rows.groupby(["trainer", "feedback_theme"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        chart = px.bar(
            theme_counts,
            x="count",
            y="feedback_theme",
            orientation="h",
            title="Most Common Feedback Themes",
            labels={"count": "Theme Mentions", "feedback_theme": "Feedback Theme"},
        )
        chart.update_layout(title_x=0, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(chart, use_container_width=True)

    with col2:
        chart = px.bar(
            theme_by_trainer,
            x="trainer",
            y="count",
            color="feedback_theme",
            barmode="stack",
            title="Feedback Themes by Trainer",
            labels={"trainer": "Trainer", "count": "Theme Mentions", "feedback_theme": "Feedback Theme"},
        )
        chart.update_layout(title_x=0)
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Theme Counts")
    st.dataframe(theme_counts.rename(columns={"feedback_theme": "Feedback Theme", "count": "Mentions"}), use_container_width=True, hide_index=True)


def show_trainer_action_plan(df: pd.DataFrame) -> None:
    st.header("Trainer Action Plan")

    trainers = sorted(df["trainer"].dropna().unique())
    if not trainers:
        st.info("No trainers are available for the current filters.")
        return

    selected_trainer = st.selectbox("Select trainer", trainers)
    trainer_df = df[df["trainer"] == selected_trainer].copy()
    summary = build_trainer_summary(trainer_df)

    if summary.empty:
        st.info("No action plan data is available for this trainer.")
        return

    trainer_metrics = summary.iloc[0]
    kpi_columns = st.columns(5)
    with kpi_columns[0]:
        kpi_card("Responses", f"{int(trainer_metrics['response_count']):,}")
    with kpi_columns[1]:
        kpi_card("Trainer Rating", format_number(trainer_metrics["avg_trainer_rating"]))
    with kpi_columns[2]:
        kpi_card("Training Rating", format_number(trainer_metrics["avg_training_rating"]))
    with kpi_columns[3]:
        kpi_card("Retention", format_percent(trainer_metrics["retention_rate"]))
    with kpi_columns[4]:
        kpi_card("Quiz Confidence", format_percent(trainer_metrics["quiz_confidence_rate"]))

    theme_rows = get_theme_rows(trainer_df)
    theme_counts = (
        theme_rows.groupby("feedback_theme")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    st.subheader(f"Top Feedback Themes for {selected_trainer}")
    st.dataframe(theme_counts.rename(columns={"feedback_theme": "Feedback Theme", "count": "Mentions"}), use_container_width=True, hide_index=True)

    st.subheader("Recommended Actions")
    actionable_themes = [
        theme
        for theme in theme_counts["feedback_theme"].tolist()
        if theme in ACTION_RECOMMENDATIONS
    ]
    if actionable_themes:
        for theme in actionable_themes:
            st.markdown(f"**{theme}:** {ACTION_RECOMMENDATIONS[theme]}")
    else:
        st.info("No specific action themes were detected. Continue monitoring comments and response volume.")

    st.subheader("Selected Raw Comments")
    comments = trainer_df[["date", "company", "product", "improvement_feedback", "overall_feedback"]].copy()
    comments["date"] = comments["date"].dt.strftime("%Y-%m-%d").fillna("Unknown Date")
    st.dataframe(
        comments.rename(
            columns={
                "date": "Date",
                "company": "Company",
                "product": "Product",
                "improvement_feedback": "Improvement Feedback",
                "overall_feedback": "Overall Feedback",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_raw_feedback_explorer(df: pd.DataFrame) -> None:
    st.header("Raw Feedback Explorer")
    st.write("Use this view to inspect the filtered survey records behind the executive metrics.")

    available_columns = [
        column
        for column in [
            "id",
            "date",
            "respondent_name",
            "company",
            "trainer",
            "product",
            "retained",
            "trainer_rating",
            "training_rating",
            "quiz_confidence",
            "improvement_feedback",
            "overall_feedback",
        ]
        if column in df.columns
    ]

    search_term = st.text_input("Search comments, company, product, or trainer")
    explorer_df = df.copy()
    if search_term:
        searchable_columns = ["company", "trainer", "product", "improvement_feedback", "overall_feedback"]
        search_mask = pd.Series(False, index=explorer_df.index)
        for column in searchable_columns:
            if column in explorer_df.columns:
                search_mask = search_mask | explorer_df[column].str.contains(search_term, case=False, na=False)
        explorer_df = explorer_df[search_mask]

    display_df = explorer_df[available_columns].copy()
    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d").fillna("Unknown Date")

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_training_center_feedback.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Trainer Improvement Dashboard")
    st.caption("Training Center Survey feedback translated into trainer performance, trend, and improvement insights.")

    if not CSV_FILE.exists():
        st.error(
            f"CSV file not found: `{CSV_FILE}`. Please place the Training Center survey CSV in the project folder."
        )
        st.stop()

    try:
        df = load_and_clean_data(CSV_FILE)
    except Exception as exc:
        st.error(f"Unable to load the survey data: {exc}")
        st.stop()

    if df.empty:
        st.warning("No valid survey responses are available after cleaning and filtering unrealistic dates.")
        st.stop()

    filtered_df = apply_sidebar_filters(df)
    if filtered_df.empty:
        st.warning("No survey responses match the current filters.")
        st.stop()

    tabs = st.tabs(
        [
            "Executive Overview",
            "Trainer Performance",
            "Trainer Evolution Over Time",
            "Feedback Theme Analysis",
            "Trainer Action Plan",
            "Raw Feedback Explorer",
        ]
    )

    with tabs[0]:
        show_executive_overview(filtered_df)
    with tabs[1]:
        show_trainer_performance(filtered_df)
    with tabs[2]:
        show_trainer_evolution(filtered_df)
    with tabs[3]:
        show_feedback_theme_analysis(filtered_df)
    with tabs[4]:
        show_trainer_action_plan(filtered_df)
    with tabs[5]:
        show_raw_feedback_explorer(filtered_df)


if __name__ == "__main__":
    main()
