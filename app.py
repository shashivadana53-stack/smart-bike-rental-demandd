
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="BikeWise | Rental Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(0, 196, 204, .13), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(124, 58, 237, .15), transparent 32%),
        #080d19;
    color: #edf4ff;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10192a, #0a1020);
    border-right: 1px solid rgba(103,232,249,.18);
    min-width: 265px;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1550px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #f3f7ff !important;
}

p, label, li {
    color: #c7d2e5;
}

.brand {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg, #67e8f9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-sub {
    color: #8fa2bd;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 25px;
}

.hero {
    padding: 30px;
    border: 1px solid rgba(103,232,249,.22);
    border-radius: 22px;
    background: linear-gradient(120deg,
        rgba(14,165,180,.15),
        rgba(124,58,237,.12));
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 35px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    font-size: 14px;
    color: #b9c8df;
}

.metric-card {
    background: linear-gradient(145deg,
        rgba(25,39,61,.88),
        rgba(14,23,40,.90));
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 17px;
    padding: 20px;
    min-height: 125px;
    margin-bottom: 12px;
}

.metric-label {
    color: #9eb0c9;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #f5f8ff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #67e8f9;
    font-size: 11px;
    margin-top: 4px;
}

.section-heading {
    font-size: 21px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 12px;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(15,23,42,.40);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 15px;
    padding: 8px;
}

.stButton > button {
    background: linear-gradient(100deg, #0891b2, #7c3aed);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 18px;
    width: 100%;
}

.stButton > button:hover {
    border: 1px solid #67e8f9;
    color: white;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 7px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,.025);
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 10px 12px;
    cursor: pointer;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(103,232,249,.08);
    border-color: rgba(103,232,249,.25);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: rgba(148,163,184,.16);
}

.small-note {
    color: #8fa2bd;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def style_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe", family="Inter"),
        margin=dict(l=20, r=20, t=55, b=25),
        title_font=dict(size=17, color="#f3f7ff"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,.12)",
            zerolinecolor="rgba(148,163,184,.15)"
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,.12)",
            zerolinecolor="rgba(148,163,184,.15)"
        )
    )
    return fig


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
def load_dataset():
    possible_paths = [
        Path("data/hour.csv"),
        Path("hour.csv"),
        Path("Smart_Bike_Rental/data/hour.csv"),
        Path(__file__).parent / "data" / "hour.csv",
        Path(__file__).parent / "hour.csv"
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path), str(path)
            except Exception:
                pass

    return None, None


loaded_df, loaded_path = load_dataset()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">🚲 BikeWise</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Rental Intelligence Platform</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📊  Demand Analytics",
            "📈  Statistics",
            "🎯  Probability",
            "🧪  Hypothesis Testing",
            "🔗  Correlation Analysis",
            "📉  Regression Analysis",
            "✨  AI Prediction Studio",
            "⚡  Model Evaluation",
            "ℹ️  About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Dataset")

    uploaded_file = st.file_uploader(
        "Upload hour.csv",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            data_source = "Uploaded CSV"
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
    elif loaded_df is not None:
        df = loaded_df.copy()
        data_source = loaded_path
    else:
        df = None
        data_source = None

    if df is not None:
        st.success("Dataset loaded")
        st.caption(f"Source: {data_source}")
        st.caption(f"Rows: {len(df):,}")
    else:
        st.warning("hour.csv not found")
        st.caption("Upload the dataset above or place it in the data folder.")


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------
if df is None:
    st.markdown("""
    <div class="hero">
        <h1>Welcome to BikeWise 🚲</h1>
        <p>Your smart bike rental demand analysis and prediction dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "Please upload your UCI Bike Sharing dataset file named hour.csv "
        "using the sidebar."
    )

    st.markdown("### Expected dataset columns")
    st.code(
        "hr, season, holiday, weekday, workingday, weathersit, "
        "temp, atemp, hum, windspeed, casual, registered, cnt"
    )
    st.stop()


# --------------------------------------------------
# VALIDATE AND PREPARE DATA
# --------------------------------------------------
required_columns = [
    "hr", "season", "workingday", "weathersit",
    "temp", "hum", "windspeed", "cnt"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.info("Please upload the original UCI Bike Sharing hourly hour.csv file.")
    st.stop()

df = df.copy()

numeric_columns = [
    "hr", "season", "workingday", "weathersit",
    "temp", "hum", "windspeed", "cnt"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_columns).reset_index(drop=True)

if len(df) < 10:
    st.error("The dataset has too few valid rows for analysis.")
    st.stop()

df["day_type"] = np.where(
    df["workingday"] == 1,
    "Working Day",
    "Non-Working Day"
)

df["weather_label"] = df["weathersit"].map({
    1: "Clear / Few Clouds",
    2: "Mist / Cloudy",
    3: "Light Rain / Snow",
    4: "Heavy Rain / Snow"
}).fillna("Other / Unknown")

df["season_label"] = df["season"].map({
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
}).fillna("Unknown")

df["hour_label"] = df["hr"].astype(int).astype(str).str.zfill(2) + ":00"

median_demand = float(df["cnt"].median())
mean_demand = float(df["cnt"].mean())
std_demand = float(df["cnt"].std())
variance_demand = float(df["cnt"].var())
min_demand = float(df["cnt"].min())
max_demand = float(df["cnt"].max())

df["demand_level"] = np.where(
    df["cnt"] > median_demand,
    "High Demand",
    "Low Demand"
)

# --------------------------------------------------
# BUILD MACHINE LEARNING MODEL
# --------------------------------------------------
features = ["hr", "season", "workingday", "weathersit",
            "temp", "hum", "windspeed"]

X = df[features]
y = df["cnt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "🏠  Overview":
    st.markdown("""
    <div class="hero">
        <h1>Smart Bike Rental Intelligence</h1>
        <p>Understand rental patterns, explore statistics, and predict bike demand
        using data analysis and machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Records", f"{len(df):,}", "Hourly observations")

    with c2:
        metric_card("Average Demand", f"{mean_demand:.1f}", "Bikes per hour")

    with c3:
        metric_card("Peak Demand", f"{max_demand:.0f}", "Maximum observed")

    with c4:
        metric_card("Model R²", f"{r2:.3f}", "Evaluation score")

    st.markdown('<div class="section-heading">Rental Demand Overview</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:
        hourly = df.groupby("hr", as_index=False)["cnt"].mean()

        fig = px.area(
            hourly,
            x="hr",
            y="cnt",
            markers=True,
            title="Average Bike Rentals by Hour",
            labels={"hr": "Hour of Day", "cnt": "Average Rentals"}
        )
        fig.update_traces(line=dict(width=3), fill="tozeroy")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        season_data = df.groupby("season_label", as_index=False)["cnt"].mean()

        fig = px.bar(
            season_data,
            x="season_label",
            y="cnt",
            title="Average Demand by Season",
            labels={"season_label": "Season", "cnt": "Average Rentals"}
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Project Highlights")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.info("📊 **Data Analysis**\n\nExplore hourly demand and seasonal trends.")

    with h2:
        st.info("🧪 **Statistical Testing**\n\nCompare demand across different day types.")

    with h3:
        st.info("🤖 **Machine Learning**\n\nEstimate demand using a regression model.")


# --------------------------------------------------
# DEMAND ANALYTICS
# --------------------------------------------------
elif page == "📊  Demand Analytics":
    st.markdown("""
    <div class="hero">
        <h1>Demand Analytics</h1>
        <p>Explore how time, season, weather, and day type affect rental demand.</p>
    </div>
    """, unsafe_allow_html=True)

    hourly = df.groupby("hr", as_index=False)["cnt"].mean()
    season_data = df.groupby("season_label", as_index=False)["cnt"].mean()
    day_data = df.groupby("day_type", as_index=False)["cnt"].mean()
    weather_data = df.groupby("weather_label", as_index=False)["cnt"].mean()

    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            hourly, x="hr", y="cnt", markers=True,
            title="Hourly Demand Pattern",
            labels={"hr": "Hour", "cnt": "Average Rentals"}
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        fig = px.bar(
            season_data, x="season_label", y="cnt",
            title="Seasonal Demand",
            labels={"season_label": "Season", "cnt": "Average Rentals"}
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.bar(
            day_data, x="day_type", y="cnt",
            title="Working Day vs Non-Working Day",
            labels={"day_type": "Day Type", "cnt": "Average Rentals"}
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c4:
        fig = px.bar(
            weather_data, x="weather_label", y="cnt",
            title="Weather and Rental Demand",
            labels={"weather_label": "Weather", "cnt": "Average Rentals"}
        )
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Explore a specific hour")

    selected_hour = st.slider("Choose hour of day", 0, 23, 8)

    selected_rows = df[df["hr"] == selected_hour]

    if not selected_rows.empty:
        st.metric(
            f"Average rentals at {selected_hour:02d}:00",
            f"{selected_rows['cnt'].mean():.1f} bikes"
        )


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------
elif page == "📈  Statistics":
    st.markdown("""
    <div class="hero">
        <h1>Statistical Analysis</h1>
        <p>Understand the distribution and central tendency of bike rentals.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Mean", f"{mean_demand:.2f}", "Average demand")

    with c2:
        metric_card("Median", f"{median_demand:.2f}", "Middle observation")

    with c3:
        metric_card("Standard Deviation", f"{std_demand:.2f}", "Spread of demand")

    with c4:
        metric_card("Variance", f"{variance_demand:.2f}", "Squared spread")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df, x="cnt", nbins=45,
            title="Distribution of Bike Rentals",
            labels={"cnt": "Rental Count"}
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        fig = px.box(
            df, y="cnt",
            title="Demand Spread and Outliers",
            labels={"cnt": "Rental Count"}
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Summary Table")

    summary = df["cnt"].describe().to_frame("Rental Demand")
    st.dataframe(summary, use_container_width=True)

    st.caption(
        "Note: Values are calculated from the dataset currently loaded."
    )


# --------------------------------------------------
# PROBABILITY
# --------------------------------------------------
elif page == "🎯  Probability":
    st.markdown("""
    <div class="hero">
        <h1>Probability Explorer</h1>
        <p>Estimate the probability of high and low bike rental demand.</p>
    </div>
    """, unsafe_allow_html=True)

    high = df["cnt"] > median_demand
    low = ~high

    p_high = high.mean()
    p_low = low.mean()

    working = df["workingday"] == 1
    clear_weather = df["weathersit"] == 1

    p_high_working = high[working].mean() if working.any() else np.nan
    p_high_clear = high[clear_weather].mean() if clear_weather.any() else np.nan

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("P(High Demand)", f"{p_high:.2%}",
                    "Demand above median")

    with c2:
        metric_card("P(Low Demand)", f"{p_low:.2%}",
                    "Demand at or below median")

    with c3:
        metric_card("P(High | Working Day)",
                    f"{p_high_working:.2%}" if pd.notna(p_high_working) else "N/A",
                    "Conditional probability")

    st.markdown("### Compare probability conditions")

    prob_df = pd.DataFrame({
        "Condition": [
            "High Demand",
            "Low Demand",
            "High Demand on Working Day",
            "High Demand in Clear Weather"
        ],
        "Probability": [
            p_high, p_low, p_high_working, p_high_clear
        ]
    })

    fig = px.bar(
        prob_df, x="Condition", y="Probability",
        title="Demand Probability Comparison",
        labels={"Condition": "Condition", "Probability": "Probability"},
        text=prob_df["Probability"].map(
            lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
        )
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Try your own condition")

    condition = st.selectbox(
        "Select a condition",
        [
            "High demand",
            "Low demand",
            "Working day",
            "Non-working day",
            "Clear weather"
        ]
    )

    if condition == "High demand":
        subset = df[high]
    elif condition == "Low demand":
        subset = df[low]
    elif condition == "Working day":
        subset = df[working]
    elif condition == "Non-working day":
        subset = df[~working]
    else:
        subset = df[clear_weather]

    st.metric(
        f"Probability of selected condition: {condition}",
        f"{len(subset) / len(df):.2%}"
    )


# --------------------------------------------------
# HYPOTHESIS TESTING
# --------------------------------------------------
elif page == "🧪  Hypothesis Testing":
    st.markdown("""
    <div class="hero">
        <h1>Hypothesis Testing</h1>
        <p>Test whether average bike rentals differ between working and non-working days.</p>
    </div>
    """, unsafe_allow_html=True)

    working_values = df.loc[df["workingday"] == 1, "cnt"]
    nonworking_values = df.loc[df["workingday"] == 0, "cnt"]

    if len(working_values) > 1 and len(nonworking_values) > 1:
        t_stat, p_value = stats.ttest_ind(
            working_values,
            nonworking_values,
            equal_var=False
        )

        mean_working = working_values.mean()
        mean_nonworking = nonworking_values.mean()

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Working Day Mean", f"{mean_working:.2f}",
                        "Average rentals")

        with c2:
            metric_card("Non-Working Mean", f"{mean_nonworking:.2f}",
                        "Average rentals")

        with c3:
            metric_card("P-Value", f"{p_value:.6f}",
                        "Welch's two-sample t-test")

        alpha = st.slider(
            "Significance level (alpha)",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01
        )

        st.markdown("### Hypotheses")

        st.write("**H₀:** The average bike rental demand is equal for both day types.")
        st.write("**H₁:** The average bike rental demand is different for the two day types.")

        if p_value < alpha:
            st.success(
                f"Reject H₀ because p-value ({p_value:.6f}) is less than alpha ({alpha:.2f}). "
                "The data provides evidence of a difference in average demand."
            )
        else:
            st.info(
                f"Fail to reject H₀ because p-value ({p_value:.6f}) is not less than "
                f"alpha ({alpha:.2f}). There is not enough evidence of a difference."
            )

        compare_df = pd.DataFrame({
            "Day Type": ["Working Day", "Non-Working Day"],
            "Average Rentals": [mean_working, mean_nonworking]
        })

        fig = px.bar(
            compare_df, x="Day Type", y="Average Rentals",
            title="Average Demand Comparison",
            text_auto=".2f"
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

        st.caption(
            "Welch's t-test is used because it does not assume equal population variances."
        )
    else:
        st.warning("Not enough records for hypothesis testing.")


# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------
elif page == "🔗  Correlation Analysis":
    st.markdown("""
    <div class="hero">
        <h1>Correlation Analysis</h1>
        <p>Explore relationships between weather, time, and bike rental demand.</p>
    </div>
    """, unsafe_allow_html=True)

    corr_cols = [
        "hr", "temp", "hum", "windspeed",
        "season", "workingday", "weathersit", "cnt"
    ]

    corr = df[corr_cols].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(style_chart(fig, 600), use_container_width=True)

    st.markdown("### Correlation with Rental Demand")

    demand_corr = (
        corr["cnt"]
        .drop("cnt")
        .sort_values()
        .reset_index()
    )
    demand_corr.columns = ["Feature", "Correlation"]

    fig = px.bar(
        demand_corr,
        x="Correlation",
        y="Feature",
        orientation="h",
        title="Feature Correlation with Bike Demand",
        color="Correlation",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Explore two variables")

    left, right = st.columns(2)

    with left:
        x_var = st.selectbox(
            "X-axis variable",
            ["hr", "temp", "hum", "windspeed", "season"]
        )

    with right:
        y_var = st.selectbox(
            "Y-axis variable",
            ["cnt", "temp", "hum", "windspeed"],
            index=0
        )

    fig = px.scatter(
        df.sample(min(2500, len(df)), random_state=42),
        x=x_var, y=y_var,
        opacity=0.55,
        trendline="ols",
        title=f"{x_var} vs {y_var}"
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)


# --------------------------------------------------
# REGRESSION ANALYSIS
# --------------------------------------------------
elif page == "📉  Regression Analysis":
    st.markdown("""
    <div class="hero">
        <h1>Regression Analysis</h1>
        <p>Understand how selected features relate to rental demand.</p>
    </div>
    """, unsafe_allow_html=True)

    coefficients = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    }).sort_values("Coefficient")

    fig = px.bar(
        coefficients,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Linear Regression Coefficients",
        color="Coefficient",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Actual vs Predicted Demand")

    comparison = pd.DataFrame({
        "Actual": np.asarray(y_test),
        "Predicted": y_pred
    })

    fig = px.scatter(
        comparison,
        x="Actual",
        y="Predicted",
        opacity=0.55,
        title="Actual Rentals vs Model Predictions",
        labels={"Actual": "Actual Rentals", "Predicted": "Predicted Rentals"}
    )

    max_value = max(comparison["Actual"].max(), comparison["Predicted"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_value],
        y=[0, max_value],
        mode="lines",
        name="Ideal prediction",
        line=dict(dash="dash")
    ))

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Model Equation")

    equation_parts = [
        f"({coef:.3f} × {feature})"
        for feature, coef in zip(features, model.coef_)
    ]

    equation = " + ".join(equation_parts)

    st.code(f"Predicted Demand = {model.intercept_:.3f} + {equation}")

    st.caption(
        "Regression coefficients describe the fitted linear model; "
        "they do not prove that a feature causes demand to change."
    )


# --------------------------------------------------
# AI PREDICTION STUDIO
# --------------------------------------------------
elif page == "✨  AI Prediction Studio":
    st.markdown("""
    <div class="hero">
        <h1>Demand Prediction Studio</h1>
        <p>Enter the conditions and estimate hourly bike rental demand.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Configure rental conditions")

    c1, c2, c3 = st.columns(3)

    with c1:
        hour = st.slider("Hour of day", 0, 23, 8)
        season = st.selectbox(
            "Season",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Spring",
                2: "Summer",
                3: "Fall",
                4: "Winter"
            }[x]
        )
        workingday = st.selectbox(
            "Day type",
            options=[1, 0],
            format_func=lambda x: "Working Day" if x == 1 else "Non-Working Day"
        )

    with c2:
        temp = st.slider(
            "Temperature (normalised)",
            0.0, 1.0, 0.50, 0.01,
            help="The UCI dataset stores temperature on a normalised 0–1 scale."
        )
        humidity = st.slider(
            "Humidity (normalised)",
            0.0, 1.0, 0.50, 0.01
        )

    with c3:
        windspeed = st.slider(
            "Wind speed (normalised)",
            0.0, 1.0, 0.20, 0.01
        )
        weather = st.selectbox(
            "Weather condition",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear / Few Clouds",
                2: "Mist / Cloudy",
                3: "Light Rain / Snow",
                4: "Heavy Rain / Snow"
            }[x]
        )

    if st.button("✨ Predict Bike Demand"):
        input_data = pd.DataFrame([{
            "hr": hour,
            "season": season,
            "workingday": workingday,
            "weathersit": weather,
            "temp": temp,
            "hum": humidity,
            "windspeed": windspeed
        }])

        prediction = float(model.predict(input_data)[0])
        prediction = max(0, prediction)

        st.markdown("---")
        st.markdown("### Prediction Result")

        r1, r2 = st.columns([1, 1])

        with r1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Rental Demand</div>
                    <div class="metric-value">{prediction:.0f} bikes</div>
                    <div class="metric-note">Predicted for the selected hour</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:
            if prediction < median_demand:
                st.info("📉 The estimate is below the dataset median demand.")
            else:
                st.success("📈 The estimate is above the dataset median demand.")

        st.caption(
            "This is a machine-learning estimate, not a guaranteed actual count. "
            "The model uses the selected inputs and the currently loaded dataset."
        )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
elif page == "⚡  Model Evaluation":
    st.markdown("""
    <div class="hero">
        <h1>Model Evaluation</h1>
        <p>Check how well the regression model estimates bike rental demand.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("MAE", f"{mae:.2f}", "Mean Absolute Error")

    with c2:
        metric_card("RMSE", f"{rmse:.2f}", "Root Mean Squared Error")

    with c3:
        metric_card("R² Score", f"{r2:.3f}", "Coefficient of determination")

    st.markdown("### What do these metrics mean?")

    st.write(
        f"**MAE = {mae:.2f}:** The average absolute difference between "
        "actual and predicted rentals is approximately this many bikes."
    )

    st.write(
        f"**RMSE = {rmse:.2f}:** This error metric penalises larger errors more strongly."
    )

    st.write(
        f"**R² = {r2:.3f}:** This measures the fraction of variation in the test-set "
        "demand explained by the fitted model. It is not the same as prediction accuracy."
    )

    st.markdown("### Actual vs Predicted")

    plot_df = pd.DataFrame({
        "Actual": np.asarray(y_test),
        "Predicted": y_pred
    }).sample(min(1500, len(y_test)), random_state=42)

    fig = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        opacity=0.5,
        title="Model Predictions on Test Data"
    )

    upper = max(plot_df["Actual"].max(), plot_df["Predicted"].max())

    fig.add_trace(go.Scatter(
        x=[0, upper],
        y=[0, upper],
        mode="lines",
        name="Perfect prediction",
        line=dict(dash="dash")
    ))

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Dataset split")

    split_df = pd.DataFrame({
        "Dataset": ["Training Set", "Testing Set"],
        "Records": [len(X_train), len(X_test)]
    })

    fig = px.pie(
        split_df,
        names="Dataset",
        values="Records",
        hole=0.55,
        title="Training and Testing Records"
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------
elif page == "ℹ️  About Project":
    st.markdown("""
    <div class="hero">
        <h1>About BikeWise</h1>
        <p>Smart Bike Rental Demand Analysis and Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Objective")

    st.write(
        "The objective is to analyse historical bike rental data, identify demand "
        "patterns, study the relationship between demand and weather/time features, "
        "and predict hourly rental demand using machine learning."
    )

    st.markdown("### Technologies Used")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.info("**Python**\n\nData processing and model development")

    with t2:
        st.info("**Pandas + NumPy**\n\nData cleaning and statistical calculations")

    with t3:
        st.info("**Streamlit + Plotly**\n\nInteractive dashboard and visualisations")

    t4, t5, t6 = st.columns(3)

    with t4:
        st.info("**SciPy**\n\nStatistical hypothesis testing")

    with t5:
        st.info("**Scikit-learn**\n\nLinear regression and model evaluation")

    with t6:
        st.info("**UCI Bike Sharing**\n\nSource of the hourly dataset")

    st.markdown("### Machine Learning Workflow")

    st.write(
        "1. Load the hourly bike-sharing dataset.\n"
        "2. Validate and prepare the required columns.\n"
        "3. Select time, weather, and day-related features.\n"
        "4. Split data into training and testing sets.\n"
        "5. Train a Linear Regression model.\n"
        "6. Evaluate the model using MAE, RMSE, and R².\n"
        "7. Use the dashboard to explore patterns and make predictions."
    )

    st.markdown("### Dataset Information")

    st.write(f"**Current records:** {len(df):,}")
    st.write(f"**Features used by model:** {', '.join(features)}")
    st.write(f"**Prediction target:** cnt (total hourly bike rentals)")

    st.markdown("### References")

    st.markdown(
        "- UCI Bike Sharing Dataset: https://doi.org/10.24432/C5W894\n"
        "- Fanaee-T, H. and Gama, J. (2013). Event labeling combining ensemble "
        "detectors and background knowledge. https://doi.org/10.1007/s13748-013-0040-3"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    '<p class="small-note" style="text-align:center;">'
    'BikeWise · Smart Bike Rental Demand Analysis and Prediction System'
    ' · Built with Python and Streamlit'
    '</p>',
    unsafe_allow_html=True
)