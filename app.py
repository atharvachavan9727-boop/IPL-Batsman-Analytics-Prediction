
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="IPL Analytics & Prediction",
    page_icon="🏏",
    layout="wide"
)

# -----------------------------------
# LOAD DATASETS
# -----------------------------------

@st.cache_data
def load_data():

    batting = pd.read_csv(os.path.join("data", "ipl_batting_career_progression.csv"))
    players = pd.read_csv(os.path.join("data", "ipl_players_master.csv"))
    bowling = pd.read_csv(os.path.join("data", "ipl_bowling_career_progression.csv"))
    predictions = pd.read_csv(os.path.join("data", "ipl_2026_run_predictions.csv"))
    ml_output = pd.read_csv(os.path.join("data", "ipl_complete_ml_output.csv"))

    batting["country"] = batting["country"].fillna("Unknown")
    players["country"] = players["country"].fillna("Unknown")
    bowling["country"] = bowling["country"].fillna("Unknown")

    return batting, players, bowling, predictions, ml_output


batting, players, bowling, predictions, ml_output = load_data()

# -----------------------------------
# CUSTOM STYLING
# -----------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #101323;
    color: white;
}
[data-testid="stMetric"] {
    background-color: #202641;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #C7C9D9;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF;
}
h1, h2, h3 {
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🏏 IPL ANALYTICS")

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "IPL Overview",
        "Batsman Analysis",
        "Player Comparison",
        "2026 Predictions",
        "ML Model Insights"
    ]
)

# -----------------------------------
# IPL OVERVIEW
# -----------------------------------

if page == "IPL Overview":

    st.title("🏏 IPL Batsman Analytics")
    st.caption("Historical IPL batting performance | 2008–2025")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total IPL Runs", f"{batting['runs_scored'].sum():,}")
    col2.metric("Total Players", batting["player_name"].nunique())
    col3.metric("IPL Seasons", batting["season"].nunique())
    col4.metric("Average Strike Rate",
                round(batting["batting_strike_rate"].mean(), 2))

    st.divider()

    season_runs = batting.groupby("season", as_index=False)["runs_scored"].sum()

    fig = px.line(
        season_runs,
        x="season",
        y="runs_scored",
        markers=True,
        title="Total IPL Runs by Season",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    top_players = (
        batting.groupby("player_name")["runs_scored"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig2 = px.bar(
        top_players.sort_values("runs_scored"),
        x="runs_scored",
        y="player_name",
        orientation="h",
        title="Top 10 IPL Run Scorers",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------
# BATSMAN ANALYSIS
# -----------------------------------

elif page == "Batsman Analysis":

    st.title("📊 Batsman Performance Analysis")

    selected_player = st.selectbox(
        "Select Batsman",
        sorted(batting["player_name"].unique())
    )

    player_data = batting[
        batting["player_name"] == selected_player
    ].sort_values("season")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Runs", int(player_data["runs_scored"].sum()))
    c2.metric("Average Runs", round(player_data["runs_scored"].mean(), 2))
    c3.metric("Strike Rate",
              round(player_data["batting_strike_rate"].mean(), 2))
    c4.metric("Highest Score", int(player_data["highest_score"].max()))

    fig = px.line(
        player_data,
        x="season",
        y="runs_scored",
        markers=True,
        title=f"{selected_player} - Runs by Season",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    boundary_data = player_data.melt(
        id_vars="season",
        value_vars=["fours", "sixes"],
        var_name="Boundary",
        value_name="Count"
    )

    fig2 = px.bar(
        boundary_data,
        x="season",
        y="Count",
        color="Boundary",
        barmode="group",
        title="Fours and Sixes by Season",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------
# PLAYER COMPARISON
# -----------------------------------

elif page == "Player Comparison":

    st.title("⚔️ Player Comparison")

    player_list = sorted(batting["player_name"].unique())

    p1 = st.selectbox("Select First Player", player_list, index=0)
    p2 = st.selectbox("Select Second Player", player_list, index=1)

    comparison = batting[batting["player_name"].isin([p1, p2])]

    comparison_summary = (
        comparison.groupby("player_name")
        .agg(
            Total_Runs=("runs_scored", "sum"),
            Average_Runs=("runs_scored", "mean"),
            Strike_Rate=("batting_strike_rate", "mean"),
            Fours=("fours", "sum"),
            Sixes=("sixes", "sum")
        )
        .reset_index()
    )

    st.dataframe(comparison_summary, use_container_width=True)

    fig = px.bar(
        comparison_summary,
        x="player_name",
        y="Total_Runs",
        color="player_name",
        title="Career Runs Comparison",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------
# 2026 PREDICTIONS
# -----------------------------------

elif page == "2026 Predictions":

    st.title("🔮 IPL 2026 Run Predictions")
    st.caption("Predictions generated using Tuned Random Forest")

    c1, c2, c3 = st.columns(3)

    c1.metric("Predicted Players", predictions["player_name"].nunique())
    c2.metric("Highest Prediction",
              round(predictions["predicted_runs_2026"].max(), 2))
    c3.metric("Average Prediction",
              round(predictions["predicted_runs_2026"].mean(), 2))

    st.divider()

    top10 = predictions.nlargest(10, "predicted_runs_2026")

    fig = px.bar(
        top10.sort_values("predicted_runs_2026"),
        x="predicted_runs_2026",
        y="player_name",
        orientation="h",
        title="Top 10 Predicted Batsmen - 2026",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Complete Prediction Rankings")

    st.dataframe(
        predictions.sort_values(
            "predicted_runs_2026", ascending=False
        ).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )


# -----------------------------------
# ML MODEL INSIGHTS
# -----------------------------------

elif page == "ML Model Insights":

    st.title("🧠 Machine Learning Model Insights")

    st.subheader("Tuned Random Forest Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", "90.22")
    c2.metric("RMSE", "128.99")
    c3.metric("R² Score", "0.5166")

    st.caption("Evaluation on the 2025 test set.")

    st.divider()

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": [
            "Average Fours",
            "Average Runs",
            "Recent Runs",
            "Average Highest Score",
            "Average Batting Average",
            "Average Strike Rate",
            "Average Innings",
            "Average Sixes",
            "Average Fifties",
            "Average Hundreds"
        ],
        "Importance": [
            0.348680,
            0.197154,
            0.105800,
            0.074715,
            0.063417,
            0.054373,
            0.053670,
            0.051888,
            0.041693,
            0.008610
        ]
    })

    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Dataset")

    st.dataframe(
        ml_output.sort_values("prediction_rank"),
        use_container_width=True,
        hide_index=True
    )
