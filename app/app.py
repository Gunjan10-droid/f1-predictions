import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# page config
st.set_page_config(
    page_title="F1 2026 Predictions",
    page_icon="🏎️",
    layout="wide"
)

# load models and data
@st.cache_resource
def load_models():
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    return rf, xgb

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/features.csv")
    df['is_podium'] = (df['finish_position'] <= 3).astype(int)
    return df

FEATURES = [
    'grid',
    'driver_avg_finish',
    'driver_dnf_rate',
    'driver_avg_points',
    'team_avg_points',
    'team_dnf_rate',
    'circuit_avg_finish',
    'elo_rating',
    'elo_gap_to_field',
    'team_competitiveness',
    'driver_consistency',
    'grid_penalty'
]

rf, xgb_model = load_models()
df = load_data()


# SIDEBAR

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
st.sidebar.title("F1 2026 Predictor")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🏆 Race Predictions", "📊 Driver Standings", "📈 Model Insights"]
)


# HOME PAGE

if page == "🏠 Home":
    st.title("🏎️ F1 2026 Season Predictor")
    st.markdown("Predicting Formula 1 race outcomes using Machine Learning")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Races Trained", "567+")
    with col2:
        st.metric("Model Accuracy", "92.6%")
    with col3:
        st.metric("Years of Data", "2000-2026")
    with col4:
        st.metric("Features Used", "12")

    st.divider()

    st.subheader("2026 Season So Far")
    races_2026 = df[df['year'] == 2026][['round', 'race_name', 'date']].drop_duplicates().sort_values('round')
    races_2026.columns = ['Round', 'Race', 'Date']
    st.dataframe(races_2026, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Current 2026 Driver Standings")
    standings = (
        df[df['year'] == 2026]
        .groupby('driver_name')['points']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    standings.columns = ['Driver', 'Points']
    standings.index = standings.index + 1

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(standings['Driver'][:10][::-1], standings['Points'][:10][::-1], color='red')
    ax.set_xlabel('Points')
    ax.set_title('2026 Driver Standings')
    for bar, val in zip(bars, standings['Points'][:10][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)


# RACE PREDICTIONS PAGE

elif page == "🏆 Race Predictions":
    st.title("🏆 Race Podium Predictions")

    # get all circuits
    circuits = sorted(df['circuit_id'].unique().tolist())
    upcoming = ['monaco', 'silverstone', 'monza', 'spa', 'suzuka', 'interlagos']
    default_circuits = [c for c in upcoming if c in circuits]

    col1, col2 = st.columns(2)
    with col1:
        circuit = st.selectbox("Select Circuit", circuits,
                               index=circuits.index('monaco') if 'monaco' in circuits else 0)
    with col2:
        model_choice = st.radio("Model", ["XGBoost", "Random Forest"], horizontal=True)

    if st.button("🔮 Predict Podium", type="primary"):
        model = xgb_model if model_choice == "XGBoost" else rf

        # get latest driver stats from 2026
        latest = df[df['year'] == 2026].sort_values('round', ascending=False)
        latest = latest.drop_duplicates('driver_id').copy()

        # update circuit history for selected circuit
        circuit_data = df[df['circuit_id'] == circuit]
        if len(circuit_data) > 0:
            circ_avg = circuit_data.groupby('driver_id')['finish_position'].mean()
            latest['circuit_avg_finish'] = latest['driver_id'].map(circ_avg).fillna(latest['circuit_avg_finish'])

        X = latest[FEATURES]
        raw_probs = model.predict_proba(X)[:, 1]
        raw_probs = np.clip(raw_probs, 0, 0.70)
        top_sum = raw_probs.sum()
        if top_sum > 0:
            raw_probs = raw_probs / top_sum * 1.5
        raw_probs = np.clip(raw_probs, 0, 0.70)
        latest['podium_prob'] = raw_probs

        result = latest[['driver_name', 'constructor', 'elo_rating', 'grid', 'podium_prob']].copy()
        result = result.sort_values('podium_prob', ascending=False).head(10)
        result['podium_prob'] = (result['podium_prob'] * 100).round(1)
        result.columns = ['Driver', 'Team', 'Elo Rating', 'Last Grid', 'Podium %']
        result['Elo Rating'] = result['Elo Rating'].round(0).astype(int)
        result = result.reset_index(drop=True)
        result.index = result.index + 1

        st.subheader(f"Podium Predictions — {circuit.replace('_', ' ').title()}")

        # top 3 podium display
        top3 = result.head(3)
        col1, col2, col3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        cols = [col1, col2, col3]

        for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
            with col:
                st.metric(
                    label=f"{medals[i]} P{i+1}",
                    value=row['Driver'],
                    delta=f"{float(row['Podium %']):.1f}% probability"
                )

        st.divider()
        st.subheader("Full Top 10 Predictions")
        st.dataframe(result, use_container_width=True)

        # bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['gold', 'silver', '#cd7f32'] + ['#ff1801'] * 7
        bars = ax.barh(result['Driver'][::-1], result['Podium %'][::-1], color=colors[::-1])
        ax.set_xlabel('Podium Probability (%)')
        ax.set_title(f'Podium Probabilities — {circuit.replace("_", " ").title()}')
        for bar, val in zip(bars, result['Podium %'][::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)


# DRIVER STANDINGS PAGE

elif page == "📊 Driver Standings":
    st.title("📊 Driver Analysis")

    driver_list = sorted(df[df['year'] == 2026]['driver_name'].unique().tolist())
    selected_driver = st.selectbox("Select Driver", driver_list)

    driver_data = df[df['driver_name'] == selected_driver].sort_values(['year', 'round'])

    col1, col2, col3 = st.columns(3)
    with col1:
        latest_elo = driver_data['elo_rating'].iloc[-1]
        st.metric("Current Elo Rating", f"{latest_elo:.0f}")
    with col2:
        avg_finish = driver_data[driver_data['year'] == 2026]['finish_position'].mean()
        st.metric("2026 Avg Finish", f"{avg_finish:.1f}")
    with col3:
        total_pts = driver_data[driver_data['year'] == 2026]['points'].sum()
        st.metric("2026 Total Points", f"{total_pts:.0f}")

    st.divider()

    # elo history chart
    st.subheader("Elo Rating History")
    elo_history = driver_data.groupby('year')['elo_rating'].last().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(elo_history['year'], elo_history['elo_rating'], marker='o', color='red', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Elo Rating')
    ax.set_title(f'{selected_driver} — Elo Rating Over Career')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # 2026 race by race results
    st.subheader("2026 Race Results")
    results_2026 = driver_data[driver_data['year'] == 2026][
        ['race_name', 'grid', 'finish_position', 'points', 'status']
    ].copy()
    results_2026.columns = ['Race', 'Grid', 'Finish', 'Points', 'Status']
    results_2026 = results_2026.reset_index(drop=True)
    results_2026.index = results_2026.index + 1
    st.dataframe(results_2026, use_container_width=True)


# MODEL INSIGHTS PAGE

elif page == "📈 Model Insights":
    st.title("📈 Model Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest — Feature Importance")
        rf_importance = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(rf_importance['Feature'][::-1], rf_importance['Importance'][::-1], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title('Random Forest')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("XGBoost — Feature Importance")
        xgb_importance = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(xgb_importance['Feature'][::-1], xgb_importance['Importance'][::-1], color='red')
        ax.set_xlabel('Importance')
        ax.set_title('XGBoost')
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    st.subheader("Grid Position vs Average Finish (2000–2026)")
    finished = df[df['finish_position'] <= 20].copy()
    grid_vs_finish = finished.groupby('grid')['finish_position'].mean().reset_index()
    grid_vs_finish = grid_vs_finish[grid_vs_finish['grid'] <= 20]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid_vs_finish['grid'], grid_vs_finish['finish_position'],
            marker='o', color='red', linewidth=2)
    ax.set_xlabel('Grid Position')
    ax.set_ylabel('Average Finish Position')
    ax.set_title('Grid Position vs Finish Position')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)