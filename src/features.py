import pandas as pd
import numpy as np

def load_and_clean(path="data/raw/race_results.csv"):
    df = pd.read_csv(path)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['year'].astype(int)
    df['round'] = df['round'].astype(int)
    
    df['finish_position'] = pd.to_numeric(df['position'], errors='coerce').fillna(20).astype(int)
    df['is_dnf'] = (~df['status'].str.contains('Finished|Lap', na=False)).astype(int)
    
    df = df.sort_values(['date', 'round', 'finish_position']).reset_index(drop=True)
    
    print(f"Cleaned data: {df.shape}")
    return df


def add_driver_form(df, window=5):

    df = df.copy()

    def weighted_rolling(series, window=5):
        result = []
        values = series
        for i in range(len(values)):
            if i == 0:
                result.append(np.nan)
                continue
            past = values[max(0, i-window):i]
            # weights: most recent = highest weight
            weights = np.arange(1, len(past)+1)
            weighted_avg = np.average(past, weights=weights)
            result.append(weighted_avg)
        return result

    for driver in df['driver_id'].unique():
        mask = df['driver_id'] == driver
        driver_rows = df[mask].copy()

        df.loc[mask, 'driver_avg_finish']  = weighted_rolling(driver_rows['finish_position'].tolist())
        df.loc[mask, 'driver_avg_points']  = weighted_rolling(driver_rows['points'].tolist())
        df.loc[mask, 'driver_dnf_rate']    = weighted_rolling(driver_rows['is_dnf'].tolist())

    # fill nulls with overall average
    df['driver_avg_finish']  = df['driver_avg_finish'].fillna(df['finish_position'].mean())
    df['driver_avg_points']  = df['driver_avg_points'].fillna(df['points'].mean())
    df['driver_dnf_rate']    = df['driver_dnf_rate'].fillna(df['is_dnf'].mean())

    print("Driver form features added with time decay!")
    return df


def add_constructor_form(df, window=5):

    df = df.copy()
    max_year = df['year'].max()

    # only use last 3 years for team rolling stats
    recent = df[df['year'] >= max_year - 3].copy()

    team_recent = (
        recent.groupby(['constructor', 'year', 'round'])
        .agg(
            team_avg_points_raw=('points', 'mean'),
            team_dnf_raw=('is_dnf', 'mean')
        )
        .reset_index()
        .sort_values(['constructor', 'year', 'round'])
    )

    team_recent['team_avg_points'] = (
        team_recent.groupby('constructor')['team_avg_points_raw']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    team_recent['team_dnf_rate'] = (
        team_recent.groupby('constructor')['team_dnf_raw']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df = df.merge(
        team_recent[['constructor', 'year', 'round', 'team_avg_points', 'team_dnf_rate']],
        on=['constructor', 'year', 'round'],
        how='left'
    )

    # fill older years with overall average
    df['team_avg_points'] = df['team_avg_points'].fillna(df['points'].mean())
    df['team_dnf_rate']   = df['team_dnf_rate'].fillna(df['is_dnf'].mean())

    print("Constructor form features added (recent 3 years only)!")
    return df


def add_circuit_history(df):
    df = df.copy()
    df['circuit_avg_finish'] = (
        df.groupby(['driver_id', 'circuit_id'])['finish_position']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['circuit_avg_finish'] = df['circuit_avg_finish'].fillna(df['driver_avg_finish'])
    print("Circuit history features added!")
    return df


def add_elo(df, k=32):

    df = df.copy()
    ratings = {}
    driver_teams = {}  # track which team each driver is on
    elo_before = []

    for _, race in df.groupby(['year', 'round']):
        drivers = race['driver_id'].tolist()
        teams   = race.set_index('driver_id')['constructor'].to_dict()

        for d in drivers:
            current_team = teams.get(d)
            previous_team = driver_teams.get(d)

            # if driver changed team — partial reset!
            if previous_team is not None and current_team != previous_team:
                old_elo = ratings.get(d, 1500.0)
                # blend 70% personal skill + 30% default
                ratings[d] = (old_elo * 0.7) + (1500.0 * 0.3)
                print(f"  Team change detected: {d} {previous_team}→{current_team}, Elo reset {old_elo:.0f}→{ratings[d]:.0f}")

            driver_teams[d] = current_team
            elo_before.append(ratings.get(d, 1500.0))

        # update ratings based on finishing order
        race_sorted = race.sort_values('finish_position')
        driver_list = race_sorted['driver_id'].tolist()

        for i, winner in enumerate(driver_list):
            for loser in driver_list[i+1:]:
                rw = ratings.get(winner, 1500.0)
                rl = ratings.get(loser,  1500.0)
                exp_w = 1 / (1 + 10 ** ((rl - rw) / 400))
                ratings[winner] = rw + k * (1 - exp_w)
                ratings[loser]  = rl + k * (0 - (1 - exp_w))

    df['elo_rating'] = elo_before
    print("Elo ratings added with team change reset!")
    return df

def add_competition_features(df):
    """
    Add features that capture how competitive the field is
    so one driver cant dominate predictions unrealistically
    """
    df = df.copy()

    # 1. gap between driver elo and race average elo
    # if everyone has similar elo → no one dominates
    race_avg_elo = df.groupby(['year', 'round'])['elo_rating'].transform('mean')
    race_std_elo = df.groupby(['year', 'round'])['elo_rating'].transform('std').fillna(1)
    df['elo_gap_to_field'] = (df['elo_rating'] - race_avg_elo) / race_std_elo

    # 2. how competitive is the team relative to others this season
    season_team_avg = df.groupby(['year', 'constructor'])['points'].transform('mean')
    season_avg      = df.groupby('year')['points'].transform('mean')
    df['team_competitiveness'] = season_team_avg / (season_avg + 0.01)

    # 3. driver consistency — std of recent finishes (lower = more consistent)
    df['driver_consistency'] = (
        df.groupby('driver_id')['finish_position']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
        .fillna(5.0)
    )

    # 4. grid penalty factor
    # even a great driver starting p15 shouldnt have 80% podium chance
    df['grid_penalty'] = df['grid'].apply(
        lambda x: 1.0 if x <= 3
        else 0.8 if x <= 6
        else 0.5 if x <= 10
        else 0.2
    )

    print("Competition features added!")
    return df

def build_features(path="data/raw/race_results.csv"):
    df = load_and_clean(path)
    df = add_driver_form(df)
    df = add_constructor_form(df)
    df = add_circuit_history(df)
    df = add_elo(df)
    df = add_competition_features(df) 

    df.to_csv("data/processed/features.csv", index=False)
    print(f"\nFinal feature table: {df.shape}")
    return df


if __name__ == "__main__":
    df = build_features()
    
    # check 2026 driver stats to verify fixes
    print("\n2026 Driver Elo Ratings:")
    latest = df[df['year']==2026].sort_values('round', ascending=False)
    latest = latest.drop_duplicates('driver_id')[['driver_name','constructor','elo_rating','driver_avg_finish','team_avg_points']]
    latest = latest.sort_values('elo_rating', ascending=False)
    print(latest.to_string(index=False))