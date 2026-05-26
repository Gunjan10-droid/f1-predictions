import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

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

def load_data():
    df = pd.read_csv("data/processed/features.csv")
    
    # target — did this driver finish in top 3?
    df['is_podium'] = (df['finish_position'] <= 3).astype(int)
    
    return df


def split_data(df):
    # train on 2000-2024, test on 2025-2026
    train = df[df['year'] <= 2024]
    test  = df[df['year'] >= 2025]
    
    X_train = train[FEATURES]
    y_train = train['is_podium']
    
    X_test  = test[FEATURES]
    y_test  = test['is_podium']
    
    print(f"Train size: {len(train)} rows")
    print(f"Test size:  {len(test)} rows")
    print(f"Podium % in train: {y_train.mean()*100:.1f}%")
    
    return X_train, y_train, X_test, y_test, test


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    print("Random Forest trained!")
    return rf


def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost trained!")
    return xgb_model


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc*100:.1f}%")
    return acc


def feature_importance(model, name):
    importance = pd.DataFrame({
        'feature':   FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n{name} — Feature Importance:")
    print(importance.to_string(index=False))
    return importance


def predict_next_race(model, df, circuit, year=2026):
    """Predict podium probabilities for a specific circuit in 2026"""
    
    # get latest stats for each driver from 2026
    latest = df[df['year'] == 2026].sort_values('round', ascending=False)
    latest = latest.drop_duplicates('driver_id')
    
    # use average grid position as placeholder (before qualifying)
    latest = latest.copy()
    latest['grid'] = latest['grid']
    
    # get circuit history if available
    circuit_data = df[df['circuit_id'] == circuit]
    if len(circuit_data) > 0:
        circ_avg = circuit_data.groupby('driver_id')['finish_position'].mean()
        latest['circuit_avg_finish'] = latest['driver_id'].map(circ_avg).fillna(latest['circuit_avg_finish'])
    
    X = latest[FEATURES]
    latest = latest.copy()
    raw_probs = model.predict_proba(X)[:, 1]

    # cap max probability at 70% — F1 is unpredictable!
    # and scale so probabilities feel realistic
    raw_probs = np.clip(raw_probs, 0, 0.70)

    # normalize top drivers so they sum to ~1.5 (roughly 3 podium spots)
    top_sum = raw_probs.sum()
    if top_sum > 0:
        raw_probs = raw_probs / top_sum * 1.5

    # clip again after normalization
    raw_probs = np.clip(raw_probs, 0, 0.70)

    latest['podium_prob'] = raw_probs
    
    result = latest[['driver_name', 'constructor', 'elo_rating', 'podium_prob']]
    result = result.sort_values('podium_prob', ascending=False).head(10)
    result['podium_prob'] = (result['podium_prob'] * 100).round(1)
    
    print(f"\nPodium predictions for {circuit} {year}:")
    print(result.to_string(index=False))
    return result


def save_models(rf, xgb_model):
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print("\nModels saved to models/ folder!")


if __name__ == "__main__":
    # load data
    df = load_data()
    
    # split
    X_train, y_train, X_test, y_test, test_df = split_data(df)
    
    # train both models
    rf  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # evaluate both
    evaluate(rf,        X_test, y_test, "Random Forest")
    evaluate(xgb_model, X_test, y_test, "XGBoost")
    
    # feature importance
    feature_importance(rf,        "Random Forest")
    feature_importance(xgb_model, "XGBoost")
    
    # predict next race — Monaco
    predict_next_race(xgb_model, df, circuit='monaco')
    
    # save models
    save_models(rf, xgb_model)
