# 🏎️ F1 2026 Race Prediction

A Machine Learning project that predicts Formula 1 race podium finishes for the 2026 season using historical data from 2000 to 2026.

🔴 **Live App:** [https://f1-predictions-7.streamlit.app](https://f1-predictions-7.streamlit.app)

---

## 📸 Dashboard Preview

| Home | Race Predictions | Driver Analysis |
|------|-----------------|-----------------|
| 2026 standings + season overview | Podium probabilities per circuit | Elo rating career history |

---

## 🧠 How It Works

1. **Data** — pulls race results from 2000 to 2026 using the Jolpica/Ergast API
2. **Features** — engineers 12 prediction features from raw data
3. **Model** — trains Random Forest and XGBoost classifiers
4. **Predict** — outputs podium probability for each driver at any circuit

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 92.6% |
| XGBoost | 92.1% |

Trained on 10,071 rows (2000-2024), tested on 567 rows (2025-2026)

---

## ⚙️ Features Used

| Feature | Description |
|---------|-------------|
| `grid` | Starting grid position |
| `driver_avg_finish` | Weighted rolling avg finish (last 5 races) |
| `driver_avg_points` | Weighted rolling avg points (last 5 races) |
| `driver_dnf_rate` | Driver retirement rate |
| `team_avg_points` | Constructor rolling pace (last 3 years) |
| `team_dnf_rate` | Constructor reliability rate |
| `circuit_avg_finish` | Driver historical avg finish at this circuit |
| `elo_rating` | Custom Elo skill rating (resets on team change) |
| `elo_gap_to_field` | How far above/below field average Elo |
| `team_competitiveness` | Team pace relative to season average |
| `driver_consistency` | Std deviation of recent finishes |
| `grid_penalty` | Podium likelihood factor based on grid slot |

---

## 🔧 Advanced ML Concepts Applied

- **Time decay** — recent races weighted higher than old ones
- **Elo rating system** — driver skill score updated after every race
- **Team change detection** — Elo partially resets when driver switches team
- **Probability calibration** — raw probabilities normalized for realism

---

## 📁 Project Structure

```
f1-predictions/
├── app/
│   └── app.py              # Streamlit dashboard
├── data/
│   ├── raw/                # race_results.csv (2000-2026)
│   └── processed/          # features.csv (engineered features)
├── models/
│   ├── rf_model.pkl        # trained Random Forest
│   └── xgb_model.pkl       # trained XGBoost
├── notebooks/
│   └── explore.ipynb       # EDA and data exploration
├── src/
│   ├── data_loader.py      # Jolpica API data pipeline
│   ├── features.py         # feature engineering
│   └── model.py            # model training and prediction
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/Gunjan10-droid/f1-predictions.git
cd f1-predictions
```

**2. Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Fetch fresh data:**
```bash
python src/data_loader.py
```

**5. Build features:**
```bash
python src/features.py
```

**6. Train models:**
```bash
python src/model.py
```

**7. Run dashboard:**
```bash
streamlit run app/app.py
```

---

## 📡 Data Sources

- **Jolpica API** (free) — race results, grid positions, standings 1950-2026
- **No API key required**

---

## 🏁 Dashboard Pages

- **Home** — 2026 season overview, current standings chart
- **Race Predictions** — select any circuit, predict podium with XGBoost or Random Forest
- **Driver Analysis** — Elo rating career history, 2026 race by race results
- **Model Insights** — feature importance charts for both models


## 🛠️ Tech Stack

- **Python 3.9**
- **pandas** — data manipulation
- **scikit-learn** — Random Forest
- **XGBoost** — gradient boosting
- **Streamlit** — dashboard
- **matplotlib** — visualizations
- **requests** — API calls

---

## 👨‍💻 Author

**Gunjan Agrawal**
- GitHub: [@Gunjan10-droid](https://github.com/Gunjan10-droid)

---

## 📜 License

MIT License — feel free to use and modify!