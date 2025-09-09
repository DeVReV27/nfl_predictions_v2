# NFL Prediction Elite

Production-ready Streamlit application for exploring NFL player props, team matchups, and league-wide insights. It combines real NFL statistics (via `nfl_data_py`) with concise AI-assisted analysis to output an OVER/UNDER recommendation accompanied by recent performance context and visual diagnostics.

- UI: Streamlit with interactive sidebar controls and polished styling
- Data: `nfl_data_py` weekly player data, team metadata, and rosters
- AI: OpenAI Chat Completions for succinct, data-aware reasoning
- Visuals: Matplotlib trend lines, rolling averages, and histograms
- Seasons: Includes support for 2025 rosters; historical and recent seasons selectable

---

## Features

- Player Props
  - OVER/UNDER call with brief, structured reasoning
  - Recent performance metrics (avg, max, hit rate over the selected line)
  - Visualizations: game-by-game trend, 3-game rolling average, distribution histogram
  - Categories supported: Passing/Rushing/Receiving yards and TDs, Receptions, Fantasy Points, Pass+Rush Yards, Total TDs

- Team Matchup
  - Compare two teams on offense, scoring (TDs), and fantasy production aggregates by week

- League Insights
  - Leaders across key stats (top 10)
  - League-average summary table

- Robust Data Handling
  - Column harmonization for common stat fields
  - Caching via `@st.cache_data` to reduce API calls and improve responsiveness

---

## Project Structure

- `nfl_app.py` — Streamlit UI, data loaders, AI prompt builder, visualizations, and pages:
  - Player Props, Team Matchup, League Insights
  - Matplotlib-based trend and distribution charts
- `nfl_module.py` — Data utilities and safeguards:
  - Cached team metadata, rosters, weekly stats
  - Team aggregates, schedules, and safe column mappings
- `requirements.txt` — Python dependencies and version constraints
- `.env.example` — Template for environment variables (copy to `.env`)
- `.gitignore` — Excludes `.env`, virtual envs, caches, and common build artifacts
- `LICENSE` — Apache License 2.0
- `README.md` — This documentation

---

## Quickstart

Prerequisites:
- Python 3.10+ recommended
- macOS/Linux/Windows with internet access

1) Clone the repository
```
git clone https://github.com/DeVReV27/nfl_predictions_v2.git
cd nfl_predictions_v2
```

2) Create and activate a virtual environment
- macOS/Linux:
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Windows (PowerShell):
  ```
  py -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

3) Install dependencies
```
pip install -r requirements.txt
```

4) Configure environment variables
```
cp .env.example .env
# then open .env and set:
# OPENAI_API_KEY=your_openai_api_key_here
```

5) Run the app
```
streamlit run nfl_app.py
```

Streamlit will open in your browser (typically http://localhost:8501).

---

## Usage Guide

1) Choose a Season
- 2025 (rosters available; stats as season progresses), 2024, 2023, or 2022

2) Select Analysis Type
- Player Props:
  - Select Player’s Team → Player → Opponent
  - Choose Stat Category and enter a line (e.g., 100.5)
  - Click “Get Player Prediction” to see an OVER/UNDER call with reasoning, metrics, and charts
- Team Matchup:
  - Select Home and Away teams to compare offense, scoring, and fantasy production
- League Insights:
  - Browse leaders (top 10) and league averages for the selected season

3) Review Outputs
- Recommendation (Over/Under) with short explanation
- Recent performance metrics and hit rate
- Game-by-game trend with rolling average and distribution histogram
- Tables for team comparisons or top leaders (depending on page)

---

## Configuration (.env)

The app uses a single key for AI reasoning:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Notes:
- `.env` is excluded from version control via `.gitignore`.
- The code currently requests the model name `"gpt-5"` as a placeholder in `nfl_app.py`. If unavailable on your account, change it to a model you can access (e.g., `gpt-4o-mini`, `gpt-4o`, etc.).

---

## Notes and Limitations

- Data Coverage
  - `nfl_data_py` schemas can vary by season; the app harmonizes common columns and shows warnings if fields are missing.
- Season Behavior
  - `get_current_season()` is set to 2025 to make pre-season roster browsing convenient. Adjust if desired.
- Caching
  - Teams/rosters are cached for 1 hour; weekly stats for 10 minutes. Clear Streamlit cache if you need fresh data immediately.
- AI Output
  - The AI explanation is data-aware but concise; it does not perform heavy numerical modeling. Use outputs as informational, not financial advice.

---

## Troubleshooting

- OPENAI_API_KEY missing
  - The app will stop with an error if not found. Ensure `.env` is present and your terminal is in the project root when launching Streamlit.
- Model access errors
  - If the requested model is unavailable, update the model string in `get_enhanced_prediction` to a model your account supports.
- Package/version issues
  - If pip has trouble resolving versions, consider relaxing upper bounds in `requirements.txt` within your environment constraints.
- Streamlit port in use
  - Run on a custom port: `streamlit run nfl_app.py --server.port 8502`.

---

## Acknowledgements

- [nfl_data_py](https://github.com/nflverse/nfl_data_py)
- [Streamlit](https://streamlit.io/)
- Matplotlib, NumPy, pandas

---

## Disclaimer

This project is for educational and entertainment purposes only. No guarantees of accuracy or profitability. Please bet responsibly.

---

## License

Apache License 2.0 — see the `LICENSE` file for details.

---

App by DeVReV27
