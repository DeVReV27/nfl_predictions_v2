# NFL Prediction Elite

Advanced Streamlit application that blends real NFL statistics with AI-generated analysis to help explore player props, team matchups, and league-wide insights. The app uses `nfl_data_py` for robust NFL datasets and OpenAI for concise, data-informed over/under calls.

For entertainment and educational purposes only.

## Features

- Player Props
  - AI prediction: OVER or UNDER with compact reasoning
  - Recent performance metrics: average, max, hit rate over a chosen line
  - Visualizations: game-by-game trend and distribution histogram
  - Supports common categories: Passing/Rushing/Receiving yards and TDs, Receptions, Fantasy Points, Pass+Rush Yards, Total TDs

- Team Matchup
  - Compare offensive, scoring, and fantasy production between two teams for a selected season

- League Insights
  - Statistical leaders (top 10) across multiple categories
  - League averages view

- Season Handling
  - Season selector including 2025 support (rosters shown; stats update as games are played)
  - Caching to reduce network calls (via Streamlit `@st.cache_data`)

## Tech Stack

- Python, Streamlit
- Data: `nfl_data_py`
- AI: OpenAI Chat Completions API
- Visualization: Matplotlib
- UI Enhancements: `streamlit-shadcn-ui`
- Data Handling: Pandas, NumPy

## Project Structure

- `nfl_app.py` — Main Streamlit app:
  - Loads environment and initializes OpenAI client
  - Defines caching and data-fetch helpers (teams, rosters, weekly stats, filtered player stats)
  - Builds data package for AI prompt
  - Renders pages: Player Props, Team Matchup, League Insights
  - Plots trends and distributions
- `nfl_module.py` — Data access/utilities:
  - Thin abstraction over `nfl_data_py` with caching and safe column harmonization
  - Roster helpers, weekly stats, team aggregates, schedule
- `requirements.txt` — Pinned and range-constrained dependencies
- `.env` — Local-only environment variables (NOT committed; see `.env.example`)
- `README.md` — This document

## Requirements

- Python 3.10+ recommended
- Internet access for `nfl_data_py` API pulls and OpenAI requests
- OpenAI API key

## Setup

1) Clone the repository
   - If you already pulled this code locally, skip to step 2
   - Otherwise:
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
   - Copy the example and set your OpenAI key:
     ```
     cp .env.example .env
     ```
     Then edit `.env`:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Running the App

From the project root (with venv activated):

```
streamlit run nfl_app.py
```

Streamlit will open a local browser tab. Use the sidebar to:

- Select Season (2025, 2024, 2023, 2022)
- Choose Analysis Type: Player Props, Team Matchup, or League Insights

For Player Props:
- Pick the player&#39;s team, the player, opponent team, category, and line
- Click "Get Player Prediction"

## Configuration Notes

- OpenAI Model
  - The code requests model "gpt-5" as a placeholder. If this model isn&#39;t available under your account, change the model string in `nfl_app.py` inside `get_enhanced_prediction` to a model you can access (e.g., `gpt-4o-mini`, `gpt-4o`, etc.).
- Current Season Behavior
  - `get_current_season()` returns `2025` to enable browsing of 2025 rosters even before the season starts. Adjust as you prefer.
- Caching
  - Streamlit cache is used with TTLs to reduce repeated network calls:
    - Teams and rosters: 1 hour
    - Weekly stats: 10 minutes
  - Clear cache via "Rerun" or Streamlit cache clearing if you need to force fresh data.

## Data Sources

- `nfl_data_py` is used to import team descriptors, rosters, schedules, and weekly player stats.
- Columns are harmonized where `nfl_data_py` might provide alternative names (e.g., `pass_yards` to `passing_yards`, `rush_td` to `rushing_tds`, etc.) to keep UIs consistent.

## Troubleshooting

- OPENAI_API_KEY not found
  - The app will stop and show a clear error if the key is missing. Ensure `.env` exists and is loaded (Streamlit runner should be launched from the project root).
- `nfl_data_py` import errors or data fetch failures
  - Make sure dependencies are installed: `pip install -r requirements.txt`
  - Network access is required to fetch data
  - Some endpoints may change over time; the app includes guards and will display warnings where possible
- Model errors
  - If the model name is invalid or you lack access, the AI prediction call will error. Update the model string as noted above.

## Security

- Do not commit your `.env`. This repo includes a `.gitignore` that excludes `.env` and common Python artifacts.
- Use `.env.example` to share required variables without secrets.

## License

No license specified by the author.

## Acknowledgments

- Data by [`nfl_data_py`](https://github.com/nflverse/nfl_data_py)
- Built with Streamlit and OpenAI
- UI components inspired by `streamlit-shadcn-ui`
