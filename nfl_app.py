# app.py - NFL Prediction Elite
import streamlit as st
import os
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import streamlit_shadcn_ui as ui
from datetime import datetime, timedelta
import nfl_data_py as nfl
from functools import lru_cache

# Load environment variables
load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    if not os.getenv('OPENAI_API_KEY'):
        st.error("OPENAI_API_KEY not found. Set it in .env or environment variables.")
        st.stop()
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")
    st.stop()

# Page Config
st.set_page_config(page_title="NFL Prediction Elite", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem;font-weight: 800;color: #013369;text-align: center;margin-bottom: 1rem;padding-bottom: 10px;border-bottom: 2px solid #D50A0A;}
    .prediction-call-over { background-color: #28a745; color: white; text-align: center; font-weight: bold; font-size: 2.5rem; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .prediction-call-under { background-color: #dc3545; color: white; text-align: center; font-weight: bold; font-size: 2.5rem; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .reasoning-text-container { padding: 10px; margin-bottom: 20px; }
    .reasoning-text { font-size: 1.1rem; line-height: 1.6; color: #333; }
    .reasoning-text strong { font-weight: bold; }
    .stSpinner > div > svg {stroke: #013369;}
    .footer {text-align: center;margin-top: 40px;padding-top: 20px;border-top: 1px solid #ddd;font-size: 0.9rem;color: #888;}
    .donation-text { font-size: 0.95rem; line-height: 1.5; margin-bottom: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px; border-left: 5px solid #013369; }
    .logo-pill {display: inline-flex; align-items: center; gap: 8px; background: #f0f2f6; padding: 4px 12px; border-radius: 15px; border: 1px solid #ddd;}
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown("""
    <div style='text-align: center; padding-top: 1rem; padding-bottom: 1rem;'>
        <h1 style='color: #013369; font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem;'>
            NFL Prediction Elite <span style="font-size: 2.5rem;">🏈</span>
        </h1>
        <p style='color: #555; font-size: 1.3rem;'>Advanced AI Insights & Statistical Analysis ⚡️</p>
    </div>
    """, unsafe_allow_html=True)

# GLOBAL CONSTANTS
STAT_MAP = {
    'Passing Yards': 'passing_yards',
    'Passing TDs': 'passing_tds', 
    'Rushing Yards': 'rushing_yards',
    'Rushing TDs': 'rushing_tds',
    'Receiving Yards': 'receiving_yards',
    'Receiving TDs': 'receiving_tds',
    'Receptions': 'receptions',
    'Fantasy Points': 'fantasy_points_ppr',
    'Total TDs': 'total_tds',
    'Pass + Rush Yards': 'pass_rush_yards'
}

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_current_season():
    now = datetime.now()
    year = now.year
    # For 2025, we want to include the upcoming season even before it starts
    # NFL season typically starts in September, but we'll show 2025 rosters
    return 2025  # Hard-coded for 2025 season support

@st.cache_data
def get_previous_season(current_season):
    return current_season - 1

# --- DATA COLLECTION FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_nfl_teams():
    """Get NFL team information"""
    try:
        teams = nfl.import_team_desc()
        teams_list = []
        for _, row in teams.iterrows():
            teams_list.append({
                'id': row['team_abbr'],
                'full_name': row['team_name'],
                'abbreviation': row['team_abbr'],
                'logo': row['team_logo_espn'],
                'color': row['team_color'],
                'color2': row['team_color2']
            })
        return sorted(teams_list, key=lambda x: x['full_name'])
    except Exception as e:
        st.error(f"Error fetching NFL teams: {e}")
        return []

@st.cache_data(ttl=3600)
def get_team_roster(season):
    """Get NFL rosters for given season"""
    try:
        rosters = nfl.import_seasonal_rosters([season])
        return rosters
    except Exception as e:
        st.warning(f"Roster fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_player_weekly_stats(seasons, player_name=None):
    """Get weekly stats for players"""
    try:
        weekly_data = nfl.import_weekly_data(seasons, downcast=True)
        
        # Add computed columns
        if 'passing_tds' not in weekly_data.columns and 'pass_td' in weekly_data.columns:
            weekly_data['passing_tds'] = weekly_data['pass_td']
        if 'rushing_tds' not in weekly_data.columns and 'rush_td' in weekly_data.columns:
            weekly_data['rushing_tds'] = weekly_data['rush_td']
        if 'receiving_tds' not in weekly_data.columns and 'rec_td' in weekly_data.columns:
            weekly_data['receiving_tds'] = weekly_data['rec_td']
        if 'passing_yards' not in weekly_data.columns and 'pass_yards' in weekly_data.columns:
            weekly_data['passing_yards'] = weekly_data['pass_yards']
        if 'rushing_yards' not in weekly_data.columns and 'rush_yards' in weekly_data.columns:
            weekly_data['rushing_yards'] = weekly_data['rush_yards']
        if 'receiving_yards' not in weekly_data.columns and 'rec_yards' in weekly_data.columns:
            weekly_data['receiving_yards'] = weekly_data['rec_yards']
        
        # Compute total TDs
        td_cols = [col for col in ['passing_tds', 'rushing_tds', 'receiving_tds'] if col in weekly_data.columns]
        if td_cols:
            weekly_data['total_tds'] = weekly_data[td_cols].fillna(0).sum(axis=1)
        
        # Compute pass + rush yards
        if 'passing_yards' in weekly_data.columns and 'rushing_yards' in weekly_data.columns:
            weekly_data['pass_rush_yards'] = weekly_data[['passing_yards', 'rushing_yards']].fillna(0).sum(axis=1)
        
        return weekly_data
    except Exception as e:
        st.warning(f"Weekly stats fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_player_stats_filtered(player_name, team_abbr, season, last_n_games=10):
    """Get filtered player stats for analysis"""
    try:
        weekly_data = get_player_weekly_stats([season])
        
        if weekly_data.empty:
            return pd.DataFrame()
        
        # Filter for specific player and team
        player_data = weekly_data[
            (weekly_data['player_display_name'].str.contains(player_name, case=False, na=False)) &
            (weekly_data['recent_team'] == team_abbr)
        ].copy()
        
        if player_data.empty:
            # Try without team filter
            player_data = weekly_data[
                weekly_data['player_display_name'].str.contains(player_name, case=False, na=False)
            ].copy()
        
        if not player_data.empty:
            # Sort by week and take last N games
            player_data = player_data.sort_values('week', ascending=False).head(last_n_games)
            player_data = player_data.sort_values('week')
            
            # Add game date if not present
            if 'game_date' not in player_data.columns and 'week' in player_data.columns:
                # Estimate game dates based on week
                player_data['game_date'] = pd.to_datetime(f'{season}-09-01') + pd.to_timedelta((player_data['week'] - 1) * 7, unit='D')
        
        return player_data
    except Exception as e:
        st.warning(f"Player stats filtering failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_team_players(team_abbr, season):
    """Get players for a specific team"""
    try:
        roster = get_team_roster(season)
        if roster.empty:
            return []
        
        team_players = roster[roster['team'] == team_abbr]
        if team_players.empty:
            # Try with recent_team column
            team_players = roster[roster.get('recent_team', '') == team_abbr]
        
        players = []
        for _, row in team_players.iterrows():
            players.append({
                'id': row.get('player_id', ''),
                'full_name': row.get('player_name', row.get('display_name', 'Unknown')),
                'position': row.get('position', 'N/A')
            })
        
        return sorted(players, key=lambda x: x['full_name'])
    except Exception as e:
        st.warning(f"Team players fetch failed: {e}")
        return []

def build_prediction_data_package(player_info=None, player_team_info=None, opponent_team_info=None, 
                                category=None, prediction_value=None, season=None, last_n_games_context=10):
    """Build comprehensive data package for AI prediction"""
    
    dp = {
        'category': category,
        'prediction_value': prediction_value,
        'season': season,
        'last_n_games_context': last_n_games_context
    }
    
    if player_info and player_team_info:
        dp['player_info'] = player_info
        dp['player_team_info'] = player_team_info
        
        # Get player stats
        player_stats_df = get_player_stats_filtered(
            player_info['full_name'], 
            player_team_info['abbreviation'], 
            season, 
            last_n_games_context * 2  # Get more data for analysis
        )
        
        dp['player_stats_df'] = player_stats_df
        
        if not player_stats_df.empty:
            recent_stats = player_stats_df.head(last_n_games_context)
            stat_col = STAT_MAP.get(category)
            
            if stat_col and stat_col in recent_stats.columns:
                perf = {
                    'games_played': len(recent_stats),
                    f'avg_{stat_col}': recent_stats[stat_col].mean(),
                    f'max_{stat_col}': recent_stats[stat_col].max(),
                    f'min_{stat_col}': recent_stats[stat_col].min(),
                    f'std_{stat_col}': recent_stats[stat_col].std(),
                    'hit_rate_over_line': (recent_stats[stat_col] > prediction_value).mean() if prediction_value else 0
                }
            else:
                perf = {'games_played': len(recent_stats)}
            
            dp['player_recent_performance'] = perf
        
        if opponent_team_info:
            dp['opponent_team_info'] = opponent_team_info
    
    return dp

def get_enhanced_prediction(data_package):
    """Get AI prediction using OpenAI"""
    # Check if we have statistical data
    has_stats = (data_package.get('player_stats_df') is not None and 
                not data_package.get('player_stats_df').empty if data_package.get('player_stats_df') is not None else False)
    
    if has_stats:
        system_prompt = """You are NFLPredictor-Elite, a world-class NFL analytics AI. Your expertise lies in synthesizing diverse data points to make informed predictions on player and team statistics.

ROLE: Analyze player performance trends, matchup dynamics, opponent strengths/weaknesses, and contextual factors (injuries, weather, rest). Provide a clear, concise, data-driven prediction.

OUTPUT FORMAT: Strictly adhere to one of the following formats: "OVER: [Detailed explanation...]" or "UNDER: [Detailed explanation...]"

GUIDELINES: 
- Base reasoning *only* on data provided
- Mention key stats and trends
- Be objective and analytical
- Consider recent form, matchup history, and situational factors
- Keep explanation under 200 words
"""
    else:
        system_prompt = """You are NFLPredictor-Elite, a world-class NFL analytics AI specializing in preseason and early season predictions.

ROLE: Since no statistical data is available for this player in the current season, provide a prediction based on player position, team context, and general NFL trends. Consider typical performance ranges for the player's position and role.

OUTPUT FORMAT: Strictly adhere to one of the following formats: "OVER: [Detailed explanation...]" or "UNDER: [Detailed explanation...]"

GUIDELINES: 
- Acknowledge the lack of current season data
- Use position-based expectations and league averages
- Consider team offensive scheme and usage patterns
- Mention that this is a preseason projection
- Be conservative and objective
- Keep explanation under 200 words
"""

    # Build user prompt
    user_prompt_parts = ["PREDICTION REQUEST:"]
    
    if data_package.get('player_info') and data_package.get('category'):
        user_prompt_parts.extend([
            f"Player: {data_package['player_info']['full_name']} ({data_package['player_info']['position']})",
            f"Team: {data_package['player_team_info']['full_name']}",
            f"Category: {data_package['category']}",
            f"Line: {data_package['prediction_value']}",
            f"Season: {data_package['season']}"
        ])
        
        if data_package.get('opponent_team_info'):
            user_prompt_parts.append(f"Opponent: {data_package['opponent_team_info']['full_name']}")
        
        # Add recent performance data
        if data_package.get('player_recent_performance'):
            perf = data_package['player_recent_performance']
            user_prompt_parts.extend([
                "",
                "RECENT PERFORMANCE:",
                f"Games analyzed: {perf.get('games_played', 0)}",
            ])
            
            stat_col = STAT_MAP.get(data_package['category'])
            if stat_col:
                avg_key = f'avg_{stat_col}'
                max_key = f'max_{stat_col}'
                min_key = f'min_{stat_col}'
                
                if avg_key in perf:
                    user_prompt_parts.append(f"Average {data_package['category']}: {perf[avg_key]:.1f}")
                if max_key in perf:
                    user_prompt_parts.append(f"Best game: {perf[max_key]:.1f}")
                if min_key in perf:
                    user_prompt_parts.append(f"Worst game: {perf[min_key]:.1f}")
                if 'hit_rate_over_line' in perf:
                    user_prompt_parts.append(f"Hit rate over {data_package['prediction_value']}: {perf['hit_rate_over_line']:.1%}")
    
    user_prompt = "\n".join(user_prompt_parts)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            #max_tokens=300,
            #temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI Prediction Error: {e}")
        return "Error: Could not retrieve prediction."

def display_performance_trends(stats_df, category, prediction_line=None, player_name="Player"):
    """Display performance trend chart"""
    if stats_df is None or stats_df.empty:
        st.info(f"No recent game data for {player_name}")
        return
    
    stat_col = STAT_MAP.get(category)
    if not stat_col or stat_col not in stats_df.columns:
        st.warning(f"Data for '{category}' not found for {player_name}")
        return
    
    plot_df = stats_df.dropna(subset=[stat_col]).copy()
    if plot_df.empty:
        st.warning(f"No valid data to plot for '{category}' for {player_name}")
        return
    
    # Create date labels
    if 'week' in plot_df.columns:
        date_labels = [f"Week {int(week)}" for week in plot_df['week']]
    else:
        date_labels = [f"Game {i+1}" for i in range(len(plot_df))]
    
    x_vals = range(len(plot_df))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, plot_df[stat_col], marker='o', linestyle='-', linewidth=2, 
            markersize=8, label=f'Actual {category}')
    
    # Add average line
    avg = plot_df[stat_col].mean()
    ax.axhline(y=avg, color='dodgerblue', linestyle='--', alpha=0.8, 
               label=f'Avg ({len(plot_df)}g): {avg:.1f}')
    
    # Add prediction line if provided
    if prediction_line is not None:
        ax.axhline(y=prediction_line, color='red', linestyle=':', linewidth=2, 
                   label=f'Line: {prediction_line}')
    
    # Add rolling average if enough data
    if len(plot_df) >= 3:
        rolling_avg = plot_df[stat_col].rolling(3, min_periods=1).mean()
        ax.plot(x_vals, rolling_avg, color='green', linestyle='-.', alpha=0.7, 
                label='3-Game Rolling Avg')
    
    ax.set_title(f'{player_name} - {category} Trend (Last {len(plot_df)} Games)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel(category, fontsize=12)
    ax.set_xlabel('Games', fontsize=12)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def display_stat_distribution(stats_df, category, prediction_line=None, player_name="Player"):
    """Display statistical distribution histogram"""
    if stats_df is None or stats_df.empty:
        return
    
    stat_col = STAT_MAP.get(category)
    if not stat_col or stat_col not in stats_df.columns:
        return
    
    plot_data = stats_df[stat_col].dropna()
    if plot_data.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    num_bins = max(5, min(15, len(plot_data) // 2)) if len(plot_data) > 10 else max(3, len(plot_data.unique()))
    ax.hist(plot_data, bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7, 
            label=f'{category} Distribution')
    
    # Add average line
    avg = plot_data.mean()
    ax.axvline(avg, color='dodgerblue', linestyle='--', linewidth=2, 
               label=f'Average: {avg:.1f}')
    
    # Add prediction line if provided
    if prediction_line is not None:
        ax.axvline(prediction_line, color='red', linestyle=':', linewidth=2, 
                   label=f'Line: {prediction_line}')
        
        # Calculate over/under percentages
        over_pct = (plot_data > prediction_line).mean() * 100
        under_pct = (plot_data <= prediction_line).mean() * 100
        
        ax.text(0.02, 0.98, f'Over {prediction_line}: {over_pct:.1f}%\nUnder {prediction_line}: {under_pct:.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'{player_name} - {category} Distribution (Last {len(stats_df)} Games)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel(category, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# --- Streamlit UI ---
st.sidebar.title("Prediction Controls ⚙️")

# Get current season
current_season = get_current_season()

# Season selector with 2025 as the primary option
season_options = [2025, 2024, 2023, 2022]
season = st.sidebar.selectbox("Season", 
                             season_options, 
                             key="season_select")

# Add season info
if season == 2025:
    st.sidebar.info("🏈 2025 Season: Rosters available, stats will be updated as games are played")
elif season >= 2024:
    st.sidebar.info("📊 Current season with live statistics")
else:
    st.sidebar.info("📈 Historical season data")

# Get NFL teams
teams_list = get_nfl_teams()
if not teams_list:
    st.error("Failed to load NFL teams. Please check your connection.")
    st.stop()

team_options_map = {team['full_name']: team for team in teams_list}
team_names = list(team_options_map.keys())

# Analysis type selector
analysis_type = st.sidebar.radio("Analysis Type:", 
                                ("Player Props", "Team Matchup", "League Insights"), 
                                key="main_analysis_type_radio")

# Initialize variables
sel_player_info = None
sel_team_player = None
sel_opp_team = None
prop_cat = None
prop_ln = None

if analysis_type == "Player Props":
    st.sidebar.header("Player Prop Settings")
    
    # Team selection
    sel_team_player = st.sidebar.selectbox("Player's Team", team_names, key="sb_player_team_select")
    
    if sel_team_player:
        player_team_details = team_options_map[sel_team_player]
        
        # Get team players
        players = get_team_players(player_team_details['abbreviation'], season)
        
        if players:
            player_options_map = {p['full_name']: p for p in players}
            sel_player_name = st.sidebar.selectbox("Player", list(player_options_map.keys()), 
                                                  key="sb_player_name_select")
            
            if sel_player_name:
                sel_player_info = player_options_map[sel_player_name]
        else:
            st.sidebar.warning(f"No roster data found for {sel_team_player} in {season}")
    
    # Opponent selection
    available_opponents = [name for name in team_names if name != sel_team_player]
    sel_opp_team = st.sidebar.selectbox("Opponent Team", available_opponents, 
                                       key="sb_opp_team_select")
    
    # Prop category
    prop_cat = st.sidebar.selectbox("Prop Category", list(STAT_MAP.keys()), 
                                   key="sb_prop_cat_select")
    
    # Prop line
    prop_ln = st.sidebar.number_input("Prop Line", value=100.5, step=0.5, 
                                     key="sb_prop_line_input")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ❤️ Support This App!")
st.sidebar.markdown(f"""<div class='donation-text'>
This NFL Prediction Elite app provides advanced AI-powered predictions and statistical analysis for entertainment purposes. 
Your support helps keep this service free and continuously improved!
</div>""", unsafe_allow_html=True)

# Main content area
if analysis_type == "Player Props":
    if sel_player_info:
        st.header(f"📊 NFL Player Prop Analysis: {sel_player_info['full_name']} ({sel_player_info['position']})")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Team:** {sel_team_player}")
            if sel_opp_team:
                st.markdown(f"**Opponent:** {sel_opp_team}")
            st.markdown(f"**Category:** {prop_cat}")
            st.markdown(f"**Line:** {prop_ln}")
        
        with col2:
            team_info = team_options_map[sel_team_player]
            st.markdown(f"""
                <div class='logo-pill'>
                    <img src='{team_info['logo']}' width='30'/> 
                    <b>{team_info['abbreviation']}</b>
                </div>
                """, unsafe_allow_html=True)
        
        if ui.button("🔮 Get Player Prediction", key="btn_get_player_pred_main"):
            player_team_info = team_options_map[sel_team_player]
            opponent_team_info = team_options_map[sel_opp_team] if sel_opp_team else None
            
            with st.spinner(f"🧠 Analyzing {sel_player_info['full_name']}..."):
                # Build data package
                dp = build_prediction_data_package(
                    player_info=sel_player_info,
                    player_team_info=player_team_info,
                    opponent_team_info=opponent_team_info,
                    category=prop_cat,
                    prediction_value=prop_ln,
                    season=season
                )
                
                # Get AI prediction
                pred_text = get_enhanced_prediction(dp)
                
                # Display prediction
                st.subheader("🤖 AI Prediction")
                
                final_call, reasoning_text = "", pred_text
                if pred_text.upper().startswith("OVER:"):
                    final_call, reasoning_text = "OVER", pred_text[len("OVER:"):].strip()
                elif pred_text.upper().startswith("UNDER:"):
                    final_call, reasoning_text = "UNDER", pred_text[len("UNDER:"):].strip()
                
                if final_call == "OVER":
                    st.markdown(f"<div class='prediction-call-over'>{final_call}</div>", unsafe_allow_html=True)
                elif final_call == "UNDER":
                    st.markdown(f"<div class='prediction-call-under'>{final_call}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div>{pred_text}</div>", unsafe_allow_html=True)
                
                if reasoning_text and final_call:
                    st.markdown(f"""<div class='reasoning-text-container'>
                        <p class='reasoning-text' style='color: white;'><strong>Reasoning:</strong> {reasoning_text}</p>
                    </div>""", unsafe_allow_html=True)
                
                # Display performance analysis
                player_stats_df = dp.get('player_stats_df')
                if player_stats_df is not None and not player_stats_df.empty:
                    st.subheader(f"📈 Performance Analysis: {sel_player_info['full_name']}")
                    
                    # Recent performance metrics
                    recent_perf = dp.get('player_recent_performance', {})
                    if recent_perf.get('games_played', 0) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        stat_col = STAT_MAP.get(prop_cat, '')
                        avg_key = f'avg_{stat_col}'
                        max_key = f'max_{stat_col}'
                        
                        with col1:
                            games = recent_perf.get('games_played', 0)
                            st.metric("Games Analyzed", games)
                        
                        with col2:
                            if avg_key in recent_perf:
                                avg_val = recent_perf[avg_key]
                                st.metric(f"Avg {prop_cat}", f"{avg_val:.1f}")
                        
                        with col3:
                            if max_key in recent_perf:
                                max_val = recent_perf[max_key]
                                st.metric("Season High", f"{max_val:.1f}")
                        
                        with col4:
                            hit_rate = recent_perf.get('hit_rate_over_line', 0)
                            st.metric(f"Hit Rate (>{prop_ln})", f"{hit_rate:.1%}")
                    
                    # Performance charts
                    st.subheader("📊 Performance Trends")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Game-by-Game Trend**")
                        display_performance_trends(player_stats_df, prop_cat, prop_ln, sel_player_info['full_name'])
                    
                    with col2:
                        st.markdown("**Statistical Distribution**")
                        display_stat_distribution(player_stats_df, prop_cat, prop_ln, sel_player_info['full_name'])
                    
                    # Raw data table
                    st.subheader("📋 Recent Game Log")
                    display_cols = ['week']
                    if 'opponent_team' in player_stats_df.columns:
                        display_cols.append('opponent_team')
                    
                    stat_col = STAT_MAP.get(prop_cat)
                    if stat_col and stat_col in player_stats_df.columns:
                        display_cols.append(stat_col)
                    
                    # Add other relevant columns
                    for col in ['fantasy_points', 'fantasy_points_ppr']:
                        if col in player_stats_df.columns:
                            display_cols.append(col)
                    
                    # Filter columns that exist
                    display_cols = [col for col in display_cols if col in player_stats_df.columns]
                    
                    if display_cols:
                        display_df = player_stats_df[display_cols].head(10).copy()
                        if stat_col and stat_col in display_df.columns:
                            # Highlight games over the line
                            def highlight_over(val):
                                if pd.isna(val):
                                    return ''
                                try:
                                    return 'background-color: lightgreen' if float(val) > prop_ln else ''
                                except:
                                    return ''
                            
                            styled_df = display_df.style.applymap(highlight_over, subset=[stat_col])
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info("No game log data available to display")
                        
                else:
                    st.warning(f"No detailed stats available for {sel_player_info['full_name']} in {season}")
    else:
        st.info("⬅️ Select a player from the sidebar to begin analysis")

elif analysis_type == "Team Matchup":
    st.header("⚔️ NFL Team Matchup Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team_name = st.selectbox("Home Team", team_names, key="home_team_select")
        home_team_info = team_options_map[home_team_name]
        st.markdown(f"""
            <div class='logo-pill'>
                <img src='{home_team_info['logo']}' width='25'/> 
                <b>{home_team_info['abbreviation']}</b> - {home_team_name}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        available_away = [name for name in team_names if name != home_team_name]
        away_team_name = st.selectbox("Away Team", available_away, key="away_team_select")
        away_team_info = team_options_map[away_team_name]
        st.markdown(f"""
            <div class='logo-pill'>
                <img src='{away_team_info['logo']}' width='25'/> 
                <b>{away_team_info['abbreviation']}</b> - {away_team_name}
            </div>
            """, unsafe_allow_html=True)
    
    # Team comparison metrics
    st.subheader("🏈 Team Comparison")
    
    try:
        # Get team performance data
        weekly_data = get_player_weekly_stats([season])
        
        if not weekly_data.empty:
            home_team_data = weekly_data[weekly_data['recent_team'] == home_team_info['abbreviation']]
            away_team_data = weekly_data[weekly_data['recent_team'] == away_team_info['abbreviation']]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Offensive Stats**")
                
                # Calculate team offensive averages
                if not home_team_data.empty and not away_team_data.empty:
                    stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards']
                    existing_cols = [col for col in stat_cols if col in weekly_data.columns]
                    
                    if existing_cols:
                        home_off = home_team_data.groupby('week')[existing_cols].sum().mean()
                        away_off = away_team_data.groupby('week')[existing_cols].sum().mean()
                        
                        comparison_data = pd.DataFrame({
                            home_team_info['abbreviation']: home_off,
                            away_team_info['abbreviation']: away_off
                        }).fillna(0)
                        
                        st.dataframe(comparison_data, use_container_width=True)
                    else:
                        st.info("Offensive stats not available")
            
            with col2:
                st.markdown("**Scoring Stats**")
                
                if not home_team_data.empty and not away_team_data.empty:
                    td_cols = ['passing_tds', 'rushing_tds', 'receiving_tds']
                    existing_td_cols = [col for col in td_cols if col in weekly_data.columns]
                    
                    if existing_td_cols:
                        home_scoring = home_team_data.groupby('week')[existing_td_cols].sum().mean()
                        away_scoring = away_team_data.groupby('week')[existing_td_cols].sum().mean()
                        
                        scoring_data = pd.DataFrame({
                            home_team_info['abbreviation']: home_scoring,
                            away_team_info['abbreviation']: away_scoring
                        }).fillna(0)
                        
                        st.dataframe(scoring_data, use_container_width=True)
                    else:
                        st.info("Scoring stats not available")
            
            with col3:
                st.markdown("**Fantasy Production**")
                
                if not home_team_data.empty and not away_team_data.empty:
                    fantasy_cols = [col for col in ['fantasy_points', 'fantasy_points_ppr'] if col in weekly_data.columns]
                    if fantasy_cols:
                        home_fantasy = home_team_data.groupby('week')[fantasy_cols].sum().mean()
                        away_fantasy = away_team_data.groupby('week')[fantasy_cols].sum().mean()
                        
                        fantasy_data = pd.DataFrame({
                            home_team_info['abbreviation']: home_fantasy,
                            away_team_info['abbreviation']: away_fantasy
                        }).fillna(0)
                        
                        st.dataframe(fantasy_data, use_container_width=True)
                    else:
                        st.info("Fantasy stats not available")
        
        else:
            st.info("No team data available for comparison")
            
    except Exception as e:
        st.warning(f"Team comparison data unavailable: {e}")

elif analysis_type == "League Insights":
    st.header("🌍 NFL League-Wide Insights")
    
    try:
        weekly_data = get_player_weekly_stats([season])
        
        if not weekly_data.empty:
            st.subheader("🏆 Statistical Leaders")
            
            # Create leader boards for different categories
            leader_categories = {
                'Passing Yards': 'passing_yards',
                'Rushing Yards': 'rushing_yards', 
                'Receiving Yards': 'receiving_yards',
                'Total TDs': 'total_tds',
                'Receptions': 'receptions'
            }
            
            col1, col2 = st.columns(2)
            
            for i, (category, stat_col) in enumerate(leader_categories.items()):
                if stat_col in weekly_data.columns:
                    # Aggregate by player
                    player_totals = weekly_data.groupby(['player_display_name', 'position', 'recent_team'])[stat_col].sum().reset_index()
                    top_players = player_totals.nlargest(10, stat_col)
                    
                    with col1 if i % 2 == 0 else col2:
                        st.markdown(f"**{category} Leaders**")
                        display_df = top_players[['player_display_name', 'position', 'recent_team', stat_col]].copy()
                        display_df.columns = ['Player', 'Pos', 'Team', category]
                        st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # League averages
            st.subheader("📊 League Averages")
            
            avg_stats = {}
            for category, stat_col in leader_categories.items():
                if stat_col in weekly_data.columns:
                    avg_stats[category] = weekly_data[stat_col].mean()
            
            if avg_stats:
                avg_df = pd.DataFrame(list(avg_stats.items()), columns=['Statistic', 'League Average'])
                avg_df['League Average'] = avg_df['League Average'].round(2)
                st.dataframe(avg_df, hide_index=True, use_container_width=True)
        
        else:
            st.info("No league data available for the selected season")
            
    except Exception as e:
        st.warning(f"League insights unavailable: {e}")

# Footer
st.markdown("<hr style='margin-top: 30px; margin-bottom:10px;'>", unsafe_allow_html=True)
st.markdown(f"""<div class='footer'>
    © {datetime.now().year} NFL Prediction Elite. For entertainment purposes only.
    <br>Powered by Magic • app by DeVReV27 
</div>""", unsafe_allow_html=True)
