import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import secrets
import string
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client
from xgboost import XGBClassifier

# --- HELPERS ---
def extract_data(resp):
    """Safely extract data from Supabase response regardless of return type."""
    if resp is None:
        return None
    # Prefer .data attribute for APIResponse objects
    if hasattr(resp, "data"):
        return resp.data
    # Use .get() ONLY if it's a dictionary
    if isinstance(resp, dict):
        return resp.get("data")
    return None

def normalize_text_series(series):
    return series.fillna("").astype(str).str.strip()

def get_supabase_client(session=None, access_token=None): # Added access_token=None
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        
        # Priority: use manual access_token if provided, otherwise use session
        token = access_token
        if not token and session and hasattr(session, 'access_token'):
            token = session.access_token
            
        if token:
            client = create_client(url, key)
            client.postgrest.auth(token)
            return client
        
        return create_client(url, key)
    return None

# --- SUPABASE CONNECTION ---
supabase_status = "Disconnected"
supabase: Client = None

try:
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        # Create a persistent client instance
        supabase = create_client(url, key)
        supabase_status = "Connected"
    else:
        supabase_status = "Missing Credentials"
except Exception as e:
    supabase_status = f"Error: {str(e)}"

# --- CONFIG & THEME ---
st.set_page_config(page_title="Fantasy XI - Pakistan Cricket League", layout="wide", page_icon="🏏")

# --- INITIALIZE SESSION STATE ---
if 'user' not in st.session_state:
    st.session_state.user = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'usage_left' not in st.session_state:
    st.session_state.usage_left = 3
if 'is_pro' not in st.session_state:
    st.session_state.is_pro = False
if 'auth_view' not in st.session_state:
    st.session_state.auth_view = "login"

# --- PERSIST AUTH CONTEXT ---
if st.session_state.access_token and supabase:
    supabase.postgrest.auth(st.session_state.access_token)

# --- VALIDATION LOGIC ---
def validate_identifier(identifier):
    # Mobile: 11 digits max starting with 03xx
    if re.match(r'^03\d{9}$', identifier):
        return True
    # Landline: 10 digits max starting with 051xxx
    if re.match(r'^051\d{7}$', identifier):
        return True
    # Standard Email
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', identifier):
        return True
    return False

def validate_email(email):
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def validate_phone(phone):
    # Mobile (11 digits max) starts with 03xx, Landline (10 digits max) starts with 051xxx
    if re.match(r'^03\d{9}$', phone):
        return True
    if re.match(r'^051\d{7}$', phone):
        return True
    return False

# Premium UI Polish
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .premium-box, .prediction-card, [data-testid="stMetric"] {
        animation: fadeIn 0.6s ease-out;
    }

    /* Result Badges */
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        display: inline-block;
        margin-left: 10px;
    }
    .badge-runs { background-color: #059669; border: 1px solid #10b981; }
    .badge-wickets { background-color: #2563eb; border: 1px solid #3b82f6; }

    /* Impact Players Polish */
    .player-headshot {
        border-radius: 50%;
        border: 3px solid #38bdf8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
        margin-bottom: 15px;
    }

    /* Standard Components */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; }

    .prediction-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 30px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .prediction-card h1 { color: #38bdf8; margin: 0; font-size: 2.5rem; }

    .premium-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }

    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    .stButton>button {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: 800 !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0f172a;
        color: #475569;
        text-align: center;
        padding: 10px;
        font-size: 11px;
        border-top: 1px solid #1e293b;
        z-index: 1000;
    }
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        color: #94a3b8;
        font-size: 0.85rem;
    }
</style>
<div class="footer">
    <b>INDEPENDENT FAN PORTAL:</b> Not affiliated with, endorsed by, or associated with the PSL or PCB. PRO CRICKET INSIGHTS © 2026
</div>
""", unsafe_allow_html=True)

# --- DATA LOADING (MIGRATED TO SUPABASE) ---
@st.cache_data(show_spinner="Analyzing match archives...")
def load_data():
    if supabase_status != "Connected":
        st.error(f"Supabase Connection Failed: {supabase_status}")
        st.stop()
    
    # Updated to pass access_token if it exists in session state
    client = get_supabase_client(session=None, access_token=st.session_state.get("access_token"))
    if client is None:
        st.error("Supabase client not available")
        st.stop()
        
    try:
        # FETCH MATCHES
        matches_res = client.table("matches").select("*").execute()
        matches_data = extract_data(matches_res)
        matches = pd.DataFrame(matches_data if matches_data else [])
        
        if matches.empty:
            st.error("Matches table is empty.")
            st.stop()
            
        # FETCH BALL BY BALL (FIXED PAGINATION)
        all_balls = []
        batch_size = 1000 
        start = 0
        
        # Progress bar for UX
        progress_text = "Downloading ball-by-ball data..."
        progress_bar = st.sidebar.progress(0, text=progress_text)
        
        while True:
            response = client.table("ball_by_ball") \
                .select("""
                    id, match_id, innings, batting_team, bowling_team, 
                    over, ball, batter, bowler, non_striker, 
                    runs_batter, runs_extras, runs_total, wide, noball, 
                    bye, legbye, is_wicket, player_out, wicket_kind, 
                    fielders, review_by, review_outcome, concussion_sub, 
                    season, match_date_ts
                """) \
                .range(start, start + batch_size - 1) \
                .execute()
            
            batch = extract_data(response)
            if not batch:
                break
            
            all_balls.extend(batch)
            if len(batch) < batch_size:
                break
            
            start += batch_size
            prog = min(start / 150000, 0.99)
            progress_bar.progress(prog, text=f"Fetched {len(all_balls)} records...")
            
        progress_bar.empty()
        balls = pd.DataFrame(all_balls if all_balls else [])
        
        if balls.empty:
            st.error("Ball_by_ball table is empty.")
            st.stop()

        # STANDARDIZATION
        for df in [matches, balls]:
            df['match_id'] = pd.to_numeric(df['match_id'], errors='coerce').astype('Int64')
            if 'season' in df.columns:
                df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')

        matches['date'] = pd.to_datetime(matches.get('date'), errors='coerce')
        matches['venue'] = matches.get('venue', '').astype(str).str.split(',').str[0]

        # Numeric conversion
        numeric_cols = [
            'runs_batter', 'runs_extras', 'runs_total', 'wide', 
            'noball', 'is_wicket', 'innings', 'over', 'ball', 'season', 'match_id'
        ]
        for col in numeric_cols:
            if col in balls.columns:
                balls[col] = pd.to_numeric(balls[col], errors='coerce').fillna(0)

        venue_map = matches.set_index('match_id')['venue'].to_dict()
        balls['venue'] = balls['match_id'].map(venue_map).fillna("Unknown Venue")

        # Text normalization
        text_cols = [
            "wicket_kind", "batter", "bowler", "non_striker", 
            "player_out", "fielders", "batting_team", "bowling_team"
        ]
        for c in text_cols:
            if c in balls.columns:
                balls[c] = normalize_text_series(balls[c])
                if c == "wicket_kind":
                    balls[c] = balls[c].replace({"": None})
                    balls[c] = balls[c].where(balls[c].isnull(), balls[c].str.lower().str.strip())

        unmapped_v = balls[balls['venue'] == "Unknown Venue"].shape[0]
        if unmapped_v > 0:
            st.sidebar.info(f"ℹ️ {unmapped_v} balls are missing match details.")

        # Corrected: Create list of players and return 3 items
        all_players_list = sorted(list(set(balls['batter'].unique()) | set(balls['bowler'].unique())))
        return matches, balls, all_players_list

    except Exception as e:
        st.error(f"Data process failed: {e}")
        st.stop()

# Initialize data correctly using the returned triple
matches_df, balls_df, all_players = load_data()


# --- ML MODEL ENGINE ---
@st.cache_resource
def train_ml_model(df):
    # Ensure date column is datetime64 for proper sorting and comparison
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    initial_rows = len(df)
    model_df = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'date']].dropna()
    model_df = model_df.sort_values('date')
    
    dropped = initial_rows - len(model_df)
    if dropped > 0:
        st.sidebar.info(f"ℹ️ ML model training on {len(model_df)} matches (dropped {dropped} with missing data).")

    def get_h2h_win_rate(t1, t2, date):
        relevant = df[
            (df['date'] < date) & 
            (
                ((df['team1'] == t1) & (df['team2'] == t2)) | 
                ((df['team1'] == t2) & (df['team2'] == t1))
            )
        ]
        return len(relevant[relevant['winner'] == t1]) / len(relevant) if len(relevant) > 0 else 0.5

    def get_venue_win_rate(team, venue, date):
        relevant = df[
            (df['date'] < date) & (df['venue'] == venue) & 
            ((df['team1'] == team) | (df['team2'] == team))
        ]
        return len(relevant[relevant['winner'] == team]) / len(relevant) if len(relevant) > 0 else 0.5

    def get_recent_form(team, date):
        relevant = df[
            (df['date'] < date) & ((df['team1'] == team) | (df['team2'] == team))
        ].sort_values('date', ascending=False).head(5)
        return len(relevant[relevant['winner'] == team]) / len(relevant) if len(relevant) > 0 else 0.5

    model_df['h2h'] = model_df.apply(lambda x: get_h2h_win_rate(x['team1'], x['team2'], x['date']), axis=1)
    model_df['v_t1'] = model_df.apply(lambda x: get_venue_win_rate(x['team1'], x['venue'], x['date']), axis=1)
    model_df['v_t2'] = model_df.apply(lambda x: get_venue_win_rate(x['team2'], x['venue'], x['date']), axis=1)
    model_df['form_t1'] = model_df.apply(lambda x: get_recent_form(x['team1'], x['date']), axis=1)
    model_df['form_t2'] = model_df.apply(lambda x: get_recent_form(x['team2'], x['date']), axis=1)

    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    le_decision = LabelEncoder()

    all_teams = pd.concat([model_df['team1'], model_df['team2']]).unique()
    le_team.fit(all_teams)
    le_venue.fit(model_df['venue'].unique())
    le_decision.fit(model_df['toss_decision'].unique())

    X = pd.DataFrame({
        'team1': le_team.transform(model_df['team1']),
        'team2': le_team.transform(model_df['team2']),
        'venue': le_venue.transform(model_df['venue']),
        'toss_winner': le_team.transform(model_df['toss_winner']),
        'toss_decision': le_decision.transform(model_df['toss_decision']),
        'h2h': model_df['h2h'],
        'v_t1': model_df['v_t1'],
        'v_t2': model_df['v_t2'],
        'form_t1': model_df['form_t1'],
        'form_t2': model_df['form_t2']
    })
    y = (model_df['winner'] == model_df['team1']).astype(int)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    return model, le_team, le_venue, le_decision

# --- ANALYTICS ENGINES ---
def get_batting_stats(df):
    if df.empty: return pd.DataFrame()
    bat = df.groupby('batter').agg({
        'runs_batter': 'sum', 
        'ball': 'count', 
        'wide': 'sum', 
        'match_id': 'nunique', 
        'is_wicket': 'sum'
    }).reset_index()
    bat['balls_faced'] = bat['ball'] - bat['wide']
    bat['strike_rate'] = (bat['runs_batter'] / (bat['balls_faced'].replace(0, 1)) * 100).round(2)
    f = df[df['runs_batter'] == 4].groupby('batter').size().reset_index(name='4s')
    s = df[df['runs_batter'] == 6].groupby('batter').size().reset_index(name='6s')
    res = bat.merge(f, on='batter', how='left').merge(s, on='batter', how='left').fillna(0)
    for col in ['runs_batter', 'balls_faced', '4s', '6s', 'match_id']:
        res[col] = res[col].astype(int)
    return res.sort_values('runs_batter', ascending=False)

def get_bowling_stats(df):
    if df.empty: return pd.DataFrame()
    bw = df[~df['wicket_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
    wickets = bw.groupby('bowler')['is_wicket'].sum().reset_index().rename(columns={'is_wicket': 'wickets'})
    df_c = df.copy()
    df_c['rc'] = df_c['runs_batter'] + df_c['wide'] + df_c['noball']
    runs = df_c.groupby('bowler')['rc'].sum().reset_index()
    bls = df_c[(df_c['wide'] == 0) & (df_c['noball'] == 0)].groupby('bowler').size().reset_index(name='balls')
    bowling = wickets.merge(runs, on='bowler').merge(bls, on='bowler')
    bowling['economy'] = (bowling['rc'] / (bowling['balls'].replace(0, 1) / 6)).round(2)
    for col in ['wickets', 'rc', 'balls']:
        bowling[col] = bowling[col].astype(int)
    return bowling.sort_values('wickets', ascending=False)

def get_inning_scorecard(df, innings_no):
    id_df = df[df['innings'] == innings_no].copy()
    if id_df.empty: return None, None
    b_stats = id_df.groupby('batter').agg({'runs_batter': 'sum', 'ball': 'count', 'wide': 'sum'}).reset_index()
    b_stats['B'] = (b_stats['ball'] - b_stats['wide']).astype(int)
    b_stats['SR'] = (b_stats['runs_batter'] / (b_stats['B'].replace(0, 1)) * 100).round(1)
    f = id_df[id_df['runs_batter'] == 4].groupby('batter').size().reset_index(name='4s')
    s = id_df[id_df['runs_batter'] == 6].groupby('batter').size().reset_index(name='6s')
    bat = b_stats.merge(f, on='batter', how='left').merge(s, on='batter', how='left').fillna(0)
    bat['runs_batter'] = bat['runs_batter'].astype(int); bat['4s'] = bat['4s'].astype(int); bat['6s'] = bat['6s'].astype(int)
    
    bw = id_df[~id_df['wicket_kind'].isin(['run out', 'retired hurt'])]
    w = bw.groupby('bowler')['is_wicket'].sum().reset_index().rename(columns={'is_wicket':'W'})
    id_df['rc_temp'] = id_df['runs_batter'] + id_df['wide'] + id_df['noball']
    r = id_df.groupby('bowler')['rc_temp'].sum().reset_index()
    bls = id_df[(id_df['wide']==0) & (id_df['noball']==0)].groupby('bowler').size().reset_index(name='bls')
    bowl = w.merge(r, on='bowler').merge(bls, on='bowler')
    bowl['O'] = ((bowl['bls']//6) + (bowl['bls']%6/10)).round(1)
    bowl['Econ'] = (bowl['rc_temp']/(bowl['bls'].replace(0,1)/6)).round(1)
    bowl['W'] = bowl['W'].astype(int); bowl['rc_temp'] = bowl['rc_temp'].astype(int)
    return bat[['batter', 'runs_batter', 'B', '4s', '6s', 'SR']], bowl[['bowler', 'O', 'rc_temp', 'W', 'Econ']]

# --- FANTASY LEAGUE ENGINE ---
FANTASY_POINTS = {
    "runs": 1, "four": 1, "six": 2, "wicket": 25,
    "three_wicket_bonus": 8, "four_wicket_bonus": 16, "five_wicket_bonus": 32,
    "captain_multiplier": 2.0, "vice_captain_multiplier": 1.5,
}

def get_player_roles(balls_df):
    bat_stats = balls_df.groupby("batter").agg(bat_balls=("ball", "count"), bat_runs=("runs_batter", "sum")).reset_index()
    bowl_stats = balls_df.groupby("bowler").agg(bowl_balls=("ball", "count"), bowl_wickets=("is_wicket", "sum")).reset_index()
    bat_stats = bat_stats.rename(columns={"batter": "player"})
    bowl_stats = bowl_stats.rename(columns={"bowler": "player"})
    merged = bat_stats.merge(bowl_stats, on="player", how="outer").fillna(0)
    merged["bat_balls"], merged["bowl_balls"] = merged["bat_balls"].astype(int), merged["bowl_balls"].astype(int)

    def assign_role(row):
        bat_b, bowl_b = row["bat_balls"], row["bowl_balls"]
        if bat_b >= 50 and bowl_b < 30: return "BAT"
        if bowl_b >= 60 and bat_b < 30: return "BWL"
        if bat_b >= 30 and bowl_b >= 30: return "AR"
        return "BAT" if bat_b > bowl_b else "BWL"

    merged["role"] = merged.apply(assign_role, axis=1)
    return merged.set_index("player")["role"].to_dict()

def get_player_teams(balls_df):
    bat_teams = balls_df.groupby("batter")["batting_team"].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown").reset_index().rename(columns={"batter": "player"})
    bowl_teams = balls_df.groupby("bowler")["bowling_team"].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown").reset_index().rename(columns={"bowler": "player"})
    merged = bat_teams.merge(bowl_teams, on="player", how="outer", suffixes=("_bat", "_bowl"))
    merged["team"] = merged["batting_team"].fillna(merged["bowling_team"]).fillna("Unknown")
    return merged.set_index("player")["team"].to_dict()

def compute_fantasy_points_for_season(balls_df, matches_df):
    bat, bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    if bat.empty and bowl.empty: return pd.DataFrame()
    pts = {}
    for _, row in bat.iterrows():
        p = row["batter"]
        runs = int(row["runs_batter"])
        fours, sixes = int(row.get("4s", 0)), int(row.get("6s", 0))
        balls_faced = int(row.get("balls_faced", row.get("ball", 1)) - int(row.get("wide", 0))
        sr = (runs / balls_faced * 100) if balls_faced > 0 else 0
        bat_pts = runs * FANTASY_POINTS["runs"] + fours * FANTASY_POINTS["four"] + sixes * FANTASY_POINTS["six"]
        sr_bonus = 6 if sr >= 150 else (4 if sr >= 130 else (2 if sr >= 110 else 0))
        pts[p] = pts.get(p, 0) + bat_pts + sr_bonus
    for _, row in bowl.iterrows():
        p = row["bowler"]
        w, rc, balls = int(row["wickets"]), int(row["rc"]), int(row["balls"])
        econ = (rc / (balls / 6)) if balls > 0 else 99
        bowl_pts = w * FANTASY_POINTS["wicket"]
        if w >= 5: bowl_pts += FANTASY_POINTS["five_wicket_bonus"]
        elif w >= 4: bowl_pts += FANTASY_POINTS["four_wicket_bonus"]
        elif w >= 3: bowl_pts += FANTASY_POINTS["three_wicket_bonus"]
        econ_bonus = 6 if econ < 6 else (4 if econ < 7 else (2 if econ < 8 else 0))
        pts[p] = pts.get(p, 0) + bowl_pts + econ_bonus
    return pd.DataFrame([{"player": k, "points": v} for k, v in pts.items()]).sort_values("points", ascending=False)

def get_ai_dream_team(balls_df, season_balls, roles, max_per_team=4):
    pts_df = compute_fantasy_points_for_season(season_balls, None)
    if pts_df.empty: return []
    player_teams = get_player_teams(balls_df)
    pts_df["team"] = pts_df["player"].map(player_teams)
    pts_df["role"] = pts_df["player"].map(roles)
    need_wk, need_bat, need_ar, need_bwl = 1, 4, 2, 4
    picked, team_counts = [], {}
    for role_req, n in [("BAT", need_bat + need_wk), ("AR", need_ar), ("BWL", need_bwl)]:
        pool = pts_df[pts_df["role"] == role_req].sort_values("points", ascending=False)
        for _, row in pool.iterrows():
            if len(picked) >= 11: break
            t = row["team"]
            if team_counts.get(t, 0) >= max_per_team: continue
            if role_req == "BAT" and need_wk > 0:
                need_wk -= 1
                picked.append((row["player"], row["points"], "WK", t))
            elif role_req == "BAT":
                picked.append((row["player"], row["points"], "BAT", t))
            else:
                picked.append((row["player"], row["points"], role_req, t))
            team_counts[t] = team_counts.get(t, 0) + 1
    return picked

def _generate_join_code():
    return "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))

def _fantasy_create_league(supabase, user_id, name, season):
    try:
        join_code = _generate_join_code()
        r = supabase.table("fantasy_leagues").insert({"name": name, "season": int(season), "created_by": str(user_id), "join_code": join_code}).execute()
        data = extract_data(r)
        return (data[0]["id"], join_code) if data and len(data) > 0 else (None, "Failed to create league")
    except Exception as e:
        return None, str(e)

def _fantasy_join_league(supabase, user_id, join_code):
    try:
        r = supabase.table("fantasy_leagues").select("id, name, season").eq("join_code", join_code.strip().upper()).execute()
        data = extract_data(r)
        if not data or len(data) == 0: return None, "Invalid join code"
        league_id = data[0]["id"]
        check = supabase.table("fantasy_teams").select("id").eq("league_id", league_id).eq("user_id", str(user_id)).execute()
        if extract_data(check) and len(extract_data(check)) > 0: return league_id, None
        supabase.table("fantasy_teams").insert({"league_id": league_id, "user_id": str(user_id), "team_name": "My Team"}).execute()
        return league_id, None
    except Exception as e:
        return None, str(e)

def _fantasy_save_team(supabase, team_id, squad_with_roles, captain, vice_captain, total_points):
    try:
        supabase.table("fantasy_teams").update({"captain": captain, "vice_captain": vice_captain, "total_points": float(total_points), "updated_at": datetime.utcnow().isoformat()}).eq("id", team_id).execute()
        supabase.table("fantasy_squad_picks").delete().eq("team_id", team_id).execute()
        for p, role in squad_with_roles:
            supabase.table("fantasy_squad_picks").insert({"team_id": team_id, "player_name": p, "role": role}).execute()
        return True, None
    except Exception as e:
        return False, str(e)

def _fantasy_load_team(supabase, team_id):
    try:
        team_r = supabase.table("fantasy_teams").select("captain, vice_captain, total_points").eq("id", team_id).execute()
        td = extract_data(team_r)
        if not td or len(td) == 0: return None, "Team not found"
        picks = extract_data(supabase.table("fantasy_squad_picks").select("player_name, role").eq("team_id", team_id).execute()) or []
        squad = [(row["player_name"], row["role"]) for row in picks]
        return (squad, td[0].get("captain"), td[0].get("vice_captain"), td[0].get("total_points", 0)), None
    except Exception as e:
        return None, str(e)

# --- NAVIGATION ---
st.sidebar.title("FANTASY XI - Pakistan Cricket League")

# 1. Define the options list explicitly so we can find the index
nav_options = ["Season Dashboard", "Fantasy League", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame", "Pro Prediction"]

# 2. Sync session state with the radio selection
if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Season Dashboard"

# Find the index of the current page to keep the radio button in sync
try:
    default_index = nav_options.index(st.session_state.page_nav)
except ValueError:
    default_index = 0

# 3. The Radio Button - Note the 'index' and 'key'
page = st.sidebar.radio("Navigation", nav_options, index=default_index, key="navigation_radio")

# Update the state if the user clicks manually
st.session_state.page_nav = page

            
# --- AUTH / CONNECTION IN SIDEBAR ---
with st.sidebar.expander("🔐 User Account", expanded=not st.session_state.user):
    if not st.session_state.user:
        if st.session_state.auth_view == "login":
            st.subheader("Login")
            identifier = st.text_input("Email / Mobile / Landline", help="Supports 03xx (11 digits) or 051 (10 digits)")
            password = st.text_input("Password", type="password")
            
            col_login, col_pro = st.columns(2)
            with col_login:
                if st.button("Sign In", use_container_width=True):
                    if validate_identifier(identifier):
                        if supabase:
                            try:
                                res = supabase.auth.sign_in_with_password({"email": identifier, "password": password})
                                if res.session:
                                    st.session_state.user = res.user
                                    st.session_state.access_token = res.session.access_token
                                    supabase.postgrest.auth(res.session.access_token)
                                    # Fetch Pro Status
                                    try:
                                        profile_res = supabase.table("profiles").select("is_pro").eq("id", res.user.id).execute()
                                        profile_data = extract_data(profile_res)
                                        if profile_data and len(profile_data) > 0:
                                            st.session_state.is_pro = profile_data[0].get('is_pro', False)
                                        else:
                                            supabase.table("profiles").insert({
                                                "id": res.user.id,
                                                "identifier": identifier,
                                                "is_pro": False
                                            }).execute()
                                            st.session_state.is_pro = False
                                    except Exception:
                                        st.session_state.is_pro = False
                                    st.success("Logged in successfully!")
                                    st.rerun()
                                else:
                                    st.error("Invalid Credentials")
                            except Exception as e:
                                st.error(f"Login Failed: {str(e)}")
                    else:
                        st.error("Invalid format. Use 11-digit Mobile (03xx) or 10-digit Landline (051xx).")
            with col_pro:
                st.link_button("Request Pro Access", "https://forms.gle/pQSrUN1TcXdTD4nXA", use_container_width=True)
            
            if st.button("Sign Up"):
                st.session_state.auth_view = "signup"
                st.rerun()
                
        elif st.session_state.auth_view == "signup":
            st.subheader("Register")
            e = st.text_input("Email")
            m = st.text_input("Mobile/Landline", max_chars=11, help="Mobile: 11 digits (03xx) | Landline: 10 digits (051xxx)")
            p = st.text_input("Password", type="password")
            if st.button("Create Account"):
                if validate_email(e) and validate_phone(m):
                    if supabase:
                        try:
                            res = supabase.auth.sign_up({
                                "email": e,
                                "password": p,
                                "options": {
                                    "data": {"phone_number": m}
                                }
                            })
                            st.success("Registration initiated.")
                            st.info("Check your email for confirmation link.")
                            st.session_state.auth_view = "login"
                        except Exception as ex:
                            st.error(f"Error creating account: {str(ex)}")
                else:
                    st.error("Validation Failed: Mobile must be 11 digits (03xx) | Landline 10 digits (051xxx)")
            if st.button("Back"):
                st.session_state.auth_view = "login"
                st.rerun()
    else:
        st.write(f"Logged in as: {st.session_state.user.email}")
        if st.session_state.is_pro:
            st.write("⭐ **PRO ACCOUNT ACTIVE**")
        if st.button("Logout"):
            if supabase:
                supabase.auth.sign_out()
            st.session_state.user = None
            st.session_state.access_token = None
            st.session_state.is_pro = False
            st.rerun()

# --- PAGE LOGIC ---
if page == "Match Center":
    st.title("Pro Scorecard & Live Analysis")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    
    def format_match_label(row):
        d = row['date']
        d_str = d.strftime('%Y-%m-%d') if pd.notnull(d) else "Unknown Date"
        return f"{row['team1']} vs {row['team2']} ({d_str})"
    
    ms = ml.apply(format_match_label, axis=1)
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        idx = ms.tolist().index(sel)
        mm = ml.iloc[idx]
        
        # Filter balls for the selected match
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        
        # --- Pre-calculate team names for tabs ---
        team_map = {}
        for inn in [1, 2]:
            t_name = mb[mb['innings'] == inn]['batting_team'].unique()
            team_map[inn] = t_name[0] if len(t_name) > 0 else f"Innings {inn}"
        
        # Match Header UI
        badge_class = "badge-runs" if str(mm['win_by']).strip().lower() == 'runs' else "badge-wickets"
        st.markdown(f"""
        <div class="premium-box" style="border-left: 5px solid #38bdf8;">
            <h2 style='margin:0;'>{mm['team1']} vs {mm['team2']}</h2>
            <p style='color:#94a3b8; font-size:1.1rem; margin:5px 0;'>🏟️ {mm['venue']} | 📅 {(mm['date'].strftime('%Y-%m-%d') if pd.notnull(mm['date']) else 'Unknown Date')}</p>
            <hr style='border-color:#334155;'>
            <p style='font-size:1.2rem;'><b>Result:</b> {mm['winner']} won by 
                <span class="badge {badge_class}">{mm['win_margin']} {mm['win_by']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Broadcast Scorecard")
        
        # Dynamic Tab Labels showing Team Names
        sc1, sc2 = st.tabs([f"1st Innings: {team_map[1]}", f"2nd Innings: {team_map[2]}"])
        
        for i, tab in enumerate([sc1, sc2], 1):
            with tab:
                bt, bl = get_inning_scorecard(mb, i)
                if bt is not None:
                    # Broadcast Style Header
                    total_runs = bt['runs_batter'].sum() # Assuming extras logic is handled or added here
                    total_wickets = bl['W'].sum()
                    
                    st.markdown(f"""
                    <div style="background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 20px;">
                        <h3 style="margin:0; color:#38bdf8;">{team_map[i]} <span style="color:white;">{total_runs}/{total_wickets}</span></h3>
                        <p style="margin:0; color:#94a3b8; font-size:0.9rem;">Innings {i} Summary</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2 = st.columns([1.2, 0.8]) # Adjusted widths for broadcast look
                    with c1:
                        st.markdown("#### 🏏 Batting")
                        st.dataframe(bt, use_container_width=True, hide_index=True)
                    with c2:
                        st.markdown("#### ⚾ Bowling")
                        st.dataframe(bl, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No data available for {team_map[i]}")

        if not mb.empty:
            st.markdown("### Performance Charts")
            chart_col1, chart_col2 = st.columns(2)
            # ... (Rest of your chart logic remains here)
            
            # 1. Match Progression (Worm)
            with chart_col1:
                mb_c = mb.copy()
                worm = mb_c.groupby(['innings', 'over'])['runs_total'].sum().groupby(level=0).cumsum().reset_index()
                # This will now work because team_map is defined above
                worm['Team'] = worm['innings'].map(team_map)
                
                fig_worm = px.line(worm, x='over', y='runs_total', color='Team', 
                                  title="Match Progression (Worm)", template="plotly_dark", 
                                  labels={"over": "Overs", "runs_total": "Total Runs"})
                st.plotly_chart(fig_worm, use_container_width=True)

            # 2. Runs Per Over Progression
            with chart_col2:
                mb_rr = mb.copy()
                mb_rr['over_runs'] = mb_rr.groupby(['innings', 'over'])['runs_total'].transform('sum')
                rr_df = mb_rr[['innings', 'over', 'over_runs']].drop_duplicates()
                # Map using the same team_map
                rr_df['Team'] = rr_df['innings'].map(team_map)
        
                fig_rr = px.bar(rr_df, x='over', y='over_runs', color='Team', 
                    barmode='group', title="Runs Per Over", labels={"over": "Overs", "over_runs": "Runs Per Over"},
                    template="plotly_dark", 
                    color_discrete_sequence=['#00d4ff', '#ff8c00'])
                st.plotly_chart(fig_rr, use_container_width=True)

elif page == "Pro Prediction":
    st.title("AI Match Predictor")
    if not st.session_state.user:
        st.warning("Please Login to Access PRO Predictions.")
    else:
        user_id = st.session_state.user.id
        can_predict, usage_left = True, 3
        if supabase:
            try:
                profile_res = supabase.table("profiles").select("is_pro").eq("id", user_id).execute()
                profile_data = extract_data(profile_res)
                if profile_data and len(profile_data) > 0:
                    st.session_state.is_pro = profile_data[0].get('is_pro', False)
                
                time_threshold = (datetime.now() - timedelta(hours=24)).isoformat()
                usage_res = supabase.table("prediction_attempts")\
                    .select("id", count="exact")\
                    .eq("user_id", user_id)\
                    .gt("created_at", time_threshold)\
                    .execute()
                
                attempts_count = getattr(usage_res, 'count', 0) or 0
                usage_left = max(0, 3 - attempts_count)
                
                if not st.session_state.is_pro and attempts_count >= 3:
                    can_predict = False
                if st.session_state.is_pro:
                    can_predict = True
                    usage_left = "Unlimited"
            except Exception as e:
                st.sidebar.error(f"Read Error: {e}")
        
        if not can_predict:
            st.error("Account Limit Reached. Upgrade to PRO for unlimited access.")
        else:
            if st.session_state.is_pro:
                st.success(f"Welcome! Unlimited Simulations Active.")
            else:
                st.info(f"Welcome! {usage_left} Simulations Remaining.")
            
            with st.spinner("Initializing AI model..."):
                model, le_t, le_v, le_d = train_ml_model(matches_df)
            
            with st.container():
                st.markdown("<div class='premium-box'>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    team1 = st.selectbox("Team 1 (Bat First)", sorted(matches_df['team1'].unique()))
                with col2:
                    team2 = st.selectbox("Team 2 (Chase)", [t for t in sorted(matches_df['team2'].unique()) if t != team1])
                with col3:
                    venue = st.selectbox("Venue / Stadium", sorted(matches_df['venue'].unique()))
                with col4:
                    toss_winner = st.selectbox("Toss Winner", [team1, team2])
                
                toss_decision = st.radio("Toss Decision", sorted(matches_df['toss_decision'].unique()), horizontal=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("PREDICT MATCH WINNER", use_container_width=True):
                    if supabase:
                        try:
                            supabase.table("prediction_attempts").insert({
                                "user_id": user_id,
                                "metadata": {"team1": team1, "team2": team2, "venue": venue}
                            }).execute()
                            supabase.rpc('increment_prediction_log', {
                                'target_user_id': user_id,
                                'target_identifier': st.session_state.user.email,
                                'target_is_pro': st.session_state.is_pro
                            }).execute()
                        except Exception as e:
                            st.sidebar.warning(f"Logging update skipped: {e}")
                    
                    with st.spinner("Analyzing Historical Variables..."):
                        h2h_matches = matches_df[((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) | ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))]
                        h2h_val = len(h2h_matches[h2h_matches['winner'] == team1]) / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
                        v_t1_m = matches_df[(matches_df['venue'] == venue) & ((matches_df['team1'] == team1) | (matches_df['team2'] == team1))]
                        v_t1_val = len(v_t1_m[v_t1_m['winner'] == team1]) / len(v_t1_m) if len(v_t1_m) > 0 else 0.5
                        v_t2_m = matches_df[(matches_df['venue'] == venue) & ((matches_df['team1'] == team2) | (matches_df['team2'] == team2))]
                        v_t2_val = len(v_t2_m[v_t2_m['winner'] == team2]) / len(v_t2_m) if len(v_t2_m) > 0 else 0.5
                        
                        def live_form(team):
                            rel = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)].sort_values('date', ascending=False).head(5)
                            return len(rel[rel['winner'] == team]) / len(rel) if len(rel) > 0 else 0.5
                            
                        input_data = pd.DataFrame({
                            'team1': le_t.transform([team1]),
                            'team2': le_t.transform([team2]),
                            'venue': le_v.transform([venue]),
                            'toss_winner': le_t.transform([toss_winner]),
                            'toss_decision': le_d.transform([toss_decision]),
                            'h2h': [h2h_val],
                            'v_t1': [v_t1_val],
                            'v_t2': [v_t2_val],
                            'form_t1': [live_form(team1)],
                            'form_t2': [live_form(team2)]
                        })
                        
                        probs = model.predict_proba(input_data)[0]
                        t1_prob = f"{probs[1] * 100:.2f}"
                        t2_prob = f"{(1 - probs[1]) * 100:.2f}"
                        
                        st.markdown("### AI Predicted Probability")
                        res1, res2 = st.columns(2)
                        res1.markdown(f"<div class='prediction-card'><h4>{team1}</h4><h1>{t1_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)
                        res2.markdown(f"<div class='prediction-card'><h4>{team2}</h4><h1>{t2_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)
                        # ... existing probability calculation ...
                        
                        st.markdown("### AI Decision Intelligence")
                        
                        # Calculate impact weights (Simplified logic based on feature values)
                        venue_impact = v_t1_val if probs[1] > 0.5 else v_t2_val
                        form_impact = live_form(team1) if probs[1] > 0.5 else live_form(team2)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write("**Venue Advantage**")
                            st.progress(venue_impact)
                            st.caption(f"{int(venue_impact*100)}% Match fit")
                        with col_b:
                            st.write("**Recent Form**")
                            st.progress(form_impact)
                            st.caption(f"{int(form_impact*100)}% Momentum")
                        with col_c:
                            st.write("**H2H Dominance**")
                            st.progress(h2h_val if probs[1] > 0.5 else (1-h2h_val))
                            st.caption("Historical edge")

                        st.info(f"💡 **AI Insight:** {team1 if probs[1] > 0.5 else team2} is favored primarily due to {'strong venue history' if venue_impact > 0.6 else 'superior recent form'}.")

elif page == "Fantasy League":
    st.title("Fantasy League")
    st.markdown("*Build your dream XI and compete using real performance data. Master the art of squad selection to dominate real fantasy leagues.*")
    st.markdown("---")
    fl_season = st.selectbox("Select Season", sorted(matches_df["season"].unique(), reverse=True), key="fl_season")
    fl_balls = balls_df[balls_df["season"] == fl_season]
    if fl_balls.empty:
        st.warning(f"No match data for season {fl_season}")
    else:
        roles = get_player_roles(balls_df)
        player_teams = get_player_teams(balls_df)
        pts_df = compute_fantasy_points_for_season(fl_balls, matches_df)
        tab_build, tab_ai, tab_rules, tab_leaderboard, tab_my_leagues = st.tabs(["Build Your XI", "AI Dream Team", "Scoring Rules", "Season Leaderboard", "My Leagues"])

        with tab_rules:
            st.markdown("""<div class="premium-box"><h3>Fantasy Points System</h3><p><b>Batting:</b> 1 pt/run, +1 per 4, +2 per 6. SR bonus: ≥150 (+6), 130–150 (+4), 110–130 (+2)</p><p><b>Bowling:</b> 25 pts/wicket. 3W +8, 4W +16, 5W +32. Economy: &lt;6 (+6), 6–7 (+4), 7–8 (+2)</p><p><b>Captain 2×</b> | <b>Vice 1.5×</b>. Squad: 11 (1 WK, 3–6 BAT, 1–5 AR, 3–6 BWL), max 4/team.</p></div>""", unsafe_allow_html=True)

        with tab_leaderboard:
            if not pts_df.empty:
                top50 = pts_df.head(50).copy()
                top50["team"] = top50["player"].map(player_teams)
                top50["role"] = top50["player"].map(roles)
                st.dataframe(top50[["player", "role", "team", "points"]], use_container_width=True, hide_index=True)
            else:
                st.info("No fantasy points data for this season.")

        with tab_ai:
            st.subheader("AI Dream Team")
            if st.button("Generate AI Dream Team", key="gen_ai_team"):
                with st.spinner("Computing optimal XI..."):
                    dream = get_ai_dream_team(balls_df, fl_balls, roles)
                if dream:
                    cols = st.columns(3)
                    for i, (name, pts, role, team) in enumerate(dream):
                        with cols[i % 3]:
                            st.markdown(f"""<div class="premium-box" style="padding:15px;"><b>{name}</b><br><span style="color:#38bdf8;">{role}</span> · {team}<br><span style="color:#10b981;">{int(pts)} pts</span></div>""", unsafe_allow_html=True)
                    st.success(f"Total projected points: {int(sum(p for _, p, _, _ in dream))}")
                else:
                    st.warning("Could not generate Dream Team.")

        with tab_my_leagues:
            if not st.session_state.user:
                st.info("Login to create or join leagues and save your teams.")
            else:
                user_id = st.session_state.user.id
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader("Create League")
                    league_name = st.text_input("League Name", key="fl_league_name")
                    if st.button("Create League", key="fl_create"):
                        if league_name and league_name.strip() and supabase:
                            lid, code = _fantasy_create_league(supabase, user_id, league_name.strip(), fl_season)
                            st.success(f"League created! Share code: **{code}**") if lid else st.error(code)
                        elif not league_name or not league_name.strip():
                            st.warning("Enter a league name.")
                with c2:
                    st.subheader("Join League")
                    join_code_in = st.text_input("Join Code", key="fl_join_code").strip().upper()
                    if st.button("Join League", key="fl_join"):
                        if join_code_in and supabase:
                            lid, err = _fantasy_join_league(supabase, user_id, join_code_in)
                            st.error(err) if err else st.success("Joined!")
                        else:
                            st.warning("Enter join code.")
                with c3:
                    st.subheader("My Leagues")
                    if supabase:
                        try:
                            my_teams_r = supabase.table("fantasy_teams").select("id, team_name, total_points, league_id").eq("user_id", str(user_id)).execute()
                            my_data = extract_data(my_teams_r) or []
                            league_ids = list({t["league_id"] for t in my_data if t.get("league_id")})
                            leagues_map = {}
                            if league_ids:
                                for row in (extract_data(supabase.table("fantasy_leagues").select("id, name, season, join_code").in_("id", league_ids).execute()) or []):
                                    leagues_map[row["id"]] = row
                            leagues_for_season = [t for t in my_data if leagues_map.get(t.get("league_id"), {}).get("season") == fl_season]
                            for t in leagues_for_season:
                                lg = leagues_map.get(t.get("league_id"), {})
                                st.markdown(f"**{lg.get('name','')}** (S{lg.get('season','')}) · {int(t.get('total_points',0))} pts")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button("Load", key=f"fl_sel_{t['id']}"):
                                        st.session_state.fl_selected_team_id = t["id"]
                                        st.rerun()
                                with col_b:
                                    if st.button("LB", key=f"fl_lb_{t['league_id']}"):
                                        st.session_state.fl_view_leaderboard_league_id = t["league_id"]
                                        st.rerun()
                            if "fl_view_leaderboard_league_id" in st.session_state:
                                lb_teams = supabase.table("fantasy_teams").select("team_name, total_points").eq("league_id", st.session_state.fl_view_leaderboard_league_id).order("total_points", desc=True).execute()
                                lb_data = extract_data(lb_teams) or []
                                if lb_data:
                                    st.dataframe(pd.DataFrame([{"Rank": i+1, "Team": r.get("team_name",""), "Points": int(r.get("total_points",0))} for i, r in enumerate(lb_data)]), hide_index=True)
                                if st.button("Close Leaderboard"):
                                    del st.session_state.fl_view_leaderboard_league_id
                                    st.rerun()
                            if not leagues_for_season:
                                st.caption("No leagues for this season.")
                        except Exception as e:
                            st.error(str(e))

        with tab_build:
            st.subheader("Build Your XI")
            if "fl_selected_team_id" not in st.session_state:
                st.session_state.fl_selected_team_id = None
            if st.session_state.user and supabase and st.session_state.fl_selected_team_id:
                loaded = _fantasy_load_team(supabase, st.session_state.fl_selected_team_id)
                if loaded[1] is None:
                    squad_data, cap, vc, _ = loaded[0]
                    st.session_state.fl_loaded_squad = [p for p, _ in squad_data]
                    st.session_state.fl_loaded_cap, st.session_state.fl_loaded_vc = cap, vc
                    st.session_state.fl_loaded_roles = {p: r for p, r in squad_data}
                st.session_state.fl_selected_team_id = None

            bat_players = [p for p, r in roles.items() if r == "BAT"]
            ar_players = [p for p, r in roles.items() if r == "AR"]
            bwl_players = [p for p, r in roles.items() if r == "BWL"]
            pts_map = pts_df.set_index("player")["points"].to_dict() if not pts_df.empty else {}

            default_wk, default_bat, default_ar, default_bwl = "— Select —", [], [], []
            if st.session_state.get("fl_loaded_squad"):
                ld, lr = st.session_state.fl_loaded_squad, st.session_state.get("fl_loaded_roles", {})
                default_wk = ([p for p in ld if lr.get(p) == "WK" and p in bat_players] or ["— Select —"])[0] if ld else "— Select —"
                if default_wk == "— Select —" and ld: default_wk = "— Select —"
                default_bat = [p for p in ld if lr.get(p) == "BAT" and p in bat_players]
                default_ar = [p for p in ld if lr.get(p) == "AR" and p in ar_players]
                default_bwl = [p for p in ld if lr.get(p) == "BWL" and p in bwl_players]
                st.session_state.fl_loaded_squad = None

            wk_opts = ["— Select —"] + bat_players
            wk_idx = wk_opts.index(default_wk) if default_wk in wk_opts else 0
            c1, c2, c3 = st.columns(3)
            with c1:
                wk = st.selectbox("Wicket-Keeper (1)", wk_opts, index=wk_idx, key="fl_wk")
            with c2:
                bat = st.multiselect("Batsmen (3–6)", [p for p in bat_players if p != wk or wk == "— Select —"], default=default_bat, key="fl_bat")
            with c3:
                ar = st.multiselect("All-Rounders (1–5)", ar_players, default=default_ar, key="fl_ar")
            bwl = st.multiselect("Bowlers (3–6)", bwl_players, default=default_bwl, key="fl_bwl")

            wk_list = [wk] if wk and wk != "— Select —" else []
            squad = list(dict.fromkeys(wk_list + bat + ar + bwl))
            n_wk, n_bat = len(wk_list), len(wk_list) + len([p for p in bat if p not in wk_list])
            n_ar, n_bwl = len(ar), len(bwl)
            team_counts = {}
            for p in squad:
                t = player_teams.get(p, "Unknown")
                team_counts[t] = team_counts.get(t, 0) + 1
            team_ok = not any(c > 4 for c in team_counts.values())
            valid = n_wk == 1 and 3 <= n_bat <= 6 and 1 <= n_ar <= 5 and 3 <= n_bwl <= 6 and len(squad) == 11 and team_ok

            if len(squad) == 11 and valid:
                cap = st.selectbox("Captain (2×)", squad, index=squad.index(st.session_state.get("fl_loaded_cap")) if st.session_state.get("fl_loaded_cap") in squad else 0, key="fl_cap")
                vc_opts = [p for p in squad if p != cap]
                vc = st.selectbox("Vice-Captain (1.5×)", vc_opts, index=vc_opts.index(st.session_state.get("fl_loaded_vc")) if st.session_state.get("fl_loaded_vc") in vc_opts else 0, key="fl_vc")
                if st.button("Calculate My Score", key="fl_calc"):
                    total = sum(pts_map.get(p, 0) * (2.0 if p == cap else (1.5 if p == vc else 1.0)) for p in squad)
                    st.balloons()
                    st.markdown(f"""<div class="prediction-card"><h2>Your Fantasy Score</h2><h1 style="color:#38bdf8;">{int(total)}</h1><p>Captain: {cap} · Vice: {vc}</p></div>""", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{"Player": p, "Role": roles.get(p,"-"), "Base": int(pts_map.get(p,0)), "Total": int(pts_map.get(p,0) * (2 if p==cap else (1.5 if p==vc else 1)))} for p in squad]), hide_index=True)

                    if st.session_state.user and supabase:
                        st.markdown("---")
                        try:
                            my_teams_r = supabase.table("fantasy_teams").select("id, team_name, league_id").eq("user_id", str(st.session_state.user.id)).execute()
                            my_data = extract_data(my_teams_r) or []
                            leagues_map = {row["id"]: row for row in (extract_data(supabase.table("fantasy_leagues").select("id, name, season").in_("id", list({t["league_id"] for t in my_data if t.get("league_id")})).execute()) or [])}
                            my_leagues_this_season = [t for t in my_data if leagues_map.get(t.get("league_id"), {}).get("season") == fl_season]
                        except Exception:
                            my_leagues_this_season = []
                        if my_leagues_this_season:
                            league_options = {f"{leagues_map.get(t['league_id'],{}).get('name','')} ({t.get('team_name','')})": t["id"] for t in my_leagues_this_season}
                            save_league = st.selectbox("Save to league", list(league_options.keys()), key="fl_save_league")
                            if st.button("Save Team", key="fl_save_btn"):
                                ok, err = _fantasy_save_team(supabase, league_options[save_league], [(p, roles.get(p,"BAT")) for p in squad], cap, vc, total)
                                st.success("Saved!") if ok else st.error(err)
                                if ok: st.rerun()
                        else:
                            st.caption("Create or join a league to save.")
            else:
                if len(squad) != 11: st.warning("Select exactly 11 players.")
                elif not valid: st.warning("Check role constraints: 1 WK, 3–6 BAT, 1–5 AR, 3–6 BWL, max 4/team.")

elif page == "Season Dashboard":
    season = st.selectbox("Select Season", sorted(matches_df['season'].unique(), reverse=True))
    st.title(f"Tournament Summary: {season}")
    s_balls = balls_df[balls_df['season'] == season]
    if not s_balls.empty:
        bat, bowl = get_batting_stats(s_balls), get_bowling_stats(s_balls)
        mvp = s_balls.groupby('batter').agg({'runs_batter': 'sum', 'is_wicket': 'sum'}).reset_index()
        mvp['score'] = (mvp['runs_batter'] * 1) + (mvp['is_wicket'] * 25)
        mvp = mvp.sort_values('score', ascending=False).head(10)
        
        m1, m2, m3 = st.columns(3)
        if not bat.empty:
            m1.metric("Orange Cap", bat.iloc[0]['batter'], f"{int(bat.iloc[0]['runs_batter'])} Runs")
        if not bowl.empty:
            m2.metric("Purple Cap", bowl.iloc[0]['bowler'], f"{int(bowl.iloc[0]['wickets'])} Wickets")
        if not mvp.empty:
            m3.metric("Top MVP", mvp.iloc[0]['batter'], f"{int(mvp.iloc[0]['score'])} Pts")
            
        st.subheader("Top Performers")
        c1, c2 = st.columns(2)
        if not bat.empty:
            c1.plotly_chart(px.bar(bat.head(10), x='batter', y='runs_batter', title="Top 10 Batters", template="plotly_dark",labels={"runs_batter": "Total Runs"} ), use_container_width=True)
        if not bowl.empty:
            c2.plotly_chart(px.bar(bowl.head(10), x='bowler', y='wickets', title="Top 10 Bowlers", template="plotly_dark", labels={"wickets": "Total Wickets"}), use_container_width=True)
        if not mvp.empty:
            st.plotly_chart(px.bar(mvp, x='batter', y='score', title="Top 10 MVP Impact", template="plotly_dark"), use_container_width=True)
    else:
        st.warning(f"No match data found for season {season}")

elif page == "Fantasy Scout":
    st.title("Fantasy Team Optimizer")
    season_f = st.selectbox("Data Context", sorted(matches_df['season'].unique(), reverse=True))
    sf_balls = balls_df[balls_df['season'] == season_f]
    if not sf_balls.empty:
        b, w = get_batting_stats(sf_balls), get_bowling_stats(sf_balls)
        fan = b.merge(w, left_on='batter', right_on='bowler', how='outer').fillna(0)
        fan['p_name'] = fan['batter'].where((fan['batter'] != 0) & (fan['batter'] != ""), fan['bowler'])
        fan['pts'] = (fan['runs_batter']*1) + (fan['wickets']*25)
        fig_fan = px.bar(fan.sort_values('pts', ascending=False).head(11), x='pts', y='p_name', orientation='h', title="Fantasy Impact Ranking", labels={"p_name": "Player Name"}, template="plotly_dark")
        st.plotly_chart(fig_fan, use_container_width=True)

elif page == "Impact Players":
    st.title("Impact Players")
    
    # 1. Use the pre-calculated all_players list from load_data
    # (This is much faster than recalculating set(batter | bowler) on every click)
    
    # 2. Check if a player was sent here from the sidebar search
    default_player_index = 0
    if 'selected_player_override' in st.session_state:
        try:
            # Match the override name to the index in our master list
            default_player_index = all_players.index(st.session_state.selected_player_override)
            # Clear it so the user can freely change players afterward
            del st.session_state.selected_player_override
        except (ValueError, NameError):
            default_player_index = 0

    # 3. The Selectbox - NOW USING THE INDEX
    p = st.selectbox("Select Player", all_players, index=default_player_index)
    
    # 4. Data Processing & UI
    all_bat, all_bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    
    # Wrap stats in a premium-box for the polish we discussed earlier
    st.markdown(f'<div class="premium-box"><h3>Performance Dashboard: {p}</h3></div>', unsafe_allow_html=True)
    
    ca, cb = st.columns(2)
    bp = all_bat[all_bat['batter'] == p]
    if not bp.empty:
        with ca:
            st.metric("Total Runs", int(bp.iloc[0]['runs_batter']))
            st.metric("Strike Rate", f"{bp.iloc[0]['strike_rate']:.2f}")
    
    wp = all_bowl[all_bowl['bowler'] == p]
    if not wp.empty:
        with cb:
            st.metric("Total Wickets", int(wp.iloc[0]['wickets']))
            st.metric("Economy", f"{wp.iloc[0]['economy']:.2f}")

elif page == "Player Comparison":
    st.title("Head-to-Head Comparison")
    all_players = sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique())))
    c1, c2 = st.columns(2)
    p1, p2 = c1.selectbox("Player 1", all_players, index=0), c2.selectbox("Player 2", all_players, index=min(1, len(all_players)-1))
    bat_all, bowl_all = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    
    def get_p_stats(name):
        b, w = bat_all[bat_all['batter'] == name], bowl_all[bowl_all['bowler'] == name]
        return {
            'Runs': int(b.iloc[0]['runs_batter']) if not b.empty else 0, 
            'SR': round(float(b.iloc[0]['strike_rate']), 2) if not b.empty else 0.00, 
            'Wickets': int(w.iloc[0]['wickets']) if not w.empty else 0, 
            'Econ': round(float(w.iloc[0]['economy']), 2) if not w.empty else 0.00
        }
        
    s1, s2 = get_p_stats(p1), get_p_stats(p2)
    st.table(pd.DataFrame({
        'Metric': ['Runs', 'SR', 'Wickets', 'Econ'],
        p1: [s1['Runs'], f"{s1['SR']:.2f}", s1['Wickets'], f"{s1['Econ']:.2f}"],
        p2: [s2['Runs'], f"{s2['SR']:.2f}", s2['Wickets'], f"{s2['Econ']:.2f}"]
    }))

elif page == "Venue Analysis":
    st.title("Venue Intelligence")
    valid_venues = [v for v in matches_df['venue'].unique() if v and str(v).lower() != 'nan' and str(v).strip() != '']
    v = st.selectbox("Select Venue", sorted(valid_venues))
    vm = matches_df[matches_df['venue'] == v]
    st.metric("Matches Hosted", int(len(vm)))
    st.metric("Defend Wins", int(len(vm[vm['win_by'] == 'runs'])))
    v_balls = balls_df[balls_df['venue'] == v]
    if not v_balls.empty:
        avg_score = v_balls[v_balls['innings']==1].groupby('match_id')['runs_batter'].sum().mean()
        st.metric("Avg 1st Innings", int(avg_score) if not np.isnan(avg_score) else 0)

elif page == "Umpire Records":
    st.title("Umpire Records")
    u = st.selectbox("Select Umpire", sorted(pd.concat([matches_df['umpire1'], matches_df['umpire2']]).unique()))
    um = matches_df[(matches_df['umpire1'] == u) | (matches_df['umpire2'] == u)]
    if not um.empty:
        fig_ump = px.bar(um['winner'].value_counts().reset_index(), x='winner', y='count', template="plotly_dark", labels={"winner": "Winner Team", "count": "Match Count"})
        st.plotly_chart(fig_ump, use_container_width=True)

elif page == "Hall of Fame":
    st.title("All-Time Records")
    t1, t2 = st.tabs(["Batting", "Bowling"])
    with t1:
        st.dataframe(get_batting_stats(balls_df).head(50), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(get_bowling_stats(balls_df).head(50), use_container_width=True, hide_index=True)

# --- GLOBAL LEGAL DISCLAIMER ---
st.markdown("""
<div class="disclaimer-box">
    <strong>Legal Disclaimer & Terms of Use:</strong><br>
    This platform is an independent fan-led project and is not affiliated with the PSL or PCB. Predictions are probabilistic and for entertainment only.
</div>
""", unsafe_allow_html=True)
