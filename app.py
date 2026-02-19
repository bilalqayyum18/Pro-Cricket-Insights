import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import re
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
st.set_page_config(page_title="Pro Cricket Insights", layout="wide", page_icon="🏏")

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
    # Mobile: 11 digits max (starts with 03xx)
    if re.match(r'^03\d{9}$', identifier):
        return True
    # Landline: 10 digits max (starts with 051xxx)
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
            
        # FETCH BALL BY BALL
        all_balls = []
        batch_size = 1000 
        start = 0
        
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

        for df in [matches, balls]:
            df['match_id'] = pd.to_numeric(df['match_id'], errors='coerce').astype('Int64')
            if 'season' in df.columns:
                df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')

        matches['date'] = pd.to_datetime(matches.get('date'), errors='coerce')
        matches['venue'] = matches.get('venue', '').astype(str).str.split(',').str[0]

        numeric_cols = [
            'runs_batter', 'runs_extras', 'runs_total', 'wide', 
            'noball', 'is_wicket', 'innings', 'over', 'ball', 'season', 'match_id'
        ]
        for col in numeric_cols:
            if col in balls.columns:
                balls[col] = pd.to_numeric(balls[col], errors='coerce').fillna(0)

        venue_map = matches.set_index('match_id')['venue'].to_dict()
        balls['venue'] = balls['match_id'].map(venue_map).fillna("Unknown Venue")

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

        all_players_list = sorted(list(set(balls['batter'].unique()) | set(balls['bowler'].unique())))
        return matches, balls, all_players_list

    except Exception as e:
        st.error(f"Data process failed: {e}")
        st.stop()

# Initialize data
matches_df, balls_df, all_players = load_data()


# --- ML MODEL ENGINE ---
@st.cache_resource
def train_ml_model(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    model_df = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'date']].dropna()
    model_df = model_df.sort_values('date')
    
    def get_h2h_win_rate(t1, t2, date):
        relevant = df[(df['date'] < date) & (((df['team1'] == t1) & (df['team2'] == t2)) | ((df['team1'] == t2) & (df['team2'] == t1)))]
        return len(relevant[relevant['winner'] == t1]) / len(relevant) if len(relevant) > 0 else 0.5

    def get_venue_win_rate(team, venue, date):
        relevant = df[(df['date'] < date) & (df['venue'] == venue) & ((df['team1'] == team) | (df['team2'] == team))]
        return len(relevant[relevant['winner'] == team]) / len(relevant) if len(relevant) > 0 else 0.5

    def get_recent_form(team, date):
        relevant = df[(df['date'] < date) & ((df['team1'] == team) | (df['team2'] == team))].sort_values('date', ascending=False).head(5)
        return len(relevant[relevant['winner'] == team]) / len(relevant) if len(relevant) > 0 else 0.5

    model_df['h2h'] = model_df.apply(lambda x: get_h2h_win_rate(x['team1'], x['team2'], x['date']), axis=1)
    model_df['v_t1'] = model_df.apply(lambda x: get_venue_win_rate(x['team1'], x['venue'], x['date']), axis=1)
    model_df['v_t2'] = model_df.apply(lambda x: get_venue_win_rate(x['team2'], x['venue'], x['date']), axis=1)
    model_df['form_t1'] = model_df.apply(lambda x: get_recent_form(x['team1'], x['date']), axis=1)
    model_df['form_t2'] = model_df.apply(lambda x: get_recent_form(x['team2'], x['date']), axis=1)

    le_team, le_venue, le_decision = LabelEncoder(), LabelEncoder(), LabelEncoder()
    le_team.fit(pd.concat([model_df['team1'], model_df['team2']]).unique())
    le_venue.fit(model_df['venue'].unique())
    le_decision.fit(model_df['toss_decision'].unique())

    X = pd.DataFrame({
        'team1': le_team.transform(model_df['team1']), 'team2': le_team.transform(model_df['team2']),
        'venue': le_venue.transform(model_df['venue']), 'toss_winner': le_team.transform(model_df['toss_winner']),
        'toss_decision': le_decision.transform(model_df['toss_decision']), 'h2h': model_df['h2h'],
        'v_t1': model_df['v_t1'], 'v_t2': model_df['v_t2'], 'form_t1': model_df['form_t1'], 'form_t2': model_df['form_t2']
    })
    y = (model_df['winner'] == model_df['team1']).astype(int)

    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, objective="binary:logistic", random_state=42)
    model.fit(X, y)
    return model, le_team, le_venue, le_decision

# --- ANALYTICS ENGINES ---
def get_batting_stats(df):
    if df.empty: return pd.DataFrame()
    bat = df.groupby('batter').agg({'runs_batter': 'sum', 'ball': 'count', 'wide': 'sum', 'match_id': 'nunique', 'is_wicket': 'sum'}).reset_index()
    bat['balls_faced'] = bat['ball'] - bat['wide']
    bat['strike_rate'] = (bat['runs_batter'] / (bat['balls_faced'].replace(0, 1)) * 100).round(2)
    f = df[df['runs_batter'] == 4].groupby('batter').size().reset_index(name='4s')
    s = df[df['runs_batter'] == 6].groupby('batter').size().reset_index(name='6s')
    res = bat.merge(f, on='batter', how='left').merge(s, on='batter', how='left').fillna(0)
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
    
    bw = id_df[~id_df['wicket_kind'].isin(['run out', 'retired hurt'])]
    w = bw.groupby('bowler')['is_wicket'].sum().reset_index().rename(columns={'is_wicket':'W'})
    id_df['rc_temp'] = id_df['runs_batter'] + id_df['wide'] + id_df['noball']
    r = id_df.groupby('bowler')['rc_temp'].sum().reset_index()
    bls = id_df[(id_df['wide']==0) & (id_df['noball']==0)].groupby('bowler').size().reset_index(name='bls')
    bowl = w.merge(r, on='bowler').merge(bls, on='bowler')
    bowl['O'] = ((bowl['bls']//6) + (bowl['bls']%6/10)).round(1)
    bowl['Econ'] = (bowl['rc_temp']/(bowl['bls'].replace(0,1)/6)).round(1)
    return bat[['batter', 'runs_batter', 'B', '4s', '6s', 'SR']], bowl[['bowler', 'O', 'rc_temp', 'W', 'Econ']]

# --- NAVIGATION ---
st.sidebar.title("Pakistan League Intelligence")
nav_options = ["Season Dashboard", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame", "Pro Prediction"]

if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Season Dashboard"

try:
    default_index = nav_options.index(st.session_state.page_nav)
except ValueError:
    default_index = 0

page = st.sidebar.radio("Navigation", nav_options, index=default_index, key="navigation_radio")
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
                                    st.session_state.user, st.session_state.access_token = res.user, res.session.access_token
                                    supabase.postgrest.auth(res.session.access_token)
                                    profile_res = supabase.table("profiles").select("is_pro").eq("id", res.user.id).execute()
                                    profile_data = extract_data(profile_res)
                                    st.session_state.is_pro = profile_data[0].get('is_pro', False) if profile_data else False
                                    st.success("Logged in successfully!"); st.rerun()
                                else: st.error("Invalid Credentials")
                            except Exception as e: st.error(f"Login Failed: {str(e)}")
                    else: st.error("Invalid format. Use 11-digit Mobile (03xx) or 10-digit Landline (051xx).")
            with col_pro: st.link_button("Request Pro Access", "https://forms.gle/pQSrUN1TcXdTD4nXA", use_container_width=True)
            if st.button("Sign Up"): st.session_state.auth_view = "signup"; st.rerun()
        elif st.session_state.auth_view == "signup":
            st.subheader("Register")
            e, m, p = st.text_input("Email"), st.text_input("Mobile/Landline", max_chars=11), st.text_input("Password", type="password")
            if st.button("Create Account"):
                if validate_email(e) and validate_phone(m):
                    if supabase:
                        try:
                            supabase.auth.sign_up({"email": e, "password": p, "options": {"data": {"phone_number": m}}})
                            st.success("Registration initiated. Check email."); st.session_state.auth_view = "login"
                        except Exception as ex: st.error(f"Error: {str(ex)}")
                else: st.error("Validation Failed.")
            if st.button("Back"): st.session_state.auth_view = "login"; st.rerun()
    else:
        st.write(f"Logged in as: {st.session_state.user.email}")
        if st.session_state.is_pro: st.write("⭐ **PRO ACCOUNT ACTIVE**")
        if st.button("Logout"):
            if supabase: supabase.auth.sign_out()
            st.session_state.user, st.session_state.access_token, st.session_state.is_pro = None, None, False
            st.rerun()

# --- PAGE LOGIC ---
if page == "Match Center":
    st.title("Pro Scorecard & Live Analysis")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    ms = ml.apply(lambda row: f"{row['team1']} vs {row['team2']} ({row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else 'Unknown'})", axis=1)
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        idx = ms.tolist().index(sel)
        mm = ml.iloc[idx]
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        badge_class = "badge-runs" if mm['win_by'].strip().lower() == 'runs' else "badge-wickets"
        st.markdown(f'<div class="premium-box" style="border-left: 5px solid #38bdf8;"><h2>{mm["team1"]} vs {mm["team2"]}</h2><p>🏟️ {mm["venue"]} | 📅 {mm["date"].strftime("%Y-%m-%d")}</p><p><b>Result:</b> {mm["winner"]} won by <span class="badge {badge_class}">{mm["win_margin"]} {mm["win_by"]}</span></p></div>', unsafe_allow_html=True)
        sc1, sc2 = st.tabs(["1st Innings", "2nd Innings"])
        for i, tab in enumerate([sc1, sc2], 1):
            with tab:
                bt, bl = get_inning_scorecard(mb, i)
                if bt is not None:
                    c1, c2 = st.columns(2); c1.dataframe(bt, hide_index=True); c2.dataframe(bl, hide_index=True)

elif page == "Pro Prediction":
    st.title("AI Match Predictor")
    if not st.session_state.user: st.warning("Please Login to Access PRO Predictions.")
    else:
        user_id = st.session_state.user.id
        can_predict, usage_left = True, 3
        if supabase:
            try:
                time_threshold = (datetime.now() - timedelta(hours=24)).isoformat()
                usage_res = supabase.table("prediction_attempts").select("id", count="exact").eq("user_id", user_id).gt("created_at", time_threshold).execute()
                attempts_count = getattr(usage_res, 'count', 0) or 0
                usage_left = "Unlimited" if st.session_state.is_pro else max(0, 3 - attempts_count)
                if not st.session_state.is_pro and attempts_count >= 3: can_predict = False
            except: pass
        
        if not can_predict: st.error("Account Limit Reached.")
        else:
            st.info(f"Welcome! {usage_left} Simulations Remaining.")
            model, le_t, le_v, le_d = train_ml_model(matches_df)
            col1, col2, col3, col4 = st.columns(4)
            t1 = col1.selectbox("Team 1", sorted(matches_df['team1'].unique()))
            t2 = col2.selectbox("Team 2", [t for t in sorted(matches_df['team2'].unique()) if t != t1])
            ven = col3.selectbox("Venue", sorted(matches_df['venue'].unique()))
            t_win = col4.selectbox("Toss Winner", [t1, t2])
            t_dec = st.radio("Toss Decision", sorted(matches_df['toss_decision'].unique()), horizontal=True)
            
            if st.button("PREDICT MATCH WINNER", use_container_width=True):
                if supabase:
                    supabase.table("prediction_attempts").insert({"user_id": user_id, "metadata": {"t1": t1, "t2": t2, "ven": ven}}).execute()
                    supabase.rpc('increment_prediction_log', {'target_user_id': user_id, 'target_identifier': st.session_state.user.email, 'target_is_pro': st.session_state.is_pro}).execute()
                
                h2h_matches = matches_df[((matches_df['team1'] == t1) & (matches_df['team2'] == t2)) | ((matches_df['team1'] == t2) & (matches_df['team2'] == t1))]
                h2h_val = len(h2h_matches[h2h_matches['winner'] == t1]) / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
                probs = model.predict_proba(pd.DataFrame({'team1': le_t.transform([t1]), 'team2': le_t.transform([t2]), 'venue': le_v.transform([ven]), 'toss_winner': le_t.transform([t_win]), 'toss_decision': le_d.transform([t_dec]), 'h2h': [h2h_val], 'v_t1': [0.5], 'v_t2': [0.5], 'form_t1': [0.5], 'form_t2': [0.5]}))[0]
                st.markdown("### AI Predicted Probability")
                res1, res2 = st.columns(2)
                res1.markdown(f"<div class='prediction-card'><h4>{t1}</h4><h1>{probs[1]*100:.2f}%</h1></div>", unsafe_allow_html=True)
                res2.markdown(f"<div class='prediction-card'><h4>{t2}</h4><h1>{(1-probs[1])*100:.2f}%</h1></div>", unsafe_allow_html=True)

elif page == "Impact Players":
    st.title("Impact Players")
    p = st.selectbox("Select Player", all_players)
    all_bat, all_bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    st.markdown(f'<div class="premium-box"><h3>Performance Dashboard: {p}</h3></div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    bp = all_bat[all_bat['batter'] == p]
    if not bp.empty: ca.metric("Total Runs", int(bp.iloc[0]['runs_batter'])); ca.metric("Strike Rate", f"{bp.iloc[0]['strike_rate']:.2f}")
    wp = all_bowl[all_bowl['bowler'] == p]
    if not wp.empty: cb.metric("Total Wickets", int(wp.iloc[0]['wickets'])); cb.metric("Economy", f"{wp.iloc[0]['economy']:.2f}")

elif page == "Player Comparison":
    st.title("Head-to-Head Comparison")
    c1, c2 = st.columns(2)
    p1, p2 = c1.selectbox("Player 1", all_players, index=0), c2.selectbox("Player 2", all_players, index=1)
    bat_all, bowl_all = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    def get_p_stats(name):
        b, w = bat_all[bat_all['batter'] == name], bowl_all[bowl_all['bowler'] == name]
        return [int(b.iloc[0]['runs_batter']) if not b.empty else 0, round(float(b.iloc[0]['strike_rate']), 2) if not b.empty else 0.0, int(w.iloc[0]['wickets']) if not w.empty else 0, round(float(w.iloc[0]['economy']), 2) if not w.empty else 0.0]
    st.table(pd.DataFrame({'Metric': ['Runs', 'SR', 'Wickets', 'Econ'], p1: get_p_stats(p1), p2: get_p_stats(p2)}))

elif page == "Venue Analysis":
    st.title("Venue Intelligence")
    valid_venues = sorted([v for v in matches_df['venue'].unique() if v and str(v) != 'nan'])
    v = st.selectbox("Select Venue", valid_venues)
    vm = matches_df[matches_df['venue'] == v]
    st.metric("Matches Hosted", len(vm)); st.metric("Defend Wins", len(vm[vm['win_by'] == 'runs']))

elif page == "Hall of Fame":
    st.title("All-Time Records")
    t1, t2 = st.tabs(["Batting", "Bowling"])
    with t1: st.dataframe(get_batting_stats(balls_df).head(50), hide_index=True)
    with t2: st.dataframe(get_bowling_stats(balls_df).head(50), hide_index=True)

st.markdown('<div class="disclaimer-box"><strong>Legal Disclaimer:</strong> Not affiliated with PSL or PCB.</div>', unsafe_allow_html=True)
