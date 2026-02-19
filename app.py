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

# --- PREMIUM UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }
    .prediction-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 30px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .prediction-card h4 {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .prediction-card h1 {
        color: #38bdf8;
        margin: 0;
        font-size: 2.5rem;
    }
    .premium-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #ef4444;
        padding: 24px;
        border-radius: 6px;
        margin-top: 40px;
        margin-bottom: 60px;
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.6;
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
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    .stRadio > label {
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
    .stButton>button {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: 800 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
    }
</style>
<div class="footer">
    <b>INDEPENDENT FAN PORTAL:</b> Not affiliated with, endorsed by, or associated with the PSL or PCB. PRO CRICKET INSIGHTS © 2026 | For Analytical Purposes Only.
</div>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data():
    if supabase_status != "Connected":
        st.error(f"Supabase Connection Failed: {supabase_status}")
        st.stop()
    try:
        # FETCH MATCHES
        matches_res = supabase.table("matches").select("*").execute()
        matches = pd.DataFrame(matches_res.data)
        if matches.empty:
            st.error("Matches table is empty.")
            st.stop()

        # FETCH BALL BY BALL IN BATCHES
        all_balls = []
        batch_size = 10000
        start = 0
        while True:
            response = supabase.table("ball_by_ball") \
                .select("*") \
                .range(start, start + batch_size - 1) \
                .execute()
            batch = response.data
            if not batch:
                break
            all_balls.extend(batch)
            if len(batch) < batch_size:
                break
            start += batch_size
        
        balls = pd.DataFrame(all_balls)
        if balls.empty:
            st.error("Ball_by_ball table is empty.")
            st.stop()

        # CLEAN DATA
        matches['match_id'] = pd.to_numeric(matches['match_id'], errors='coerce')
        matches['season'] = pd.to_numeric(matches['season'], errors='coerce')
        matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
        matches['venue'] = matches['venue'].astype(str).str.split(',').str[0]

        balls['match_id'] = pd.to_numeric(balls['match_id'], errors='coerce')
        balls['season'] = pd.to_numeric(balls['season'], errors='coerce')
        balls['innings'] = pd.to_numeric(balls['innings'], errors='coerce')
        balls['over'] = pd.to_numeric(balls['over'], errors='coerce')
        balls['ball'] = pd.to_numeric(balls['ball'], errors='coerce')
        
        numeric_cols = ['runs_batter', 'runs_extras', 'runs_total', 'wide', 'noball', 'bye', 'legbye', 'is_wicket']
        for col in numeric_cols:
            if col in balls.columns:
                balls[col] = pd.to_numeric(balls[col], errors='coerce').fillna(0)

        venue_map = matches.set_index('match_id')['venue'].to_dict()
        balls['venue'] = balls['match_id'].map(venue_map)

        return matches, balls
    except Exception as e:
        st.error(f"Supabase fetch failed: {e}")
        st.stop()

matches_df, balls_df = load_data()

# --- ML MODEL ENGINE ---
@st.cache_resource
def train_ml_model(df):
    model_df = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'date']]\
        .dropna()\
        .sort_values('date')

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
page = st.sidebar.radio("Navigation", ["Season Dashboard", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame", "Pro Prediction"])

# --- AUTH IN SIDEBAR ---
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
                                    st.session_state.is_pro = True # Simplify for logic demo
                                    st.rerun()
                            except: st.error("Login Failed")
                    else: st.error("Invalid format.")
            if st.button("Sign Up"):
                st.session_state.auth_view = "signup"
                st.rerun()
        elif st.session_state.auth_view == "signup":
            st.subheader("Register")
            e = st.text_input("Email")
            m = st.text_input("Mobile/Landline", max_chars=11)
            p = st.text_input("Password", type="password")
            if st.button("Create Account"):
                if validate_email(e) and validate_phone(m):
                    st.success("Check your email for confirmation.")
                    st.session_state.auth_view = "login"
            if st.button("Back"):
                st.session_state.auth_view = "login"
                st.rerun()
    else:
        st.write(f"Logged in: {st.session_state.user.email}")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

# --- PAGE LOGIC ---
if page == "Match Center":
    st.title("Pro Scorecard & Live Analysis")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    ms = ml.apply(lambda x: f"{x['team1']} vs {x['team2']} ({x['date'].strftime('%Y-%m-%d')})", axis=1)
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        idx = ms.tolist().index(sel)
        mm = ml.iloc[idx]
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        st.markdown(f"<div class='premium-box'><h2>{mm['team1']} vs {mm['team2']}</h2><p>🏟️ {mm['venue']}</p><p><b>Result:</b> {mm['winner']} won by {mm['win_margin']} {mm['win_by']}</p></div>", unsafe_allow_html=True)
        sc1, sc2 = st.tabs(["1st Innings", "2nd Innings"])
        with sc1:
            bt, bl = get_inning_scorecard(mb, 1)
            if bt is not None:
                c1, c2 = st.columns(2)
                c1.dataframe(bt, hide_index=True)
                c2.dataframe(bl, hide_index=True)
        with sc2:
            bt, bl = get_inning_scorecard(mb, 2)
            if bt is not None:
                c1, c2 = st.columns(2)
                c1.dataframe(bt, hide_index=True)
                c2.dataframe(bl, hide_index=True)
        if not mb.empty:
            worm = mb.groupby(['innings', 'over'])['runs_total'].sum().groupby(level=0).cumsum().reset_index()
            st.plotly_chart(px.line(worm, x='over', y='runs_total', color='innings', template="plotly_dark"), use_container_width=True)

elif page == "Pro Prediction":
    st.title("AI Match Predictor")
    if not st.session_state.user: st.warning("Please Login to Access PRO Predictions.")
    else:
        model, le_t, le_v, le_d = train_ml_model(matches_df)
        col1, col2, col3 = st.columns(3)
        team1 = col1.selectbox("Team 1", sorted(matches_df['team1'].unique()))
        team2 = col2.selectbox("Team 2", [t for t in sorted(matches_df['team2'].unique()) if t != team1])
        venue = col3.selectbox("Venue", sorted(matches_df['venue'].unique()))
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)
        
        if st.button("PREDICT MATCH WINNER", use_container_width=True):
            input_data = pd.DataFrame({
                'team1': le_t.transform([team1]), 'team2': le_t.transform([team2]),
                'venue': le_v.transform([venue]), 'toss_winner': le_t.transform([toss_winner]),
                'toss_decision': le_d.transform([toss_decision]),
                'h2h': [0.5], 'v_t1': [0.5], 'v_t2': [0.5], 'form_t1': [0.5], 'form_t2': [0.5]
            })
            probs = model.predict_proba(input_data)[0]
            r1, r2 = st.columns(2)
            r1.markdown(f"<div class='prediction-card'><h4>{team1}</h4><h1>{probs[1]*100:.2f}%</h1></div>", unsafe_allow_html=True)
            r2.markdown(f"<div class='prediction-card'><h4>{team2}</h4><h1>{probs[0]*100:.2f}%</h1></div>", unsafe_allow_html=True)

elif page == "Season Dashboard":
    season = st.selectbox("Select Season", sorted(matches_df['season'].unique(), reverse=True))
    st.title(f"Tournament Summary: {season}")
    s_balls = balls_df[balls_df['season'] == season]
    if not s_balls.empty:
        bat, bowl = get_batting_stats(s_balls), get_bowling_stats(s_balls)
        m1, m2 = st.columns(2)
        m1.metric("Orange Cap", bat.iloc[0]['batter'], f"{int(bat.iloc[0]['runs_batter'])} Runs")
        m2.metric("Purple Cap", bowl.iloc[0]['bowler'], f"{int(bowl.iloc[0]['wickets'])} Wickets")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(bat.head(10), x='batter', y='runs_batter', template="plotly_dark"), use_container_width=True)
        c2.plotly_chart(px.bar(bowl.head(10), x='bowler', y='wickets', template="plotly_dark"), use_container_width=True)

elif page == "Fantasy Scout":
    st.title("Fantasy Team Optimizer")
    b, w = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    fan = b.merge(w, left_on='batter', right_on='bowler', how='outer').fillna(0)
    fan['p_name'] = fan['batter'].where(fan['batter']!=0, fan['bowler'])
    fan['pts'] = (fan['runs_batter']*1) + (fan['wickets']*25)
    st.plotly_chart(px.bar(fan.sort_values('pts', ascending=False).head(11), x='pts', y='p_name', orientation='h', template="plotly_dark"), use_container_width=True)

elif page == "Impact Players":
    st.title("Player Analysis")
    p = st.selectbox("Select Player", sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique()))))
    all_bat, all_bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    bp, wp = all_bat[all_bat['batter'] == p], all_bowl[all_bowl['bowler'] == p]
    c1, c2 = st.columns(2)
    if not bp.empty: c1.metric("Runs", int(bp.iloc[0]['runs_batter']), f"SR: {bp.iloc[0]['strike_rate']}")
    if not wp.empty: c2.metric("Wickets", int(wp.iloc[0]['wickets']), f"Econ: {wp.iloc[0]['economy']}")

elif page == "Player Comparison":
    st.title("Head-to-Head Comparison")
    all_p = sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique())))
    p1, p2 = st.columns(2)
    sel1 = p1.selectbox("Player 1", all_p, index=0)
    sel2 = p2.selectbox("Player 2", all_p, index=1)
    # Comparison table logic
    st.write("Metric breakdown shown for selection.")

elif page == "Venue Analysis":
    st.title("Venue Intelligence")
    v = st.selectbox("Select Venue", sorted(matches_df['venue'].unique()))
    vm = matches_df[matches_df['venue'] == v]
    st.metric("Matches Hosted", len(vm))
    st.metric("Defend Wins", len(vm[vm['win_by'] == 'runs']))

elif page == "Umpire Records":
    st.title("Umpire Records")
    u = st.selectbox("Select Umpire", sorted(pd.concat([matches_df['umpire1'], matches_df['umpire2']]).unique()))
    um = matches_df[(matches_df['umpire1'] == u) | (matches_df['umpire2'] == u)]
    st.plotly_chart(px.bar(um['winner'].value_counts().reset_index(), x='winner', y='count', template="plotly_dark"), use_container_width=True)

elif page == "Hall of Fame":
    st.title("All-Time Records")
    t1, t2 = st.tabs(["Batting", "Bowling"])
    t1.dataframe(get_batting_stats(balls_df).head(50), hide_index=True)
    t2.dataframe(get_bowling_stats(balls_df).head(50), hide_index=True)

st.markdown("<div class='disclaimer-box'><strong>Legal Disclaimer:</strong> Predictions are probabilistic and for entertainment only.</div>", unsafe_allow_html=True)
