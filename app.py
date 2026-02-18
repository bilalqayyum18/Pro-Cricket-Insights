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
except Exception as e:
    supabase_status = f"Error: {str(e)}"

# --- CONFIG & THEME ---
st.set_page_config(page_title="Pro Cricket Insights", layout="wide", page_icon="🏏")

# --- INITIALIZE SESSION STATE ---
if 'user' not in st.session_state: st.session_state.user = None
if 'access_token' not in st.session_state: st.session_state.access_token = None
if 'usage_left' not in st.session_state: st.session_state.usage_left = 3
if 'is_pro' not in st.session_state: st.session_state.is_pro = False
if 'auth_view' not in st.session_state: st.session_state.auth_view = "login"

# --- PERSIST AUTH CONTEXT ---
if st.session_state.access_token and supabase:
    supabase.postgrest.auth(st.session_state.access_token)

# --- VALIDATION LOGIC ---
def validate_identifier(identifier):
    # Mobile: 11 digits starting with 03
    if re.match(r'^03\d{9}$', identifier): return True
    # Landline: 10 digits starting with 051
    if re.match(r'^051\d{7}$', identifier): return True
    # Standard Email
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', identifier): return True
    return False

def validate_email(email):
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def validate_phone(phone):
    # Strict validation for 03xx (11 digits) and 051 (10 digits)
    if re.match(r'^03\d{9}$', phone): return True
    if re.match(r'^051\d{7}$', phone): return True
    return False

# Premium UI Polish
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f172a; color: #f1f5f9; }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.9rem !important; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; }
    .prediction-card {
        background: #1e293b; border-radius: 12px; padding: 30px;
        border: 1px solid #334155; text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .prediction-card h4 { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 8px; }
    .prediction-card h1 { color: #38bdf8; margin: 0; font-size: 2.5rem; }
    .premium-box {
        background: #1e293b; border-radius: 12px; padding: 25px;
        border: 1px solid #334155; margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #0f172a; border: 1px solid #1e293b; border-left: 4px solid #ef4444;
        padding: 24px; border-radius: 6px; margin-top: 40px; margin-bottom: 60px;
        color: #94a3b8; font-size: 0.85rem; line-height: 1.6;
    }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #0f172a; color: #475569; text-align: center;
        padding: 10px; font-size: 11px; border-top: 1px solid #1e293b; z-index: 1000;
    }
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
    .stRadio > label { font-weight: 600 !important; color: #f1f5f9 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 4px; color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }
    .stButton>button {
        background-color: #38bdf8 !important; color: #0f172a !important;
        font-weight: 800 !important; border-radius: 8px !important;
        border: none !important; padding: 10px 24px !important;
    }
    </style>
    <div class="footer">
        <b>INDEPENDENT FAN PORTAL:</b> Not affiliated with, endorsed by, or associated with the PSL or PCB. 
        PRO CRICKET INSIGHTS © 2026 | For Analytical Purposes Only.
    </div>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    matches = pd.read_csv("psl_matches_meta_clean.csv")
    balls = pd.read_csv("psl_ball_by_ball_clean.csv")
    
    matches['venue'] = matches['venue'].str.split(',').str[0]
    matches['date'] = pd.to_datetime(matches['date'], dayfirst=True)
    
    if 'over' not in balls.columns:
        balls['over'] = balls['ball'].astype(int)
    
    if 'venue' not in balls.columns:
        venue_map = matches.set_index('match_id')['venue'].to_dict()
        balls['venue'] = balls['match_id'].map(venue_map)

    if 'extra_runs' not in balls.columns:
        w = balls['wide'] if 'wide' in balls.columns else 0
        nb = balls['noball'] if 'noball' in balls.columns else 0
        b = balls['byes'] if 'byes' in balls.columns else 0
        lb = balls['legbyes'] if 'legbyes' in balls.columns else 0
        balls['extra_runs'] = w + nb + b + lb

    return matches, balls

matches_df, balls_df = load_data()

# --- ML MODEL ENGINE ---
@st.cache_resource
def train_ml_model(df):
    model_df = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'date']]\
        .dropna()\
        .sort_values('date')

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
            (df['date'] < date) &
            (df['venue'] == venue) &
            ((df['team1'] == team) | (df['team2'] == team))
        ]
        return len(relevant[relevant['winner'] == team]) / len(relevant) if len(relevant) > 0 else 0.5

    def get_recent_form(team, date):
        relevant = df[
            (df['date'] < date) &
            ((df['team1'] == team) | (df['team2'] == team))
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
    bat = df.groupby('batter').agg({'runs_batter': 'sum', 'ball': 'count', 'wide': 'sum', 'match_id': 'nunique', 'is_wicket': 'sum'}).reset_index()
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
    nb = df_c['noball'] if 'noball' in df_c.columns else 0
    df_c['rc'] = df_c['runs_batter'] + df_c['wide'] + nb
    runs = df_c.groupby('bowler')['rc'].sum().reset_index()
    bls = df_c[(df_c['wide'] == 0) & (nb == 0)].groupby('bowler').size().reset_index(name='balls')
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
    nb = id_df['noball'] if 'noball' in id_df.columns else 0
    id_df['rc_temp'] = id_df['runs_batter'] + id_df['wide'] + nb
    r = id_df.groupby('bowler')['rc_temp'].sum().reset_index()
    bls = id_df[(id_df['wide']==0) & (nb==0)].groupby('bowler').size().reset_index(name='bls')
    bowl = w.merge(r, on='bowler').merge(bls, on='bowler')
    bowl['O'] = ((bowl['bls']//6) + (bowl['bls']%6/10)).round(1)
    bowl['Econ'] = (bowl['rc_temp']/(bowl['bls'].replace(0,1)/6)).round(1)
    bowl['W'] = bowl['W'].astype(int); bowl['rc_temp'] = bowl['rc_temp'].astype(int)
    return bat[['batter', 'runs_batter', 'B', '4s', '6s', 'SR']], bowl[['bowler', 'O', 'rc_temp', 'W', 'Econ']]

# --- NAVIGATION ---
st.sidebar.title("Pakistan League Intelligence")
page = st.sidebar.radio("Navigation", ["Season Dashboard", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame", "Pro Prediction"])

# --- AUTH / CONNECTION IN SIDEBAR ---
with st.sidebar.expander("🔐 User Account", expanded=not st.session_state.user):
    if not st.session_state.user:
        if st.session_state.auth_view == "login":
            st.subheader("Login")
            identifier = st.text_input("Email / Mobile / Landline", help="Supports 03xx (11 digits) or 051 (10 digits)")
            password = st.text_input("Password", type="password")
            
            # Form button columns
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
                                    
                                    # Fetch Pro Status safely with try-except for RLS
                                    try:
                                        profile_res = supabase.table("profiles").select("is_pro").eq("id", res.user.id).execute()
                                        
                                        if profile_res.data and len(profile_res.data) > 0:
                                            st.session_state.is_pro = profile_res.data[0].get('is_pro', False)
                                        else:
                                            # Lazy initialization if profile doesn't exist
                                            supabase.table("profiles").insert({
                                                "id": res.user.id,
                                                "identifier": identifier,
                                                "is_pro": False
                                            }).execute()
                                            st.session_state.is_pro = False
                                    except Exception as profile_err:
                                        # If RLS blocks the check, default to non-pro but don't crash login
                                        st.session_state.is_pro = False
                                        st.sidebar.warning("Note: Profile access restricted by database policies.")
                                    
                                    st.success("Logged in successfully!")
                                    st.rerun()
                                else:
                                    st.error("Invalid Credentials")
                            except Exception as e: 
                                st.error(f"Login Failed: {str(e)}")
                        else:
                            st.error("Supabase connection not established.")
                    else: 
                        st.error("Invalid format. Use 11-digit Mobile (03xx) or 10-digit Landline (051xx).")
            
            with col_pro:
                st.link_button("Request Pro Access", "https://forms.gle/pQSrUN1TcXdTD4nXA", use_container_width=True)

            if st.button("Sign Up"): st.session_state.auth_view = "signup"; st.rerun()
            
        elif st.session_state.auth_view == "signup":
            st.subheader("Register")
            e = st.text_input("Email")
            m = st.text_input("Mobile/Landline", max_chars=11, help="Mobile: 11 digits (03xx) | Landline: 10 digits (051xxx)")
            p = st.text_input("Password", type="password")
            if st.button("Create Account"):
                if validate_email(e) and validate_phone(m):
                    if supabase:
                        try:
                            # Using professional auth flow
                            res = supabase.auth.sign_up({
                                "email": e, 
                                "password": p, 
                                "options": {
                                    "data": {"phone_number": m}
                                }
                            })
                            st.success("Registration initiated successfully.")
                            st.info(f"A confirmation link has been sent to **{e}**. Please verify your email to activate your account.")
                            st.session_state.auth_view = "login"
                        except Exception as ex: 
                            st.error(f"Error creating account: {str(ex)}")
                    else:
                        st.error("Supabase connection not established.")
                else: 
                    st.error("Validation Failed: Mobile must be 11 digits (03xx) | Landline 10 digits (051xxx)")
            if st.button("Back"): st.session_state.auth_view = "login"; st.rerun()
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
    ms = ml.apply(lambda x: f"{x['team1']} vs {x['team2']} ({x['date'].strftime('%Y-%m-%d')})", axis=1)
    
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        idx = ms.tolist().index(sel)
        mm = ml.iloc[idx]
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        
        st.markdown(f"""
        <div class="premium-box" style="border-left: 5px solid #38bdf8;">
            <h2 style='margin:0;'>{mm['team1']} vs {mm['team2']}</h2>
            <p style='color:#94a3b8; font-size:1.1rem; margin:5px 0;'>🏟️ {mm['venue']} | 📅 {mm['date'].strftime('%Y-%m-%d')}</p>
            <hr style='border-color:#334155;'>
            <p><b>Toss:</b> {mm['toss_winner']} won and chose to {mm['toss_decision']}</p>
            <p style='font-size:1.2rem; color:#38bdf8;'><b>Result:</b> {mm['winner']} won by {mm['win_margin']} {mm['win_by']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Scorecard Summary")
        sc1, sc2 = st.tabs([f"1st Innings", f"2nd Innings"])
        
        with sc1:
            bt, bl = get_inning_scorecard(mb, 1)
            if bt is not None:
                c1, c2 = st.columns(2)
                c1.markdown("**Batting**"); c1.dataframe(bt, use_container_width=True, hide_index=True)
                c2.markdown("**Bowling**"); c2.dataframe(bl, use_container_width=True, hide_index=True)
        with sc2:
            bt, bl = get_inning_scorecard(mb, 2)
            if bt is not None:
                c1, c2 = st.columns(2)
                c1.markdown("**Batting**"); c1.dataframe(bt, use_container_width=True, hide_index=True)
                c2.markdown("**Bowling**"); c2.dataframe(bl, use_container_width=True, hide_index=True)

        mb_c = mb.copy()
        mb_c['runs_total_ball'] = mb_c['runs_batter'] + mb_c['extra_runs']
        worm = mb_c.groupby(['innings', 'over'])['runs_total_ball'].sum().groupby(level=0).cumsum().reset_index()
        # Cleaned labels for Match Worm
        fig_worm = px.line(worm, x='over', y='runs_total_ball', color='innings', 
                           title="Match Progression", template="plotly_dark",
                           labels={'over': 'Overs Completed', 'runs_total_ball': 'Cumulative Runs', 'innings': 'Innings No.'})
        st.plotly_chart(fig_worm, use_container_width=True)

elif page == "Pro Prediction":
    st.title("AI Match Predictor")
    
    if not st.session_state.user:
        st.warning("Please Login to Access PRO Predictions.")
    else:
        user_id = st.session_state.user.id
        can_predict, usage_left = True, 3
        
        if supabase:
            try:
                # 1. Fetch current Pro status safely from profiles
                profile_res = supabase.table("profiles").select("is_pro").eq("id", user_id).execute()
                if profile_res.data and len(profile_res.data) > 0:
                    st.session_state.is_pro = profile_res.data[0].get('is_pro', False)

                # 2. Count attempts in the LAST 24 HOURS from prediction_attempts
                time_threshold = (datetime.now() - timedelta(hours=24)).isoformat()
                usage_res = supabase.table("prediction_attempts")\
                    .select("id", count="exact")\
                    .eq("user_id", user_id)\
                    .gt("created_at", time_threshold)\
                    .execute()
                
                attempts_count = usage_res.count if usage_res.count is not None else 0
                usage_left = max(0, 3 - attempts_count)
                
                if not st.session_state.is_pro and attempts_count >= 3:
                    can_predict = False

                if st.session_state.is_pro:
                    can_predict = True
                    usage_left = "Unlimited"

            except Exception as e:
                st.sidebar.error(f"Read Error: {e}")

        if not can_predict:
            st.error("Account Limit: You have reached 3 simulations in the last 24 hours. Upgrade to PRO for unlimited access.")
        else:
            if st.session_state.is_pro:
                st.success(f"Welcome! You have Unlimited Simulations.")
            else:
                st.info(f"Welcome! You have {usage_left} Simulations Remaining (Resets every 24h).")
            
            st.info("Initializing AI model - please allow up to 10 seconds on first load.")

            with st.spinner("Loading historical models and encoders…"):
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
                # --- LOGGING BLOCK FIX ---
                if supabase:
                    try:
                        # Log detailed attempt
                        supabase.table("prediction_attempts").insert({
                            "user_id": user_id,
                            "metadata": {"team1": team1, "team2": team2, "venue": venue}
                        }).execute()
                        
                        # Sync with prediction_logs for aggregate tracking
                        # Using upsert logic to ensure prediction_logs records data
                        supabase.table("prediction_logs").upsert({
                            "user_id": user_id,
                            "user_identifier": st.session_state.user.email,
                            "is_pro": st.session_state.is_pro,
                            "usage_count": attempts_count + 1
                        }, on_conflict="user_id").execute()
                        
                        if not st.session_state.is_pro:
                            usage_left = max(0, 3 - (attempts_count + 1))
                            
                    except Exception as log_error:
                        st.error(f"Database Error: {log_error}")
                        st.info("Check RLS policies and table permissions in Supabase.")
                        st.stop()

                # --- SIMULATION LOGIC ---
                with st.spinner("Running Prediction Model: Analyzing Historical Variables..."):
                    time.sleep(1)
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
                        'team1': le_t.transform([team1]), 'team2': le_t.transform([team2]),
                        'venue': le_v.transform([venue]), 'toss_winner': le_t.transform([toss_winner]),
                        'toss_decision': le_d.transform([toss_decision]), 'h2h': [h2h_val],
                        'v_t1': [v_t1_val], 'v_t2': [v_t2_val], 'form_t1': [live_form(team1)], 'form_t2': [live_form(team2)]
                    })
                    
                    probs = model.predict_proba(input_data)[0]
                    t1_prob = round(probs[1] * 100, 1)
                    t2_prob = round(100 - t1_prob, 1)
                    
                    st.markdown("### AI Predicted Probability")
                    res1, res2 = st.columns(2)
                    res1.markdown(f"<div class='prediction-card'><h4>{team1}</h4><h1>{t1_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)
                    res2.markdown(f"<div class='prediction-card'><h4>{team2}</h4><h1>{t2_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)
                    
                    if st.session_state.is_pro:
                        st.info("Simulations Remaining: Unlimited")
                    else:
                        st.info(f"Simulations Remaining: {usage_left}")
                    
                    # NOTE: Removed st.rerun() from here to prevent the UI from clearing results immediately.
                    # This allows the user to see the prediction results.

elif page == "Season Dashboard":
    season = st.selectbox("Select Season", sorted(matches_df['season'].unique(), reverse=True))
    st.title(f"Tournament Summary: {season}")
    s_balls = balls_df[balls_df['season'] == season]
    bat, bowl = get_batting_stats(s_balls), get_bowling_stats(s_balls)
    
    # MVP Calculation (Simple weighted score)
    mvp = s_balls.groupby('batter').agg({'runs_batter': 'sum', 'is_wicket': 'sum'}).reset_index()
    mvp['score'] = (mvp['runs_batter'] * 1) + (mvp['is_wicket'] * 25)
    mvp = mvp.sort_values('score', ascending=False).head(10)

    m1, m2, m3 = st.columns(3)
    if not bat.empty: m1.metric("Orange Cap", bat.iloc[0]['batter'], f"{int(bat.iloc[0]['runs_batter'])} Runs")
    if not bowl.empty: m2.metric("Purple Cap", bowl.iloc[0]['bowler'], f"{int(bowl.iloc[0]['wickets'])} Wickets")
    if not mvp.empty: m3.metric("Top MVP", mvp.iloc[0]['batter'], f"{int(mvp.iloc[0]['score'])} Pts")

    st.subheader("Top Performers")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(bat.head(10), x='batter', y='runs_batter', title="Top 10 Batters", template="plotly_dark", labels={'runs_batter': 'Runs', 'batter': 'Player'}), use_container_width=True)
    c2.plotly_chart(px.bar(bowl.head(10), x='bowler', y='wickets', title="Top 10 Bowlers", template="plotly_dark", labels={'wickets': 'Wickets', 'bowler': 'Player'}), use_container_width=True)
    st.plotly_chart(px.bar(mvp, x='batter', y='score', title="Top 10 MVP Impact", template="plotly_dark", labels={'score': 'Impact Score', 'batter': 'Player'}), use_container_width=True)

elif page == "Fantasy Scout":
    st.title("Fantasy Team Optimizer")
    season_f = st.selectbox("Data Context", sorted(matches_df['season'].unique(), reverse=True))
    sf_balls = balls_df[balls_df['season'] == season_f]
    b, w = get_batting_stats(sf_balls), get_bowling_stats(sf_balls)
    fan = b.merge(w, left_on='batter', right_on='bowler', how='outer').fillna(0)
    fan['p_name'] = fan['batter'].where(fan['batter']!=0, fan['bowler'])
    fan['pts'] = (fan['runs_batter']*1) + (fan['wickets']*25)
    # Cleaned labels for Fantasy Scout
    fig_fan = px.bar(fan.sort_values('pts', ascending=False).head(11), x='pts', y='p_name', 
                     orientation='h', title="Optimal Performance Scout", template="plotly_dark",
                     labels={'pts': 'Performance Index', 'p_name': 'Player Name'})
    st.plotly_chart(fig_fan, use_container_width=True)

elif page == "Impact Players":
    st.title("Player Analysis & Rankings")
    p = st.selectbox("Select Player", sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique()))))
    all_bat, all_bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    ca, cb = st.columns(2)
    bp = all_bat[all_bat['batter'] == p]
    if not bp.empty:
        with ca:
            st.metric("Total Runs", int(bp.iloc[0]['runs_batter']))
            st.metric("Strike Rate", bp.iloc[0]['strike_rate'])
    wp = all_bowl[all_bowl['bowler'] == p]
    if not wp.empty:
        with cb:
            st.metric("Total Wickets", int(wp.iloc[0]['wickets']))
            st.metric("Economy", wp.iloc[0]['economy'])

elif page == "Player Comparison":
    st.title("Head-to-Head Comparison")
    all_players = sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique())))
    c1, c2 = st.columns(2)
    p1, p2 = c1.selectbox("Player 1", all_players, index=0), c2.selectbox("Player 2", all_players, index=min(1, len(all_players)-1))
    bat_all, bowl_all = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    def get_p_stats(name):
        b, w = bat_all[bat_all['batter'] == name], bowl_all[bowl_all['bowler'] == name]
        return {'Runs': int(b.iloc[0]['runs_batter']) if not b.empty else 0, 'SR': b.iloc[0]['strike_rate'] if not b.empty else 0.0,
                'Wickets': int(w.iloc[0]['wickets']) if not w.empty else 0, 'Econ': w.iloc[0]['economy'] if not w.empty else 0.0}
    s1, s2 = get_p_stats(p1), get_p_stats(p2)
    st.table(pd.DataFrame({'Metric': ['Runs', 'SR', 'Wickets', 'Econ'], p1: [s1['Runs'], s1['SR'], s1['Wickets'], s1['Econ']], p2: [s2['Runs'], s2['SR'], s2['Wickets'], s2['Econ']]}))

elif page == "Venue Analysis":
    st.title("Venue Intelligence")
    v = st.selectbox("Select Venue", sorted(matches_df['venue'].unique()))
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
    # Cleaned labels for Umpire Records
    fig_ump = px.bar(um['winner'].value_counts().reset_index(), x='winner', y='count', 
                     template="plotly_dark", labels={'winner': 'Winning Team', 'count': 'Match Count'})
    st.plotly_chart(fig_ump, use_container_width=True)

elif page == "Hall of Fame":
    st.title("All-Time Records")
    t1, t2 = st.tabs(["Batting", "Bowling"])
    with t1: st.dataframe(get_batting_stats(balls_df).head(50), use_container_width=True, hide_index=True)
    with t2: st.dataframe(get_bowling_stats(balls_df).head(50), use_container_width=True, hide_index=True)

# --- GLOBAL LEGAL DISCLAIMER ---
st.markdown("""
    <div class="disclaimer-box">
        <strong>Legal Disclaimer & Terms of Use:</strong><br>
        This platform is an <strong>independent fan-led project</strong> and is not affiliated with, endorsed by, or 
        associated with the Pakistan Super League (PSL), the Pakistan Cricket Board (PCB), or any specific cricket 
        franchise. All trademarks and copyrights belong to their respective owners.<br><br>
        This tool utilizes Machine Learning (ML) to generate probabilistic outcomes based on historical data. 
        These predictions are for <strong>informational and entertainment purposes only</strong> and do not provide 
        any guarantees regarding match results.
    </div>
    """, unsafe_allow_html=True)
