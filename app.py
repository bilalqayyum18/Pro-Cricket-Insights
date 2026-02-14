import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client

# --- SUPABASE CONNECTION ---
supabase_status = "Disconnected"
supabase = None
try:
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        supabase: Client = create_client(url, key)
        supabase_status = "Connected"
except Exception:
    supabase_status = "Disconnected"

# --- CONFIG & THEME ---
st.set_page_config(page_title="Pro Cricket Insights", layout="wide", page_icon="🏏")

# --- INITIALIZE SESSION STATE ---
if 'user' not in st.session_state: st.session_state.user = None
if 'usage_left' not in st.session_state: st.session_state.usage_left = 0
if 'is_pro' not in st.session_state: st.session_state.is_pro = False
if 'auth_view' not in st.session_state: st.session_state.auth_view = "login" 

# Premium UI Polish (CSS Only)
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
    .prediction-card {
        background: #1e293b; border-radius: 12px; padding: 30px;
        border: 1px solid #334155; text-align: center;
    }
    .premium-box { background: #1e293b; border-radius: 12px; padding: 25px; border: 1px solid #334155; margin-bottom: 20px; }
    .disclaimer-box {
        background-color: #0f172a; border: 1px solid #1e293b; border-left: 4px solid #ef4444;
        padding: 24px; border-radius: 6px; margin-top: 40px; color: #94a3b8; font-size: 0.85rem;
    }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #0f172a; color: #475569; text-align: center;
        padding: 10px; font-size: 11px; border-top: 1px solid #1e293b; z-index: 1000;
    }
    .stButton>button {
        background-color: #38bdf8 !important; color: #0f172a !important;
        font-weight: 800 !important; border-radius: 8px !important;
    }
    </style>
    <div class="footer">
        <b>INDEPENDENT FAN PORTAL:</b> Not affiliated with PSL or PCB. PRO CRICKET INSIGHTS © 2026
    </div>
    """, unsafe_allow_html=True)

# --- VALIDATION LOGIC ---
def validate_identifier(identifier):
    if re.match(r'^03\d{9}$', identifier): return True
    if re.match(r'^051\d{7}$', identifier): return True
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', identifier): return True
    return False

def validate_email(email):
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def validate_phone(phone):
    if re.match(r'^03\d{9}$', phone): return True
    if re.match(r'^051\d{7}$', phone): return True
    return False

# --- DATA LOADING ---
@st.cache_data
def load_data():
    matches = pd.read_csv("psl_matches_meta_clean.csv")
    balls = pd.read_csv("psl_ball_by_ball_clean.csv")
    matches['venue'] = matches['venue'].str.split(',').str[0]
    matches['date'] = pd.to_datetime(matches['date'], dayfirst=True)
    return matches, balls

matches_df, balls_df = load_data()

# --- ML MODEL ENGINE ---
@st.cache_resource
def train_ml_model(df):
    model_df = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'date']].dropna().sort_values('date')
    
    def get_h2h_win_rate(t1, t2, date):
        relevant = df[((df['date'] < date)) & (((df['team1'] == t1) & (df['team2'] == t2)) | ((df['team1'] == t2) & (df['team2'] == t1)))]
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
        'h2h': model_df['h2h'], 'v_t1': model_df['v_t1'], 'v_t2': model_df['v_t2'],
        'form_t1': model_df['form_t1'], 'form_t2': model_df['form_t2']
    })
    
    y = (model_df['winner'] == model_df['team1']).astype(int)
    model = LogisticRegression(max_iter=1000)
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
    df_c.loc[:, 'rc'] = df_c['runs_batter'] + df_c['wide'] + df_c['noball']
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
    id_df.loc[:, 'rc_temp'] = id_df['runs_batter'] + id_df['wide'] + id_df['noball']
    r = id_df.groupby('bowler')['rc_temp'].sum().reset_index()
    bls = id_df[(id_df['wide']==0) & (id_df['noball']==0)].groupby('bowler').size().reset_index(name='bls')
    bowl = w.merge(r, on='bowler').merge(bls, on='bowler')
    bowl['O'] = ((bowl['bls']//6) + (bowl['bls']%6/10)).round(1)
    bowl['Econ'] = (bowl['rc_temp']/(bowl['bls'].replace(0,1)/6)).round(1)
    bowl['W'] = bowl['W'].astype(int); bowl['rc_temp'] = bowl['rc_temp'].astype(int)
    return bat[['batter', 'runs_batter', 'B', '4s', '6s', 'SR']], bowl[['bowler', 'O', 'rc_temp', 'W', 'Econ']]

# --- AUTHENTICATION HELPERS ---
def sync_user_data(user):
    if supabase and user:
        try:
            res = supabase.table("prediction_logs").select("usage_count, is_pro").eq("user_id", user.id).execute()
            
            if res.data and len(res.data) > 0:
                st.session_state.is_pro = res.data[0]['is_pro']
                st.session_state.usage_left = max(0, 3 - res.data[0]['usage_count'])
                return True
            else:
                try:
                    supabase.table("prediction_logs").insert({
                        "user_id": user.id, "user_identifier": user.email, "usage_count": 0, "is_pro": False
                    }).execute()
                    st.session_state.is_pro = False
                    st.session_state.usage_left = 3
                    return True
                except Exception:
                    st.error("Account initializing. Please click 'Sign In' once more.")
                    return False
        except Exception as e:
            st.error(f"Sync error: {e}")
            return False
    return False

# --- NAVIGATION ---
st.sidebar.title("Cricket Intelligence")
page = st.sidebar.radio("Navigation", ["Pro Prediction", "Season Dashboard", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame"])

# --- SUPABASE AUTH SIDEBAR ---
with st.sidebar.expander("🔐 User Account", expanded=not st.session_state.user):
    if not st.session_state.user:
        if st.session_state.auth_view == "login":
            st.subheader("Login")
            identifier = st.text_input("Email / Mobile", placeholder="03xx or email")
            password = st.text_input("Password", type="password")
            
            if st.button("Sign In", use_container_width=True):
                if validate_identifier(identifier):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": identifier, "password": password})
                        if res.user:
                            if sync_user_data(res.user):
                                st.session_state.user = res.user
                                st.rerun()
                    except Exception: st.error("Invalid Credentials or Unverified Email")
                else: st.error("Invalid Format (Mobile: 11 digits, Landline: 10 digits)")
            
            col_s, col_r = st.columns(2)
            if col_s.button("New? Sign Up"): st.session_state.auth_view = "signup"; st.rerun()
            if col_r.button("Forgot Password?"): st.session_state.auth_view = "reset"; st.rerun()

        elif st.session_state.auth_view == "signup":
            st.subheader("Create Account")
            new_email = st.text_input("Email (Required for Verification)")
            new_mobile = st.text_input("Mobile Number (03xx or 051xxx)")
            new_pass = st.text_input("New Password", type="password")
            
            if st.button("Register", use_container_width=True):
                if validate_email(new_email) and validate_phone(new_mobile):
                    try:
                        res = supabase.auth.sign_up({"email": new_email, "password": new_pass})
                        if res.user:
                            st.success("Registration successful! Check email for verification. (Please check your spam folder)")
                            st.session_state.auth_view = "login"
                    except Exception as e: st.error(f"Signup failed: {str(e)}")
                else: st.error("Invalid Format (03xx - 11 digits, 051xxx - 10 digits)")
            if st.button("Back to Login"): st.session_state.auth_view = "login"; st.rerun()

        elif st.session_state.auth_view == "reset":
            st.subheader("Reset Password")
            reset_email = st.text_input("Enter Registered Email")
            if st.button("Send Reset Link"):
                try: supabase.auth.reset_password_for_email(reset_email); st.success("Link sent!")
                except: st.error("Failed.")
            if st.button("Back"): st.session_state.auth_view = "login"; st.rerun()
    else:
        st.write(f"Logged in: {st.session_state.user.email}")
        if st.button("Logout", use_container_width=True):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

# --- PAGE LOGIC ---
if page == "Pro Prediction":
    st.title("AI Match Predictor")
    if not st.session_state.user:
        st.warning("Please Login to access AI Predictions.")
    else:
        st.info(f"Simulations Remaining: {st.session_state.usage_left}" if not st.session_state.is_pro else "💎 PRO ACCOUNT")
        model, le_t, le_v, le_d = train_ml_model(matches_df)
        with st.container():
            st.markdown("<div class='premium-box'>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            t1 = c1.selectbox("Team 1 (Bat First)", sorted(matches_df['team1'].unique()))
            t2 = c2.selectbox("Team 2 (Chase)", [t for t in sorted(matches_df['team2'].unique()) if t != t1])
            v = c3.selectbox("Venue", sorted(matches_df['venue'].unique()))
            tw = c4.selectbox("Toss Winner", [t1, t2])
            td = st.radio("Toss Decision", sorted(matches_df['toss_decision'].unique()), horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("RUN PRO SIMULATION", use_container_width=True, disabled=not (st.session_state.is_pro or st.session_state.usage_left > 0)):
            if not st.session_state.is_pro:
                current_used = 3 - st.session_state.usage_left
                new_used_count = current_used + 1
                supabase.table("prediction_logs").update({"usage_count": new_used_count}).eq("user_id", st.session_state.user.id).execute()
                sync_user_data(st.session_state.user)
            
            with st.spinner("Analyzing variables..."):
                time.sleep(1)
                h2h = len(matches_df[((matches_df['team1']==t1)&(matches_df['team2']==t2))&(matches_df['winner']==t1)]) / len(matches_df[((matches_df['team1']==t1)&(matches_df['team2']==t2))]) if len(matches_df[((matches_df['team1']==t1)&(matches_df['team2']==t2))]) > 0 else 0.5
                v1 = len(matches_df[(matches_df['venue']==v)&(matches_df['winner']==t1)]) / len(matches_df[matches_df['venue']==v]) if len(matches_df[matches_df['venue']==v]) > 0 else 0.5
                v2 = len(matches_df[(matches_df['venue']==v)&(matches_df['winner']==t2)]) / len(matches_df[matches_df['venue']==v]) if len(matches_df[matches_df['venue']==v]) > 0 else 0.5
                
                def get_form(team):
                    rel = matches_df[(matches_df['team1']==team)|(matches_df['team2']==team)].sort_values('date', ascending=False).head(5)
                    return len(rel[rel['winner']==team])/len(rel) if len(rel)>0 else 0.5

                input_df = pd.DataFrame({
                    'team1': le_t.transform([t1]), 'team2': le_t.transform([t2]), 'venue': le_v.transform([v]), 
                    'toss_winner': le_t.transform([tw]), 'toss_decision': le_d.transform([td]), 
                    'h2h': [h2h], 'v_t1': [v1], 'v_t2': [v2], 'form_t1': [get_form(t1)], 'form_t2': [get_form(t2)]
                })
                prob = model.predict_proba(input_df)[0]
                p1 = round(prob[1]*100,1)
                
            res1, res2 = st.columns(2)
            res1.markdown(f"<div class='prediction-card'><h4>{t1}</h4><h1>{p1}%</h1>Win Probability</div>", unsafe_allow_html=True)
            res2.markdown(f"<div class='prediction-card'><h4>{t2}</h4><h1>{100-p1}%</h1>Win Probability</div>", unsafe_allow_html=True)

# (Other pages Match Center, Dashboard, etc. remain unchanged)
elif page == "Match Center":
    st.title("Pro Scorecard & Live Analysis")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    ms = ml.apply(lambda x: f"{x['team1']} vs {x['team2']} ({x['date'].strftime('%Y-%m-%d')})", axis=1)
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        mm = ml.iloc[ms.tolist().index(sel)]
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        st.markdown(f"<div class='premium-box'><h2>{mm['team1']} vs {mm['team2']}</h2><p>{mm['venue']}</p><p><b>Result:</b> {mm['winner']} won</p></div>", unsafe_allow_html=True)
        t1, t2 = st.tabs(["1st Innings", "2nd Innings"])
        for i, t in enumerate([t1, t2]):
            with t:
                bt, bl = get_inning_scorecard(mb, i+1)
                if bt is not None:
                    st.dataframe(bt, hide_index=True, use_container_width=True)
                    st.dataframe(bl, hide_index=True, use_container_width=True)

elif page == "Season Dashboard":
    season = st.selectbox("Select Season", sorted(matches_df['season'].unique(), reverse=True))
    st.title(f"Tournament Summary: {season}")
    s_matches = matches_df[matches_df['season'] == season]
    winner = s_matches.sort_values('match_id').iloc[-1]['winner'] if not s_matches.empty else "N/A"
    s_balls = balls_df[balls_df['season'] == season]
    bat, bowl = get_batting_stats(s_balls), get_bowling_stats(s_balls)
    m1, m2, m3 = st.columns(3)
    m1.metric("Champion", winner)
    if not bat.empty: m2.metric("Orange Cap", bat.iloc[0]['batter'], f"{int(bat.iloc[0]['runs_batter'])} R")
    if not bowl.empty: m3.metric("Purple Cap", bowl.iloc[0]['bowler'], f"{int(bowl.iloc[0]['wickets'])} W")
    st.plotly_chart(px.bar(bat.head(10), x='batter', y='runs_batter', text='runs_batter', title="Top 10 Batters", template="plotly_dark"), use_container_width=True)

elif page == "Fantasy Scout":
    st.title("Fantasy Team Optimizer")
    season_f = st.selectbox("Data Context", sorted(matches_df['season'].unique(), reverse=True))
    sf_balls = balls_df[balls_df['season'] == season_f]
    b, w = get_batting_stats(sf_balls), get_bowling_stats(sf_balls)
    fan = b.merge(w, left_on='batter', right_on='bowler', how='outer').fillna(0)
    fan['p_name'] = fan['batter'].where(fan['batter']!=0, fan['bowler'])
    fan['pts'] = (fan['runs_batter']*1) + (fan['wickets']*25)
    st.plotly_chart(px.bar(fan.sort_values('pts', ascending=False).head(11), x='pts', y='p_name', orientation='h', title="My Dream XI", template="plotly_dark"), use_container_width=True)

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
    p1, p2 = c1.selectbox("Player 1", all_players, index=0), c2.selectbox("Player 2", all_players, index=1)
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

elif page == "Umpire Records":
    st.title("Umpire Records")
    u = st.selectbox("Select Umpire", sorted(pd.concat([matches_df['umpire1'], matches_df['umpire2']]).unique()))
    um = matches_df[(matches_df['umpire1'] == u) | (matches_df['umpire2'] == u)]
    st.plotly_chart(px.bar(um['winner'].value_counts().reset_index(), x='winner', y='count', template="plotly_dark"), use_container_width=True)

elif page == "Hall of Fame":
    st.title("All-Time Records")
    t1, t2 = st.tabs(["Batting", "Bowling"])
    with t1: st.dataframe(get_batting_stats(balls_df).head(50), use_container_width=True, hide_index=True)
    with t2: st.dataframe(get_bowling_stats(balls_df).head(50), use_container_width=True, hide_index=True)

# --- RESTORED ORIGINAL DETAILED DISCLAIMER ---
st.markdown("""
<div class='disclaimer-box'>
    <strong>Legal Disclaimer:</strong> This platform is an independent fan-developed project and is 
    NOT affiliated, associated, authorized, endorsed by, or in any way officially connected with 
    the Pakistan Super League (PSL), the Pakistan Cricket Board (PCB), or any of its teams. 
    Predictions are generated using historical data and machine learning; they do not guarantee 
    outcomes and should be used for entertainment/insight purposes only.
</div>
""", unsafe_allow_html=True)
