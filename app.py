import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import re
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client

# --- SUPABASE CONNECTION ---
supabase_status = "Disconnected"
supabase: Client = None

try:
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"] 
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

if st.session_state.access_token and supabase:
    supabase.postgrest.auth(st.session_state.access_token)

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
        balls['over'] = (balls['ball']).astype(int)
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

# --- PLAYER ARCHETYPES LOGIC ---
def get_player_archetype(name, df):
    b_stats = df[df['batter'] == name]
    w_stats = df[df['bowler'] == name]
    
    archetype = "All-Rounder"
    if b_stats.empty and not w_stats.empty:
        # Bowling Logic
        death_overs = w_stats[w_stats['over'] >= 16]
        if not death_overs.empty and (death_overs['is_wicket'].sum() > 2):
            archetype = "Death Over Specialist"
        else:
            archetype = "Bowling Asset"
    elif not b_stats.empty:
        # Batting Logic
        sr = (b_stats['runs_batter'].sum() / (b_stats.shape[0] - b_stats['wide'].sum()).replace(0,1)) * 100
        pp_runs = b_stats[b_stats['over'] <= 6]['runs_batter'].sum()
        if pp_runs > 50 and sr > 140:
            archetype = "Powerplay Enforcer"
        elif sr > 150:
            archetype = "Finisher"
        else:
            archetype = "Anchor"
    return archetype

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
    b_stats['SR'] = (b_stats['runs_batter'] / (b_stats['B'].replace(0, 1)) * 100).round(2)
    f = id_df[id_df['runs_batter'] == 4].groupby('batter').size().reset_index(name='4s')
    s = id_df[id_df['runs_batter'] == 6].groupby('batter').size().reset_index(name='6s')
    bat = b_stats.merge(f, on='batter', how='left').merge(s, on='batter', how='left').fillna(0)
    
    bw = id_df[~id_df['wicket_kind'].isin(['run out', 'retired hurt'])]
    w = bw.groupby('bowler')['is_wicket'].sum().reset_index().rename(columns={'is_wicket':'W'})
    nb = id_df['noball'] if 'noball' in id_df.columns else 0
    id_df['rc_temp'] = id_df['runs_batter'] + id_df['wide'] + nb
    r = id_df.groupby('bowler')['rc_temp'].sum().reset_index()
    bls = id_df[(id_df['wide']==0) & (nb==0)].groupby('bowler').size().reset_index(name='bls')
    bowl = w.merge(r, on='bowler').merge(bls, on='bowler')
    bowl['O'] = ((bowl['bls']//6) + (bowl['bls']%6/10)).round(1)
    bowl['Econ'] = (bowl['rc_temp']/(bowl['bls'].replace(0,1)/6)).round(2)
    return bat[['batter', 'runs_batter', 'B', '4s', '6s', 'SR']], bowl[['bowler', 'O', 'rc_temp', 'W', 'Econ']]

# --- UPGRADED ML MODEL ENGINE ---
@st.cache_resource
def train_ml_model(df):
    model_df = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'date']].dropna().copy()
    
    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    le_decision = LabelEncoder()
    
    all_teams = pd.concat([model_df['team1'], model_df['team2']]).unique()
    le_team.fit(all_teams)
    le_venue.fit(model_df['venue'].unique())
    le_decision.fit(model_df['toss_decision'].unique())
    
    def get_h2h(t1, t2, dt):
        rel = df[((df['date'] < dt)) & (((df['team1'] == t1) & (df['team2'] == t2)) | ((df['team1'] == t2) & (df['team2'] == t1)))]
        return len(rel[rel['winner'] == t1]) / len(rel) if len(rel) > 0 else 0.5

    model_df['h2h'] = model_df.apply(lambda x: get_h2h(x['team1'], x['team2'], x['date']), axis=1)
    
    X = pd.DataFrame({
        'team1': le_team.transform(model_df['team1']),
        'team2': le_team.transform(model_df['team2']),
        'venue': le_venue.transform(model_df['venue']),
        'toss_winner': le_team.transform(model_df['toss_winner']),
        'toss_decision': le_decision.transform(model_df['toss_decision']),
        'h2h': model_df['h2h']
    })
    y = (model_df['winner'] == model_df['team1']).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_team, le_venue, le_decision

# --- NAVIGATION ---
st.sidebar.title("Cricket Intelligence")
page = st.sidebar.radio("Navigation", ["Season Dashboard", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame", "Pro Prediction"])

# --- AUTH IN SIDEBAR ---
with st.sidebar.expander("🔐 User Account", expanded=not st.session_state.user):
    if not st.session_state.user:
        if st.session_state.auth_view == "login":
            st.subheader("Login")
            identifier = st.text_input("Email / Mobile / Landline")
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
                                    st.session_state.is_pro = False
                                    try:
                                        p_res = supabase.table("profiles").select("is_pro").eq("id", res.user.id).execute()
                                        if p_res.data: st.session_state.is_pro = p_res.data[0].get('is_pro', False)
                                    except: pass
                                    st.rerun()
                            except Exception as e: st.error(f"Login Failed: {e}")
            with col_pro: st.link_button("Request Pro", "https://forms.gle/pQSrUN1TcXdTD4nXA")
            if st.button("Sign Up"): st.session_state.auth_view = "signup"; st.rerun()
        elif st.session_state.auth_view == "signup":
            st.subheader("Register")
            e, m, p = st.text_input("Email"), st.text_input("Mobile/Landline"), st.text_input("Password", type="password")
            if st.button("Create Account"):
                if validate_email(e) and validate_phone(m):
                    try:
                        supabase.auth.sign_up({"email": e, "password": p, "options": {"data": {"phone": m}}})
                        st.success("Verification email sent!"); st.session_state.auth_view = "login"
                    except Exception as ex: st.error(str(ex))
            if st.button("Back"): st.session_state.auth_view = "login"; st.rerun()
    else:
        st.write(f"Logged in as: {st.session_state.user.email}")
        if st.button("Logout"): 
            st.session_state.user = None; st.rerun()

# --- PAGE LOGIC ---
if page == "Match Center":
    st.title("Pro Scorecard & Phase Impact")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    ms = ml.apply(lambda x: f"{x['team1']} vs {x['team2']} ({x['date'].strftime('%Y-%m-%d')})", axis=1)
    
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        mm = ml.iloc[ms.tolist().index(sel)]
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        
        st.markdown(f"<div class='premium-box' style='border-left: 5px solid #38bdf8;'><h2>{mm['team1']} vs {mm['team2']}</h2><p>{mm['venue']} | {mm['winner']} won</p></div>", unsafe_allow_html=True)

        sc1, sc2, sc3 = st.tabs(["1st Innings", "2nd Innings", "Phase Impact"])
        with sc1:
            bt, bl = get_inning_scorecard(mb, 1)
            if bt is not None:
                c1, c2 = st.columns(2); c1.dataframe(bt, hide_index=True); c2.dataframe(bl, hide_index=True)
        with sc2:
            bt, bl = get_inning_scorecard(mb, 2)
            if bt is not None:
                c1, c2 = st.columns(2); c1.dataframe(bt, hide_index=True); c2.dataframe(bl, hide_index=True)
        with sc3:
            # Phase Impact Logic
            mb['phase'] = pd.cut(mb['over'], bins=[0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death'])
            phase_data = mb.groupby(['innings', 'phase'])['runs_batter'].sum().reset_index()
            fig_phase = px.bar(phase_data, x='phase', y='runs_batter', color='innings', barmode='group', 
                               title="Scoring by Match Phase", template="plotly_dark",
                               labels={'runs_batter': 'Runs', 'phase': 'Match Phase'})
            st.plotly_chart(fig_phase, use_container_width=True)

elif page == "Pro Prediction":
    st.title("AI Match Predictor (RandomForest)")
    if not st.session_state.user:
        st.warning("Please Login to access AI Predictions.")
    else:
        model, le_t, le_v, le_d = train_ml_model(matches_df)
        with st.container():
            st.markdown("<div class='premium-box'>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            t1 = c1.selectbox("Team 1", sorted(matches_df['team1'].unique()))
            t2 = c2.selectbox("Team 2", [t for t in sorted(matches_df['team2'].unique()) if t != t1])
            ven = c3.selectbox("Venue", sorted(matches_df['venue'].unique()))
            
            # New Inputs
            p_col, w_col, toss_col = st.columns(3)
            pitch = p_col.selectbox("Pitch Condition", ["Flat", "Green", "Dry/Dusty", "Balanced"])
            weather = w_col.selectbox("Weather", ["Clear", "Overcast", "Humid"])
            t_win = toss_col.selectbox("Toss Winner", [t1, t2])
            t_dec = st.radio("Toss Decision", ["bat", "field"], horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("RUN PRO SIMULATION"):
            with st.spinner("Calculating variables..."):
                time.sleep(1)
                input_data = pd.DataFrame({
                    'team1': le_t.transform([t1]), 'team2': le_t.transform([t2]),
                    'venue': le_v.transform([ven]), 'toss_winner': le_t.transform([t_win]),
                    'toss_decision': le_d.transform([t_dec]), 'h2h': [0.5]
                })
                probs = model.predict_proba(input_data)[0]
                r1, r2 = st.columns(2)
                r1.markdown(f"<div class='prediction-card'><h4>{t1}</h4><h1>{round(probs[1]*100,1)}%</h1></div>", unsafe_allow_html=True)
                r2.markdown(f"<div class='prediction-card'><h4>{t2}</h4><h1>{round(probs[0]*100,1)}%</h1></div>", unsafe_allow_html=True)

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
    top_11 = fan.sort_values('pts', ascending=False).head(11)

    st.subheader("Interactive Candidate Comparison")
    selected_players = st.multiselect("Select Players to Compare", top_11['p_name'].tolist(), default=top_11['p_name'].tolist()[:3])
    
    if selected_players:
        comp_df = fan[fan['p_name'].isin(selected_players)]
        fig = go.Figure()
        for _, row in comp_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['runs_batter']/10, row['strike_rate']/10, row['wickets']*5, row['economy']*2],
                theta=['Runs/10', 'SR/10', 'Wickets*5', 'Econ*2'],
                fill='toself', name=row['p_name']
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Impact Players":
    st.title("Player Analysis & Archetypes")
    p = st.selectbox("Select Player", sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique()))))
    arch = get_player_archetype(p, balls_df)
    st.info(f"Archetype Identified: **{arch}**")
    all_bat, all_bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    ca, cb = st.columns(2)
    bp = all_bat[all_bat['batter'] == p]
    if not bp.empty:
        with ca:
            st.metric("Total Runs", int(bp.iloc[0]['runs_batter']))
            st.metric("Strike Rate", round(float(bp.iloc[0]['strike_rate']), 2))
    wp = all_bowl[all_bowl['bowler'] == p]
    if not wp.empty:
        with cb:
            st.metric("Total Wickets", int(wp.iloc[0]['wickets']))
            st.metric("Economy", round(float(wp.iloc[0]['economy']), 2))

elif page == "Player Comparison":
    st.title("Head-to-Head Comparison")
    all_p = sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique())))
    c1, c2 = st.columns(2)
    p1, p2 = c1.selectbox("Player 1", all_p, index=0), c2.selectbox("Player 2", all_p, index=1)
    b_all, w_all = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    def stats(n):
        b, w = b_all[b_all['batter']==n], w_all[w_all['bowler']==n]
        return [int(b.iloc[0]['runs_batter']) if not b.empty else 0, round(b.iloc[0]['strike_rate'],2) if not b.empty else 0.0,
                int(w.iloc[0]['wickets']) if not w.empty else 0, round(w.iloc[0]['economy'],2) if not w.empty else 0.0]
    st.table(pd.DataFrame({'Metric': ['Runs', 'SR', 'Wickets', 'Econ'], p1: stats(p1), p2: stats(p2)}))

elif page == "Venue Analysis":
    st.title("Venue Intelligence")
    v = st.selectbox("Select Venue", sorted(matches_df['venue'].unique()))
    vm = matches_df[matches_df['venue'] == v]
    st.metric("Matches Hosted", len(vm))
    v_balls = balls_df[balls_df['venue'] == v]
    if not v_balls.empty:
        avg = v_balls[v_balls['innings']==1].groupby('match_id')['runs_batter'].sum().mean()
        st.metric("Avg 1st Innings", round(avg, 2))

elif page == "Umpire Records":
    st.title("Umpire Records")
    u = st.selectbox("Select Umpire", sorted(pd.concat([matches_df['umpire1'], matches_df['umpire2']]).unique()))
    um = matches_df[(matches_df['umpire1'] == u) | (matches_df['umpire2'] == u)]
    st.plotly_chart(px.bar(um['winner'].value_counts().reset_index(), x='winner', y='count', template="plotly_dark", labels={'count':'Wins','winner':'Team'}), use_container_width=True)

elif page == "Hall of Fame":
    st.title("All-Time Records")
    t1, t2 = st.tabs(["Batting", "Bowling"])
    with t1: st.dataframe(get_batting_stats(balls_df).head(50), use_container_width=True, hide_index=True)
    with t2: st.dataframe(get_bowling_stats(balls_df).head(50), use_container_width=True, hide_index=True)

st.markdown("""<div class='disclaimer-box'><strong>Legal Disclaimer:</strong> Independent fan portal. Predictions are probabilistic and for entertainment only. PRO CRICKET INSIGHTS © 2026</div>""", unsafe_allow_html=True)
