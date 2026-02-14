import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client

# --- SUPABASE CONNECTION & DEBUGGING ---
supabase_status = "Not Initialized"
supabase_error = None

try:
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        supabase: Client = create_client(url, key)
        supabase_status = "Connected"
    else:
        supabase = None
        supabase_status = "Missing Secrets"
except Exception as e:
    supabase = None
    supabase_status = "Connection Error"
    supabase_error = str(e)

# --- CONFIG & THEME (Original UI) ---
st.set_page_config(page_title="Pro Cricket Insights", layout="wide", page_icon="🏏")

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
    .premium-box { background: #1e293b; border-radius: 12px; padding: 25px; border: 1px solid #334155; margin-bottom: 20px; }
    .disclaimer-box {
        background-color: #0f172a; border: 1px solid #1e293b; border-left: 4px solid #ef4444;
        padding: 24px; border-radius: 6px; margin-top: 40px; margin-bottom: 60px; color: #94a3b8; font-size: 0.85rem; line-height: 1.6;
    }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #0f172a; color: #475569; text-align: center;
        padding: 10px; font-size: 11px; border-top: 1px solid #1e293b; z-index: 1000;
    }
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
    .stButton>button {
        background-color: #38bdf8 !important; color: #0f172a !important;
        font-weight: 800 !important; border-radius: 8px !important; border: none !important; padding: 10px 24px !important;
    }
    .paywall-box {
        background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444;
        padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;
    }
    </style>
    <div class="footer">
        <b>INDEPENDENT FAN PORTAL:</b> Not affiliated with, endorsed by, or associated with the PSL or PCB. 
        PRO CRICKET INSIGHTS © 2026 | For Analytical Purposes Only.
    </div>
    """, unsafe_allow_html=True)

# --- DATA LOADING (Original) ---
@st.cache_data
def load_data():
    matches = pd.read_csv("psl_matches_meta_clean.csv")
    balls = pd.read_csv("psl_ball_by_ball_clean.csv")
    matches['venue'] = matches['venue'].str.split(',').str[0]
    matches['date'] = pd.to_datetime(matches['date'], dayfirst=True)
    return matches, balls

matches_df, balls_df = load_data()

# --- ML MODEL ENGINE (Original) ---
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

    le_team, le_venue, le_decision = LabelEncoder(), LabelEncoder(), LabelEncoder()
    le_team.fit(pd.concat([model_df['team1'], model_df['team2']]).unique())
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
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return model, le_team, le_venue, le_decision

# --- STATS HELPERS (Original) ---
def get_batting_stats(df):
    if df.empty: return pd.DataFrame()
    bat = df.groupby('batter').agg({'runs_batter': 'sum', 'ball': 'count', 'wide': 'sum', 'match_id': 'nunique', 'is_wicket': 'sum'}).reset_index()
    bat['balls_faced'] = bat['ball'] - bat['wide']
    bat['strike_rate'] = (bat['runs_batter'] / (bat['balls_faced'].replace(0, 1)) * 100).round(2)
    f = df[df['runs_batter'] == 4].groupby('batter').size().reset_index(name='4s')
    s = df[df['runs_batter'] == 6].groupby('batter').size().reset_index(name='6s')
    res = bat.merge(f, on='batter', how='left').merge(s, on='batter', how='left').fillna(0)
    for col in ['runs_batter', 'balls_faced', '4s', '6s', 'match_id']: res[col] = res[col].astype(int)
    return res.sort_values('runs_batter', ascending=False)

def get_bowling_stats(df):
    if df.empty: return pd.DataFrame()
    bw = df[~df['wicket_kind'].isin(['run out', 'retired hurt'])]
    wickets = bw.groupby('bowler')['is_wicket'].sum().reset_index().rename(columns={'is_wicket': 'wickets'})
    df_c = df.copy()
    df_c.loc[:, 'rc'] = df_c['runs_batter'] + df_c['wide'] + df_c['noball']
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
    id_df.loc[:, 'rc_temp'] = id_df['runs_batter'] + id_df['wide'] + id_df['noball']
    r = id_df.groupby('bowler')['rc_temp'].sum().reset_index()
    bls = id_df[(id_df['wide']==0) & (id_df['noball']==0)].groupby('bowler').size().reset_index(name='bls')
    bowl = w.merge(r, on='bowler').merge(bls, on='bowler')
    bowl['O'] = ((bowl['bls']//6) + (bowl['bls']%6/10)).round(1)
    bowl['Econ'] = (bowl['rc_temp']/(bowl['bls'].replace(0,1)/6)).round(1)
    return bat[['batter', 'runs_batter', 'B', '4s', '6s', 'SR']], bowl[['bowler', 'O', 'rc_temp', 'W', 'Econ']]

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Season Dashboard", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame", "Pro Prediction"])

# --- DEBUG & REVIEWS ---
with st.sidebar.expander("🛠️ Connection Status"):
    st.write(f"Supabase: {supabase_status}")
    if supabase_error: st.error(supabase_error)
    if supabase and st.button("Refresh Log View"):
        try:
            logs = supabase.table("prediction_logs").select("*").execute()
            st.dataframe(logs.data)
        except Exception as e: st.write(f"Fetch failed: {e}")

# --- PAGE LOGIC ---
if page == "Pro Prediction":
    st.title("AI Match Predictor")
    
    # Simple ID for tracking (can be changed to user login later)
    user_id = "guest_session_001"
    can_predict, usage_left = True, 3

    if supabase:
        try:
            res = supabase.table("prediction_logs").select("*").eq("user_ip", user_id).execute()
            if res.data:
                count = res.data[0]['usage_count']
                usage_left = max(0, 3 - count)
                if count >= 3: can_predict = False
            else:
                # If they aren't in the DB yet, we'll insert when they first run a simulation
                usage_left = 3
        except Exception as e: st.sidebar.error(f"Read Error: {e}")

    if not can_predict:
        st.markdown("<div class='paywall-box'><h3>Free Limit Reached</h3><p>Upgrade to Pro for unlimited AI simulations.</p></div>", unsafe_allow_html=True)
    else:
        st.info(f"Account: {user_id} | {usage_left} Simulations Remaining Today.")
        model, le_t, le_v, le_d = train_ml_model(matches_df)
        
        with st.container():
            st.markdown("<div class='premium-box'>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            team1 = col1.selectbox("Team 1", sorted(matches_df['team1'].unique()))
            team2 = col2.selectbox("Team 2", [t for t in sorted(matches_df['team2'].unique()) if t != team1])
            venue = col3.selectbox("Venue", sorted(matches_df['venue'].unique()))
            toss_winner = col4.selectbox("Toss Winner", [team1, team2])
            toss_decision = st.radio("Toss Decision", sorted(matches_df['toss_decision'].unique()), horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("RUN PRO SIMULATION", use_container_width=True):
            if supabase:
                try:
                    # Check if user exists to update, otherwise insert
                    check = supabase.table("prediction_logs").select("usage_count").eq("user_ip", user_id).execute()
                    if check.data:
                        new_count = check.data[0]['usage_count'] + 1
                        supabase.table("prediction_logs").update({"usage_count": new_count}).eq("user_ip", user_id).execute()
                    else:
                        supabase.table("prediction_logs").insert({"user_ip": user_id, "usage_count": 1}).execute()
                except Exception as e: st.error(f"Log error: {e}")

            with st.spinner("Analyzing Historical Data..."):
                time.sleep(1)
                h2h_matches = matches_df[((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) | ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))]
                h2h_val = len(h2h_matches[h2h_matches['winner'] == team1]) / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
                
                input_data = pd.DataFrame({
                    'team1': le_t.transform([team1]), 'team2': le_t.transform([team2]),
                    'venue': le_v.transform([venue]), 'toss_winner': le_t.transform([toss_winner]),
                    'toss_decision': le_d.transform([toss_decision]), 'h2h': [h2h_val],
                    'v_t1': [0.5], 'v_t2': [0.5], 'form_t1': [0.5], 'form_t2': [0.5]
                })
                
                probs = model.predict_proba(input_data)[0]
                t1_p, t2_p = round(probs[1]*100, 1), round(probs[0]*100, 1)
                res1, res2 = st.columns(2)
                res1.markdown(f"<div class='prediction-card'><h4>{team1}</h4><h1>{t1_p}%</h1>Win Prob</div>", unsafe_allow_html=True)
                res2.markdown(f"<div class='prediction-card'><h4>{team2}</h4><h1>{t2_p}%</h1>Win Prob</div>", unsafe_allow_html=True)

# (All other pages: Season Dashboard, Fantasy Scout, Match Center, etc. are identical to original)
elif page == "Match Center":
    st.title("Pro Scorecard & Live Analysis")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    ms = ml.apply(lambda x: f"{x['team1']} vs {x['team2']} ({x['date'].strftime('%Y-%m-%d')})", axis=1)
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        mm = ml.iloc[ms.tolist().index(sel)]
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        st.markdown(f"<div class='premium-box' style='border-left:5px solid #38bdf8;'><h2 style='margin:0;'>{mm['team1']} vs {mm['team2']}</h2>🏟️ {mm['venue']}</div>", unsafe_allow_html=True)
        sc1, sc2 = st.tabs(["1st Innings", "2nd Innings"])
        with sc1:
            bt, bl = get_inning_scorecard(mb, 1)
            if bt is not None:
                c1, c2 = st.columns(2)
                c1.dataframe(bt, use_container_width=True, hide_index=True); c2.dataframe(bl, use_container_width=True, hide_index=True)
        with sc2:
            bt, bl = get_inning_scorecard(mb, 2)
            if bt is not None:
                c1, c2 = st.columns(2)
                c1.dataframe(bt, use_container_width=True, hide_index=True); c2.dataframe(bl, use_container_width=True, hide_index=True)

elif page == "Season Dashboard":
    season = st.selectbox("Select Season", sorted(matches_df['season'].unique(), reverse=True))
    s_balls = balls_df[balls_df['season'] == season]
    bat, bowl = get_batting_stats(s_balls), get_bowling_stats(s_balls)
    m1, m2 = st.columns(2)
    if not bat.empty: m1.metric("Orange Cap", bat.iloc[0]['batter'], f"{int(bat.iloc[0]['runs_batter'])} R")
    if not bowl.empty: m2.metric("Purple Cap", bowl.iloc[0]['bowler'], f"{int(bowl.iloc[0]['wickets'])} W")

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
    st.title("Player Analysis")
    p = st.selectbox("Select Player", sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique()))))
    all_bat, all_bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    bp = all_bat[all_bat['batter'] == p]
    if not bp.empty: st.metric("Total Runs", int(bp.iloc[0]['runs_batter']))

elif page == "Player Comparison":
    st.title("Head-to-Head Comparison")
    all_players = sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique())))
    p1, p2 = st.columns(2)[0].selectbox("Player 1", all_players), st.columns(2)[1].selectbox("Player 2", all_players)

elif page == "Venue Analysis":
    v = st.selectbox("Select Venue", sorted(matches_df['venue'].unique()))
    vm = matches_df[matches_df['venue'] == v]
    st.metric("Matches Hosted", int(len(vm)))

elif page == "Umpire Records":
    u = st.selectbox("Select Umpire", sorted(pd.concat([matches_df['umpire1'], matches_df['umpire2']]).unique()))

elif page == "Hall of Fame":
    t1, t2 = st.tabs(["Batting", "Bowling"])
    with t1: st.dataframe(get_batting_stats(balls_df).head(50), use_container_width=True, hide_index=True)
    with t2: st.dataframe(get_bowling_stats(balls_df).head(50), use_container_width=True, hide_index=True)

st.markdown("<div class='disclaimer-box'><strong>Legal Disclaimer:</strong> Independent fan portal. Analytical purposes only.</div>", unsafe_allow_html=True)
