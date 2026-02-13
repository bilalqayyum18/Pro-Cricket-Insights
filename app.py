import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client

# --- SUPABASE CONNECTION ---
# These will be set in Streamlit Cloud Secrets later
try:
    url = st.secrets["https://esbctbsundqyvdjwawxa.supabase.co"]
    key = st.secrets["sb_publishable_ONUxA1TPV__q68SNLxt4TQ_B7Nk_Xz6"]
    supabase: Client = create_client(url, key)
except:
    supabase = None

# --- CONFIG & THEME ---
st.set_page_config(page_title="Pro Cricket Insights", layout="wide", page_icon="🏏")

# Premium UI Polish (CSS Only)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0f172a; color: #f1f5f9; }
    
    /* Premium Metric Styling */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.9rem !important; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; }

    /* Prediction Card */
    .prediction-card {
        background: #1e293b; 
        border-radius: 12px; 
        padding: 30px;
        border: 1px solid #334155; 
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .prediction-card h4 { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 8px; }
    .prediction-card h1 { color: #38bdf8; margin: 0; font-size: 2.5rem; }

    /* Content Containers */
    .premium-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }

    /* Professional Disclaimer - Global Style */
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

    /* Fixed Footer */
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #0f172a; color: #475569; text-align: center;
        padding: 10px; font-size: 11px; border-top: 1px solid #1e293b; z-index: 1000;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
    .stRadio > label { font-weight: 600 !important; color: #f1f5f9 !important; }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }

    /* Button Styling */
    .stButton>button {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: 800 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
    }

    .paywall-box {
        background: #ef444422;
        border: 1px solid #ef4444;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
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
        'h2h': model_df['h2h'],
        'v_t1': model_df['v_t1'],
        'v_t2': model_df['v_t2'],
        'form_t1': model_df['form_t1'],
        'form_t2': model_df['form_t2']
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

# --- NAVIGATION ---
st.sidebar.title("Cricket Intelligence")
page = st.sidebar.radio("Navigation", ["Season Dashboard", "Fantasy Scout", "Match Center", "Impact Players", "Player Comparison", "Venue Analysis", "Umpire Records", "Hall of Fame", "Pro Prediction"])

# --- PAGE LOGIC ---
if page == "Match Center":
    st.title("Pro Scorecard & Live Analysis")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    ms = ml.apply(lambda x: f"{x['team1']} vs {x['team2']} ({x['date'].strftime('%Y-%m-%d')})", axis=1)
    
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        mm = ml.iloc[ms.tolist().index(sel)]
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

        worm = mb.groupby(['innings', 'over'])['runs_total'].sum().groupby(level=0).cumsum().reset_index()
        fig_worm = px.line(worm, x='over', y='runs_total', color='innings', title="Match Worm", template="plotly_dark")
        st.plotly_chart(fig_worm, use_container_width=True)

elif page == "Pro Prediction":
    st.title("AI Match Predictor")
    
    # Track usage logic
    can_predict = True
    usage_left = 3
    
    if supabase:
        # Simple IP tracking for non-logged in users to prevent abuse
        try:
            # We use a placeholder for local testing, but this works on Streamlit Cloud
            dummy_ip = "user_guest" 
            res = supabase.table("prediction_logs").select("*").eq("user_ip", dummy_ip).execute()
            if res.data:
                count = res.data[0]['usage_count']
                usage_left = max(0, 3 - count)
                if count >= 3:
                    can_predict = False
            else:
                supabase.table("prediction_logs").insert({"user_ip": dummy_ip, "usage_count": 0}).execute()
        except:
            pass

    if not can_predict:
        st.markdown("""
            <div class="paywall-box">
                <h3 style='color:#f1f5f9;'>Daily Free Limit Reached</h3>
                <p style='color:#94a3b8;'>Unlock unlimited AI predictions and detailed player impact reports.</p>
                <a href='#' style='background:#38bdf8; color:#0f172a; padding:10px 20px; border-radius:5px; text-decoration:none; font-weight:bold;'>UPGRADE TO PRO</a>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"You have {usage_left} free predictions remaining today.")
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

        if st.button("RUN PRO SIMULATION", use_container_width=True):
            # Update usage in Supabase
            if supabase:
                try:
                    supabase.rpc('increment_usage', {'u_ip': 'user_guest'}).execute()
                    # Fallback if RPC not set up:
                    curr = supabase.table("prediction_logs").select("usage_count").eq("user_ip", "user_guest").execute()
                    new_count = curr.data[0]['usage_count'] + 1
                    supabase.table("prediction_logs").update({"usage_count": new_count}).eq("user_ip", "user_guest").execute()
                except: pass

            with st.spinner("Analyzing historical variables..."):
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
                t2_prob = 100 - t1_prob
                
                st.markdown("### AI Predicted Probability")
                res1, res2 = st.columns(2)
                res1.markdown(f"<div class='prediction-card'><h4>{team1}</h4><h1>{t1_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)
                res2.markdown(f"<div class='prediction-card'><h4>{team2}</h4><h1>{t2_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)

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

# --- GLOBAL LEGAL DISCLAIMER ---
st.markdown("""
    <div class="disclaimer-box">
        <strong>Legal Disclaimer & Terms of Use:</strong><br>
        This platform is an <strong>independent fan-led project</strong> and is not affiliated with, endorsed by, or 
        associated with the Pakistan Super League (PSL), the Pakistan Cricket Board (PCB), or any specific cricket 
        franchise. All trademarks and copyrights belong to their respective owners.<br><br>
        This tool utilizes Machine Learning (ML) to generate probabilistic outcomes based on historical data. 
        These predictions are for <strong>informational and entertainment purposes only</strong> and do not provide 
        guaranteed results. <strong>The use of this tool for illegal gambling or match manipulation is strictly prohibited.</strong>
    </div>
""", unsafe_allow_html=True)