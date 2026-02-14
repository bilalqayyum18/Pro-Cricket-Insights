import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client

# --- SUPABASE CONNECTION ---
# These must be set in Streamlit Cloud Secrets
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
except Exception:
    supabase = None

# --- CONFIG & THEME ---
st.set_page_config(page_title="Pro Cricket Insights", layout="wide", page_icon="🏏")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    [data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; }
    
    .rank-label {
        color: #39FF14; font-size: 0.9rem; font-weight: 700;
        text-transform: uppercase; margin-top: -10px; display: block; margin-bottom: 15px;
    }

    .pro-insight-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38bdf8; border-radius: 10px; padding: 20px; margin: 10px 0;
    }

    .prediction-card {
        background: #1e293b; border-radius: 15px; padding: 25px;
        border-top: 5px solid #38bdf8; text-align: center;
    }

    .match-header-box {
        background: #1e293b; border-radius: 15px; padding: 20px;
        border-left: 5px solid #38bdf8; margin-bottom: 25px;
    }

    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #0f172a; color: #64748b; text-align: center;
        padding: 10px; font-size: 11px; border-top: 1px solid #1e283b; z-index: 1000;
    }
    .scorecard-header { background-color: #38bdf8; color: #0f172a; padding: 10px; border-radius: 5px; font-weight: bold; margin-top: 20px;}
    [data-testid="stSidebar"] { background-color: #1e293b !important; }
    
    .paywall-box {
        background: #ef444422; border: 1px solid #ef4444;
        padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;
    }
    </style>
    <div class="footer">
        <b>Fan-led Data Project:</b> Not affiliated with PSL. PRO CRICKET INSIGHTS © 2026
    </div>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    matches = pd.read_csv("psl_matches_meta_clean.csv")
    balls = pd.read_csv("psl_ball_by_ball_clean.csv")
    matches['venue'] = matches['venue'].str.split(',').str[0]
    return matches, balls

matches_df, balls_df = load_data()

# --- ML MODEL ENGINE ---
@st.cache_resource
def train_ml_model(df):
    model_df = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner']].dropna()
    
    def get_h2h_win_rate(t1, t2):
        relevant = df[((df['team1'] == t1) & (df['team2'] == t2)) | ((df['team1'] == t2) & (df['team2'] == t1))]
        return len(relevant[relevant['winner'] == t1]) / len(relevant) if len(relevant) > 0 else 0.5

    def get_venue_win_rate(team, venue):
        relevant = df[(df['venue'] == venue) & ((df['team1'] == team) | (df['team2'] == team))]
        return len(relevant[relevant['winner'] == team]) / len(relevant) if len(relevant) > 0 else 0.5

    model_df['h2h'] = model_df.apply(lambda x: get_h2h_win_rate(x['team1'], x['team2']), axis=1)
    model_df['v_t1'] = model_df.apply(lambda x: get_venue_win_rate(x['team1'], x['venue']), axis=1)
    model_df['v_t2'] = model_df.apply(lambda x: get_venue_win_rate(x['team2'], x['venue']), axis=1)

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
        'v_t2': model_df['v_t2']
    })
    
    y = (model_df['winner'] == model_df['team1']).astype(int)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
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
st.sidebar.title("🏏 PRO INSIGHTS")
page = st.sidebar.radio("Navigation", ["🏆 Season Dashboard", "🔮 Fantasy Scout", "🏏 Match Center", "⚡ Impact Players", "⚔️ Player Comparison", "🏟️ Venue Analysis", "⚖️ Umpire Records", "⭐ Hall of Fame", "🤖 Pro Prediction"])

# --- PAGE: MATCH CENTER ---
if page == "🏏 Match Center":
    st.title("Pro Scorecard & Live Analysis")
    s = st.selectbox("Season", sorted(matches_df['season'].unique(), reverse=True))
    ml = matches_df[matches_df['season'] == s]
    ms = ml.apply(lambda x: f"{x['team1']} vs {x['team2']} ({x['date']})", axis=1)
    
    if not ms.empty:
        sel = st.selectbox("Pick Match", ms)
        mm = ml.iloc[ms.tolist().index(sel)]
        mb = balls_df[balls_df['match_id'] == mm['match_id']]
        
        st.markdown(f"""
        <div class="match-header-box">
            <h2 style='margin:0;'>{mm['team1']} vs {mm['team2']}</h2>
            <p style='color:#94a3b8; font-size:1.1rem; margin:5px 0;'>🏟️ {mm['venue']} | 📅 {mm['date']}</p>
            <hr style='border-color:#334155;'>
            <p><b>Toss:</b> {mm['toss_winner']} won and chose to {mm['toss_decision']}</p>
            <p style='font-size:1.2rem; color:#38bdf8;'><b>Result:</b> {mm['winner']} won by {mm['win_margin']} {mm['win_by']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📝 Scorecard Summary")
        sc1, sc2 = st.tabs([f"1st Innings: {mm['team1'] if mm['toss_decision']=='bat' and mm['toss_winner']==mm['team1'] or mm['toss_decision']=='field' and mm['toss_winner']==mm['team2'] else mm['team2']}", 
                            f"2nd Innings: {mm['team2'] if mm['toss_decision']=='bat' and mm['toss_winner']==mm['team1'] or mm['toss_decision']=='field' and mm['toss_winner']==mm['team2'] else mm['team1']}"])
        
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
        fig_worm = px.line(worm, x='over', y='runs_total', color='innings', title="📈 Match Worm", template="plotly_dark")
        for i, team in enumerate([mm['team1'], mm['team2']]):
            idat = worm[worm['innings'] == (i+1)]
            if not idat.empty:
                last = idat.iloc[-1]
                fig_worm.add_annotation(x=last['over'], y=last['runs_total'], text=team, showarrow=False, xanchor="left", xshift=10)
        st.plotly_chart(fig_worm, use_container_width=True)

        over_runs = mb.groupby(['innings', 'over'])['runs_total'].sum().reset_index()
        over_runs['momentum'] = over_runs.groupby('innings')['runs_total'].diff().fillna(over_runs['runs_total'])
        fig_mom = px.area(over_runs, x='over', y='momentum', color='innings', title="⚡ Match Momentum (Runs per Over)", template="plotly_dark")
        st.plotly_chart(fig_mom, use_container_width=True)

        st.markdown("### ⚔️ Pro Match-Ups")
        match_batters = sorted(mb['batter'].unique())
        match_bowlers = sorted(mb['bowler'].unique())
        
        col_b, col_w = st.columns(2)
        sel_bat = col_b.selectbox("Select Batter from Match", match_batters)
        sel_bow = col_w.selectbox("Select Bowler from Match", match_bowlers)
        
        h2h = balls_df[(balls_df['batter'] == sel_bat) & (balls_df['bowler'] == sel_bow)]
        if not h2h.empty:
            h_runs, h_balls, h_outs = h2h['runs_batter'].sum(), h2h['ball'].count() - h2h['wide'].sum(), h2h['is_wicket'].sum()
            h_sr = round((h_runs/h_balls*100), 1) if h_balls > 0 else 0
            st.markdown(f"<div class='pro-insight-box'><b>{sel_bat} vs {sel_bow} (Historical)</b><br>Runs: {h_runs} | Balls: {h_balls} | SR: {h_sr} | Outs: <span style='color:#ff4b4b;'>{h_outs}</span></div>", unsafe_allow_html=True)
        else:
            st.info("No historical head-to-head data for this pair.")

# --- PAGE: PRO PREDICTION ---
elif page == "🤖 Pro Prediction":
    st.title("🔮 AI Match Predictor")
    
    # --- USAGE TRACKING LOGIC ---
    can_predict = True
    usage_left = 3
    user_id = "guest_user" # Tracking by unique guest ID

    if supabase:
        try:
            # Check existing usage
            res = supabase.table("prediction_logs").select("*").eq("user_ip", user_id).execute()
            if res.data:
                count = res.data[0]['usage_count']
                usage_left = max(0, 3 - count)
                if count >= 3:
                    can_predict = False
            else:
                # Initialize new user tracking
                supabase.table("prediction_logs").insert({"user_ip": user_id, "usage_count": 0}).execute()
        except Exception:
            pass

    if not can_predict:
        st.markdown("""
            <div class="paywall-box">
                <h3 style='color:#f1f5f9;'>Daily Free Limit Reached</h3>
                <p style='color:#94a3b8;'>Unlock unlimited AI predictions and advanced analytics.</p>
                <a href='#' style='background:#38bdf8; color:#0f172a; padding:10px 20px; border-radius:5px; text-decoration:none; font-weight:bold;'>UPGRADE TO PRO</a>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"You have {usage_left} free simulations remaining today.")
        st.markdown("Run a high-fidelity simulation using historical Head-to-Head, Venue dynamics, and Recent Team Form.")
        
        model, le_t, le_v, le_d = train_ml_model(matches_df)
        
        with st.container():
            c1, c2, c3 = st.columns(3)
            season = c1.selectbox("Target Season", sorted(matches_df['season'].unique(), reverse=True))
            venue = c2.selectbox("Venue / Stadium", sorted(matches_df['venue'].unique()))
            toss_decision = c3.selectbox("Toss Decision", sorted(matches_df['toss_decision'].unique()))
            
            t1_col, vs_col, t2_col = st.columns([4, 1, 4])
            team1 = t1_col.selectbox("Team 1 (Home/Bat First)", sorted(matches_df['team1'].unique()))
            vs_col.markdown("<h2 style='text-align: center; margin-top: 30px;'>VS</h2>", unsafe_allow_html=True)
            team2 = t2_col.selectbox("Team 2 (Away/Chase)", [t for t in sorted(matches_df['team2'].unique()) if t != team1])
            
            toss_winner = st.selectbox("Toss Winner", [team1, team2])

        if st.button("🚀 RUN ML SIMULATION", use_container_width=True):
            # Update usage in Supabase
            if supabase:
                try:
                    curr = supabase.table("prediction_logs").select("usage_count").eq("user_ip", user_id).execute()
                    new_count = curr.data[0]['usage_count'] + 1
                    supabase.table("prediction_logs").update({"usage_count": new_count}).eq("user_ip", user_id).execute()
                except Exception:
                    pass

            with st.spinner("Analyzing match dynamics..."):
                time.sleep(1.2)
                
                h2h_matches = matches_df[((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) | ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))]
                h2h_val = len(h2h_matches[h2h_matches['winner'] == team1]) / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
                
                v_t1_m = matches_df[(matches_df['venue'] == venue) & ((matches_df['team1'] == team1) | (matches_df['team2'] == team1))]
                v_t1_val = len(v_t1_m[v_t1_m['winner'] == team1]) / len(v_t1_m) if len(v_t1_m) > 0 else 0.5
                
                v_t2_m = matches_df[(matches_df['venue'] == venue) & ((matches_df['team1'] == team2) | (matches_df['team2'] == team2))]
                v_t2_val = len(v_t2_m[v_t2_m['winner'] == team2]) / len(v_t2_m) if len(v_t2_m) > 0 else 0.5

                input_data = pd.DataFrame({
                    'team1': le_t.transform([team1]),
                    'team2': le_t.transform([team2]),
                    'venue': le_v.transform([venue]),
                    'toss_winner': le_t.transform([toss_winner]),
                    'toss_decision': le_d.transform([toss_decision]),
                    'h2h': [h2h_val],
                    'v_t1': [v_t1_val],
                    'v_t2': [v_t2_val]
                })
                
                probs = model.predict_proba(input_data)[0]
                t1_prob = round(probs[1] * 100, 1)
                t2_prob = 100 - t1_prob
                
                st.markdown("### 📊 Simulation Results")
                res1, res2 = st.columns(2)
                res1.markdown(f"<div class='prediction-card'><h4>{team1}</h4><h1>{t1_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)
                res2.markdown(f"<div class='prediction-card'><h4>{team2}</h4><h1>{t2_prob}%</h1>Win Probability</div>", unsafe_allow_html=True)
                
                st.info(f"**ML Insight:** Factors like H2H dominance ({round(h2h_val*100)}%) and Venue proficiency at {venue} were prioritized in this simulation.")

elif page == "🏆 Season Dashboard":
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
    fig_bat = px.bar(bat.head(10), x='batter', y='runs_batter', text='runs_batter', title="Top 10 Batters", template="plotly_dark")
    st.plotly_chart(fig_bat, use_container_width=True)

elif page == "🔮 Fantasy Scout":
    st.title("Fantasy Team Optimizer")
    season_f = st.selectbox("Data Context", sorted(matches_df['season'].unique(), reverse=True))
    sf_balls = balls_df[balls_df['season'] == season_f]
    b, w = get_batting_stats(sf_balls), get_bowling_stats(sf_balls)
    fan = b.merge(w, left_on='batter', right_on='bowler', how='outer').fillna(0)
    fan['p_name'] = fan['batter'].where(fan['batter']!=0, fan['bowler'])
    fan['pts'] = (fan['runs_batter']*1) + (fan['wickets']*25)
    top_11 = fan.sort_values('pts', ascending=False).head(11)
    fig = px.bar(top_11, x='pts', y='p_name', orientation='h', title="My Dream XI", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Impact Players":
    st.title("Player Analysis & Rankings")
    p = st.selectbox("Select Player", sorted(list(set(balls_df['batter'].unique()) | set(balls_df['bowler'].unique()))))
    all_bat, all_bowl = get_batting_stats(balls_df), get_bowling_stats(balls_df)
    all_bat['run_rank'] = all_bat['runs_batter'].rank(ascending=False, method='min').astype(int)
    all_bat['sr_rank'] = all_bat['strike_rate'].rank(ascending=False, method='min').astype(int)
    all_bowl['wick_rank'] = all_bowl['wickets'].rank(ascending=False, method='min').astype(int)
    all_bowl['econ_rank'] = all_bowl['economy'].rank(ascending=True, method='min').astype(int)
    ca, cb = st.columns(2)
    bp = all_bat[all_bat['batter'] == p]
    if not bp.empty:
        with ca:
            st.metric("Total Runs", int(bp.iloc[0]['runs_batter']))
            st.markdown(f"<span class='rank-label'>Rank #{bp.iloc[0]['run_rank']}</span>", unsafe_allow_html=True)
            st.metric("Strike Rate", bp.iloc[0]['strike_rate'])
            st.markdown(f"<span class='rank-label'>Rank #{bp.iloc[0]['sr_rank']}</span>", unsafe_allow_html=True)
    wp = all_bowl[all_bowl['bowler'] == p]
    if not wp.empty:
        with cb:
            st.metric("Total Wickets", int(wp.iloc[0]['wickets']))
            st.markdown(f"<span class='rank-label'>Rank #{wp.iloc[0]['wick_rank']}</span>", unsafe_allow_html=True)
            st.metric("Economy", wp.iloc[0]['economy'])
            st.markdown(f"<span class='rank-label'>Rank #{wp.iloc[0]['econ_rank']}</span>", unsafe_allow_html=True)

elif page == "⚔️ Player Comparison":
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

elif page == "🏟️ Venue Analysis":
    st.title("Venue Intelligence")
    v = st.selectbox("Select Venue", sorted(matches_df['venue'].unique()))
    vm = matches_df[matches_df['venue'] == v]
    st.metric("Matches Hosted", int(len(vm)))
    st.metric("Defend Wins", int(len(vm[vm['win_by'] == 'runs'])))

elif page == "⚖️ Umpire Records":
    st.title("Umpire Records")
    u = st.selectbox("Select Umpire", sorted(pd.concat([matches_df['umpire1'], matches_df['umpire2']]).unique()))
    um = matches_df[(matches_df['umpire1'] == u) | (matches_df['umpire2'] == u)]
    st.plotly_chart(px.bar(um['winner'].value_counts().reset_index(), x='winner', y='count', template="plotly_dark"), use_container_width=True)

elif page == "⭐ Hall of Fame":
    st.title("All-Time Records")
    t1, t2 = st.tabs(["Batting", "Bowling"])
    with t1: st.dataframe(get_batting_stats(balls_df).head(50), use_container_width=True)
    with t2: st.dataframe(get_bowling_stats(balls_df).head(50), use_container_width=True)

# --- GLOBAL LEGAL DISCLAIMER ---
st.markdown("""
    <div class="disclaimer-box" style="background-color: #0f172a; border: 1px solid #1e293b; border-left: 4px solid #ef4444; padding: 24px; border-radius: 6px; margin-top: 40px; margin-bottom: 60px; color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
        <strong>Legal Disclaimer & Terms of Use:</strong><br>
        This platform is an <strong>independent fan-led project</strong> and is not affiliated with, endorsed by, or 
        associated with the Pakistan Super League (PSL), the Pakistan Cricket Board (PCB), or any specific cricket 
        franchise. All trademarks and copyrights belong to their respective owners.<br><br>
        This tool utilizes Machine Learning (ML) to generate probabilistic outcomes based on historical data. 
        These predictions are for <strong>informational and entertainment purposes only</strong> and do not provide 
        guaranteed results. <strong>The use of this tool for illegal gambling or match manipulation is strictly prohibited.</strong>
    </div>
""", unsafe_allow_html=True)
