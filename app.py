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
        supabase = create_client(url, key)
        supabase_status = "Connected"
    else:
        supabase_status = "Missing Credentials"
except Exception as e:
    supabase_status = f"Error: {str(e)}"

st.set_page_config(page_title="Pro Cricket Insights", layout="wide", page_icon="🏏")

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data():
    if supabase_status != "Connected":
        st.error(f"Supabase Connection Failed: {supabase_status}")
        st.stop()

    try:
        matches_res = supabase.table("matches").select("*").execute()
        matches = pd.DataFrame(matches_res.data)

        if matches.empty:
            st.error("Matches table is empty.")
            st.stop()

        # Batched fetch for ball_by_ball
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

        # Clean matches
        matches['match_id'] = pd.to_numeric(matches['match_id'], errors='coerce')
        matches['season'] = pd.to_numeric(matches['season'], errors='coerce')
        matches = matches.dropna(subset=['season'])
        matches['season'] = matches['season'].astype(int)
        matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
        matches['venue'] = matches['venue'].astype(str).str.split(',').str[0]

        # Clean balls
        balls['match_id'] = pd.to_numeric(balls['match_id'], errors='coerce')
        balls['innings'] = pd.to_numeric(balls['innings'], errors='coerce')
        balls['over'] = pd.to_numeric(balls['over'], errors='coerce')
        balls['ball'] = pd.to_numeric(balls['ball'], errors='coerce')

        numeric_cols = [
            'runs_batter', 'runs_extras', 'runs_total',
            'wide', 'noball', 'bye', 'legbye', 'is_wicket'
        ]

        for col in numeric_cols:
            if col in balls.columns:
                balls[col] = pd.to_numeric(balls[col], errors='coerce').fillna(0)

        # Map venue into balls
        venue_map = matches.set_index('match_id')['venue'].to_dict()
        balls['venue'] = balls['match_id'].map(venue_map)

        return matches, balls

    except Exception as e:
        st.error(f"Supabase fetch failed: {e}")
        st.stop()

matches_df, balls_df = load_data()

# --- ANALYTICS FUNCTIONS ---
def get_batting_stats(df):
    if df.empty:
        return pd.DataFrame()

    bat = df.groupby('batter').agg({
        'runs_batter': 'sum',
        'ball': 'count',
        'wide': 'sum',
        'match_id': 'nunique',
        'is_wicket': 'sum'
    }).reset_index()

    bat['balls_faced'] = bat['ball'] - bat['wide']
    bat['strike_rate'] = (bat['runs_batter'] / bat['balls_faced'].replace(0, 1) * 100).round(2)

    f = df[df['runs_batter'] == 4].groupby('batter').size().reset_index(name='4s')
    s = df[df['runs_batter'] == 6].groupby('batter').size().reset_index(name='6s')

    res = bat.merge(f, on='batter', how='left').merge(s, on='batter', how='left').fillna(0)

    for col in ['runs_batter', 'balls_faced', '4s', '6s', 'match_id']:
        res[col] = res[col].astype(int)

    return res.sort_values('runs_batter', ascending=False)


def get_bowling_stats(df):
    if df.empty:
        return pd.DataFrame()

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

# --- NAVIGATION ---
st.sidebar.title("Pakistan League Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["Season Dashboard", "Fantasy Scout"]
)

# --- SEASON DASHBOARD ---
if page == "Season Dashboard":
    season = st.selectbox("Select Season", sorted(matches_df['season'].unique(), reverse=True))
    st.title(f"Tournament Summary: {season}")

    s_matches = matches_df[matches_df['season'] == season]
    s_balls = balls_df[balls_df['match_id'].isin(s_matches['match_id'])]

    if not s_balls.empty:
        bat = get_batting_stats(s_balls)
        bowl = get_bowling_stats(s_balls)

        m1, m2 = st.columns(2)
        if not bat.empty:
            m1.metric("Orange Cap", bat.iloc[0]['batter'], f"{bat.iloc[0]['runs_batter']} Runs")
        if not bowl.empty:
            m2.metric("Purple Cap", bowl.iloc[0]['bowler'], f"{bowl.iloc[0]['wickets']} Wickets")

        st.subheader("Top Performers")

        if not bat.empty:
            st.plotly_chart(
                px.bar(bat.head(10), x='batter', y='runs_batter', template="plotly_dark"),
                use_container_width=True
            )

        if not bowl.empty:
            st.plotly_chart(
                px.bar(bowl.head(10), x='bowler', y='wickets', template="plotly_dark"),
                use_container_width=True
            )
    else:
        st.warning(f"No match data found for season {season}")

# --- FANTASY SCOUT ---
elif page == "Fantasy Scout":
    st.title("Fantasy Team Optimizer")

    season_f = st.selectbox("Data Context", sorted(matches_df['season'].unique(), reverse=True))
    sf_matches = matches_df[matches_df['season'] == season_f]
    sf_balls = balls_df[balls_df['match_id'].isin(sf_matches['match_id'])]

    if not sf_balls.empty:
        b = get_batting_stats(sf_balls)
        w = get_bowling_stats(sf_balls)

        fan = b.merge(w, left_on='batter', right_on='bowler', how='outer').fillna(0)
        fan['p_name'] = fan['batter'].where(fan['batter'] != 0, fan['bowler'])
        fan['pts'] = fan['runs_batter'] + fan['wickets'] * 25

        st.plotly_chart(
            px.bar(
                fan.sort_values('pts', ascending=False).head(11),
                x='pts', y='p_name',
                orientation='h',
                template="plotly_dark"
            ),
            use_container_width=True
        )
