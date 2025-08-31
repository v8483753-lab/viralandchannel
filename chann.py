import streamlit as st
import requests
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# YouTube API endpoints
YOUTUBE_SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL    = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL  = "https://www.googleapis.com/youtube/v3/channels"

# Your API Key
API_KEY = "AIzaSyAeMNLtJxQwsIlk8Z99TyrC9Xvo6DRDbf8"

# Streamlit setup
st.set_page_config(page_title="Viral Topics Dashboard", layout="wide")
st.title("📈 YouTube Viral Topics Dashboard")

# Sidebar: filters & keywords
st.sidebar.header("Search & Filter Settings")
days      = st.sidebar.slider("Search Past Days", 1, 30, value=7)
min_views = st.sidebar.number_input("Min Views", min_value=0, value=1000, step=500)
max_subs  = st.sidebar.number_input("Max Subscribers", min_value=0, value=3000, step=500)
sort_by   = st.sidebar.selectbox("Sort By", ["Views", "Likes", "Comments", "PublishDate", "EngagementRatio", "ViralityScore"])
ascending = st.sidebar.checkbox("Ascending Order", value=False)

st.sidebar.header("Keywords (one per line)")
keywords = st.sidebar.text_area(
    "Enter keywords",
    value="Affair Relationship Stories\nReddit Relationship Advice\nCheating Story Real"
).splitlines()

# Hashable cache wrapper
@lru_cache(maxsize=128)
def fetch_json(url: str, params_tuple: tuple):
    params = dict(params_tuple)
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# Core data-fetching function
def get_results(keywords: list, api_key: str, days: int) -> pd.DataFrame:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat("T") + "Z"
    rows = []

    for kw in keywords:
        search_params = {
            "part": "snippet", "q": kw, "type": "video",
            "order": "viewCount", "publishedAfter": cutoff,
            "maxResults": 10, "key": api_key
        }
        search_data = fetch_json(YOUTUBE_SEARCH_URL, tuple(sorted(search_params.items())))
        for item in search_data.get("items", []):
            vid_id = item["id"].get("videoId")
            if not vid_id:
                continue

            stats = fetch_json(
                YOUTUBE_VIDEO_URL,
                tuple(sorted({"part": "statistics", "id": vid_id, "key": api_key}.items()))
            )
            snip = fetch_json(
                YOUTUBE_VIDEO_URL,
                tuple(sorted({"part": "snippet", "id": vid_id, "key": api_key}.items()))
            )
            ch_id = snip["items"][0]["snippet"]["channelId"]
            ch_data = fetch_json(
                YOUTUBE_CHANNEL_URL,
                tuple(sorted({"part": "statistics,snippet", "id": ch_id, "key": api_key}.items()))
            )

            vi  = snip["items"][0]["snippet"]
            vs  = stats["items"][0]["statistics"]
            ci  = ch_data["items"][0]["snippet"]
            cs  = ch_data["items"][0]["statistics"]

            rows.append({
                "Keyword":     kw,
                "Title":       vi["title"],
                "Channel":     ci["title"],
                "ChannelDesc": ci.get("description", ""),
                "PublishDate": vi["publishedAt"],
                "Views":       int(vs.get("viewCount", 0)),
                "Likes":       int(vs.get("likeCount", 0)),
                "Comments":    int(vs.get("commentCount", 0)),
                "Subscribers": int(cs.get("subscriberCount", 0)),
                "VideoURL":    f"https://youtu.be/{vid_id}",
                "ChannelURL":  f"https://www.youtube.com/channel/{ch_id}"
            })

    return pd.DataFrame(rows)

# Similar channel suggestion logic
def suggest_similar_channels(df: pd.DataFrame) -> pd.DataFrame:
    descs = df["ChannelDesc"].fillna("").replace("", "No description available")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(descs)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    df["SimilarChannels"] = [
        ", ".join(df.iloc[similarity_matrix[i].argsort()[-4:-1]]["Channel"].values)
        for i in range(len(df))
    ]
    return df

# Main App Logic
if st.button("Fetch & Analyze"):
    with st.spinner("Fetching data…"):
        df = get_results(keywords, API_KEY, days)

    if df.empty:
        st.warning("No videos matched your criteria.")
        st.stop()

    # Apply filters
    df = df[df.Views >= min_views]
    df = df[df.Subscribers <= max_subs]

    # Add computed metrics
    df["EngagementRatio"] = (df["Likes"] + df["Comments"]) / df["Views"]
    df["ViralityScore"] = df["Views"] * df["EngagementRatio"]

    # Sort
    df = df.sort_values(by=sort_by, ascending=ascending)

    # Suggest similar channels
    df = suggest_similar_channels(df)

    # Display summary table
    st.dataframe(df.drop(columns=["VideoURL", "ChannelURL", "ChannelDesc"]), height=400)

    # Display clickable links
    st.markdown("### 🔗 Video & Channel Links")
    for _, row in df.iterrows():
        st.markdown(
            f"- **{row['Title']}**  \n"
            f"  • [Watch Video ▶️]({row['VideoURL']})  \n"
            f"  • [Channel 👤]({row['ChannelURL']})"
        )

    # Display similar channels
    st.markdown("### 🤝 Suggested Similar Channels")
    for _, row in df.iterrows():
        st.markdown(f"**{row['Channel']}** → {row['SimilarChannels']}")

    # CSV Export
    export_df = df.copy()
    export_df["Title"]   = export_df.apply(
        lambda r: f'=HYPERLINK("{r.VideoURL}", "{r.Title}")', axis=1
    )
    export_df["Channel"] = export_df.apply(
        lambda r: f'=HYPERLINK("{r.ChannelURL}", "{r.Channel}")', axis=1
    )
    export_df = export_df.drop(columns=["VideoURL", "ChannelURL", "ChannelDesc"])
    csv = export_df.to_csv(index=False)
    st.download_button("Download CSV with Links", csv, "viral_topics_with_links.csv", "text/csv")

    # Bar chart
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(sort_by, sort=alt.SortField(sort_by, order="descending")),
            y=alt.Y("Title", sort="-x"),
            color="Keyword"
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart)
