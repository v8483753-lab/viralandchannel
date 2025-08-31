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
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# Streamlit setup
st.set_page_config(page_title="Find Similar Channels", layout="wide")
st.title("🔍 Find Similar YouTube Channels")

# Cache wrapper
@lru_cache(maxsize=128)
def fetch_json(url: str, params_tuple: tuple):
    params = dict(params_tuple)
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# Fetch videos based on keywords
def get_results(keywords: list, api_key: str, days: int = 7) -> pd.DataFrame:
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

# Suggest similar channels based on description
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

# UI
channel_name = st.text_input("Enter a Channel Name")

if st.button("Find Similar Channels"):
    with st.spinner("Analyzing channel…"):
        # Step 1: Search for channel
        search_params = {
            "part": "snippet", "q": channel_name, "type": "channel",
            "maxResults": 1, "key": API_KEY
        }
        search_data = fetch_json(YOUTUBE_SEARCH_URL, tuple(sorted(search_params.items())))
        if not search_data.get("items"):
            st.error("Channel not found.")
            st.stop()

        ch_id = search_data["items"][0]["snippet"]["channelId"]
        ch_title = search_data["items"][0]["snippet"]["title"]

        # Step 2: Get most popular video
        video_params = {
            "part": "snippet", "channelId": ch_id,
            "order": "viewCount", "maxResults": 1, "type": "video", "key": API_KEY
        }
        video_data = fetch_json(YOUTUBE_SEARCH_URL, tuple(sorted(video_params.items())))
        if not video_data.get("items"):
            st.error("No videos found for this channel.")
            st.stop()

        vid_snip = video_data["items"][0]["snippet"]
        video_title = vid_snip["title"]
        video_desc  = vid_snip.get("description", "")

        # Step 3: Extract keywords
        combined_text = f"{video_title} {video_desc}"
        words = pd.Series(combined_text.lower().split())
        keywords = words[words.str.len() > 4].value_counts().head(5).index.tolist()

        st.markdown(f"**Most Popular Video:** {video_title}")
        st.markdown(f"**Extracted Keywords:** {', '.join(keywords)}")

        # Step 4: Fetch videos using keywords
        df = get_results(keywords, API_KEY)
        df = suggest_similar_channels(df)
        df = df.sort_values(by="Views", ascending=False).head(10)

        st.markdown(f"### 🔗 Channels Similar to **{ch_title}**")
        for _, row in df.iterrows():
            st.markdown(
                f"- **{row['Channel']}**  \n"
                f"  • Subscribers: {row['Subscribers']:,}  \n"
                f"  • [Visit Channel 👤]({row['ChannelURL']})"
            )

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x="Views",
                y="Channel",
                color="Keyword"
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(chart)
