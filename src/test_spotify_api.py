import os
import base64
import requests
import time
import pandas as pd
import json
from dotenv import load_dotenv
from datetime import datetime, timezone

# =========================
# LOAD ENV + AUTH
# =========================
load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Spotify credentials not found in .env")

auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
auth_base64 = base64.b64encode(auth_string.encode()).decode()

token_url = "https://accounts.spotify.com/api/token"

token_response = requests.post(
    token_url,
    headers={"Authorization": f"Basic {auth_base64}"},
    data={"grant_type": "client_credentials"}
)

token_response.raise_for_status()
access_token = token_response.json()["access_token"]

search_headers = {
    "Authorization": f"Bearer {access_token}"
}

print("✅ Access token fetched")

# =========================
# CONFIG
# =========================
SEARCH_URL = "https://api.spotify.com/v1/search"
AUDIO_FEATURES_URL = "https://api.spotify.com/v1/audio-features"

LIMIT = 50
TOTAL_TRACKS = 150
MARKETS = ["US", "IN", "GB"]

data_fetched_at = datetime.now(timezone.utc)

# =========================
# FETCH TRACK DATA
# =========================
all_tracks = []
raw_search_responses = []

for market in MARKETS:
    print(f"\n🎧 Fetching market: {market}")

    for offset in range(0, TOTAL_TRACKS, LIMIT):
        params = {
            "q": "year:2024",
            "type": "track",
            "limit": LIMIT,
            "offset": offset,
            "market": market
        }

        response = requests.get(SEARCH_URL, headers=search_headers, params=params)
        response.raise_for_status()
        data = response.json()

        raw_search_responses.append(data)

        for item in data.get("tracks", {}).get("items", []):
            all_tracks.append({
                "track_id": item["id"],
                "track_name": item["name"],
                "artist_name": item["artists"][0]["name"],
                "album_name": item["album"]["name"],
                "release_date": item["album"]["release_date"],
                "popularity": item["popularity"],
                "duration_ms": item["duration_ms"],
                "market": market,
                "data_fetched_at": data_fetched_at
            })

        print(f"Fetched {len(all_tracks)} tracks so far")

# =========================
# SAVE RAW SEARCH DATA
# =========================
os.makedirs("data/raw", exist_ok=True)

raw_path = f"data/raw/spotify_search_raw_{data_fetched_at.isoformat()}.json"
with open(raw_path, "w") as f:
    json.dump(raw_search_responses, f, indent=2)

print(f"🗃 Raw data saved → {raw_path}")

# =========================
# AUDIO FEATURES
# =========================
track_ids = list({t["track_id"] for t in all_tracks})
print(f"🎼 Unique tracks for audio features: {len(track_ids)}")

audio_rows = []

BATCH_SIZE = 50  # smaller batch = fewer 403s

for i in range(0, len(track_ids), BATCH_SIZE):
    batch = track_ids[i:i + BATCH_SIZE]
    ids_param = ",".join(batch)

    try:
        response = requests.get(
            AUDIO_FEATURES_URL,
            headers=search_headers,
            params={"ids": ids_param}
        )

        if response.status_code == 403:
            print("⚠️ Hit rate limit, sleeping 5s...")
            time.sleep(5)
            continue

        response.raise_for_status()
        features = response.json().get("audio_features", [])

        for f in features:
            if f is None:
                continue

            audio_rows.append({
                "track_id": f["id"],
                "danceability": f["danceability"],
                "energy": f["energy"],
                "valence": f["valence"],
                "tempo": f["tempo"],
                "speechiness": f["speechiness"],
                "acousticness": f["acousticness"],
                "instrumentalness": f["instrumentalness"],
                "liveness": f["liveness"]
            })

        time.sleep(0.2)  # polite delay

    except Exception as e:
        print(f"❌ Skipping batch due to error: {e}")
        time.sleep(5)


# Build audio features DataFrame safely
if len(audio_rows) == 0:
    print("⚠️ No audio features fetched — continuing with empty columns")
    audio_df = pd.DataFrame(columns=[
        "track_id",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness"
    ])
else:
    audio_df = pd.DataFrame(audio_rows)


# =========================
# BUILD DATAFRAME
# =========================
df_tracks = pd.DataFrame(all_tracks)

# Clean column names
df_tracks.columns = [c.lower() for c in df_tracks.columns]
audio_df.columns = [c.lower() for c in audio_df.columns]

# Merge audio features
df_tracks = df_tracks.merge(audio_df, on="track_id", how="left")

print("\n📊 Columns after merge:")
print(df_tracks.columns.tolist())

# =========================
# PHASE 2 — CLEANING
# =========================
df_tracks["release_date"] = pd.to_datetime(df_tracks["release_date"], errors="coerce")

df_tracks["days_since_release"] = (
    df_tracks["data_fetched_at"].dt.tz_localize(None)
    - df_tracks["release_date"]
).dt.days

def popularity_bucket(p):
    if p < 50:
        return "Low"
    elif p < 75:
        return "Medium"
    return "High"

df_tracks["popularity_bucket"] = df_tracks["popularity"].apply(popularity_bucket)

# =========================
# DEDUP
# =========================
before = len(df_tracks)
df_tracks = df_tracks.drop_duplicates(
    subset=["track_id", "market", "data_fetched_at"]
)
after = len(df_tracks)

print(f"🧹 Removed {before - after} duplicate rows")

# =========================
# PHASE 3 — VIBE SCORE
# =========================
df_tracks["vibe_score"] = (
    df_tracks["danceability"]
    + df_tracks["energy"]
    + df_tracks["valence"]
) / 3

print("\n🎵 Vibe score preview:")
print(df_tracks[[
    "track_name", "danceability", "energy", "valence", "vibe_score"
]].head())

# =========================
# SAVE FINAL DATASET
# =========================
os.makedirs("data/processed", exist_ok=True)

output_path = "data/spotify_cleaned.csv"
df_tracks.to_csv(output_path, index=False)

print(f"\n✅ Final dataset saved → {output_path}")
