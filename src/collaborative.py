import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from common import load_playlists, get_track_details

load_dotenv()


def build_user_item_matrix(df, unique_tracks):
    playlist_ids = df["pid"].unique()
    track_to_idx = {track: idx for idx, track in enumerate(unique_tracks)}
    playlist_to_idx = {pid: idx for idx, pid in enumerate(playlist_ids)}

    data, rows, cols = [], [], []

    for _, row in df.iterrows():
        pid_idx = playlist_to_idx[row["pid"]]
        for track in row["tracks"]:
            track_id = track["track_uri"].split("spotify:track:")[1]
            if track_id in track_to_idx:
                track_idx = track_to_idx[track_id]
                rows.append(pid_idx)
                cols.append(track_idx)
                data.append(1)

    matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(playlist_ids), len(unique_tracks))
    )
    return matrix, playlist_to_idx, track_to_idx


def recommend_tracks(user_item_matrix, playlist_idx, idx_to_track, top_n=5):
    similarities = cosine_similarity(
        user_item_matrix[playlist_idx], user_item_matrix
    ).flatten()
    similarities[playlist_idx] = 0  # ignore self-similarity
    similar_playlists_idx = similarities.argsort()[::-1][:10]

    track_scores = np.asarray(
        user_item_matrix[similar_playlists_idx].sum(axis=0)
    ).flatten()
    existing_tracks = set(user_item_matrix[playlist_idx].indices)
    recommended_indices = np.argsort(track_scores)[::-1]

    recommended_tracks = [
        idx_to_track[idx] for idx in recommended_indices if idx not in existing_tracks
    ][:top_n]
    return recommended_tracks


def get_playlist_tracks(df, playlist_id, max_tracks=5):
    playlist = df[df["pid"] == playlist_id].iloc[0]
    tracks = [
        track["track_uri"].split("spotify:track:")[1]
        for track in playlist["tracks"][:max_tracks]
    ]
    return tracks


if __name__ == "__main__":
    raw_files = os.getenv("RAW_FILES")
    df, unique_tracks, metadata = load_playlists(raw_files)

    user_item_matrix, playlist_to_idx, track_to_idx = build_user_item_matrix(
        df, unique_tracks
    )
    idx_to_track = {idx: track for track, idx in track_to_idx.items()}

    sample_playlist_id = np.random.choice(list(playlist_to_idx.keys()))
    sample_playlist_idx = playlist_to_idx[sample_playlist_id]

    # Get original playlist songs
    original_track_ids = get_playlist_tracks(df, sample_playlist_id, max_tracks=5)
    original_tracks = get_track_details(original_track_ids, metadata)

    # Get recommendations
    recommended_track_ids = recommend_tracks(
        user_item_matrix, sample_playlist_idx, idx_to_track
    )
    recommendations = get_track_details(recommended_track_ids, metadata)

    print(f"Sample tracks from playlist {sample_playlist_id}:")
    for _, row in original_tracks.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']} ({row['track_id']})")

    print("\nRecommended tracks:")
    for _, row in recommendations.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']} ({row['track_id']})")
