import os
import pandas as pd
from dotenv import load_dotenv
from common import load_playlists, get_track_details
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def recommend(track_id, metadata_features, top_n=5):
    if track_id not in metadata_features.index:
        print(f"Track ID {track_id} not found in metadata features.")
        return []

    seed_track = metadata_features.loc[[track_id]]
    similarities = cosine_similarity(seed_track, metadata_features)[0]
    top_indices = similarities.argsort()[::-1][1 : top_n + 1]
    return metadata_features.iloc[top_indices].index.tolist()


if __name__ == "__main__":
    raw_files = os.getenv("RAW_FILES")
    df, unique_tracks, metadata = load_playlists(raw_files)

    # Load your additional features metadata
    metadata_features_file = os.getenv("METADATA_FILE")
    features_df = pd.read_csv(metadata_features_file).set_index("track_id")

    # Filter features to match available track IDs
    available_features = features_df.index.intersection(metadata["track_id"])
    metadata_features = features_df.loc[available_features]

    # Choose a random track as the seed from available features
    track_id = metadata_features.sample(1).index.values[0]

    # Get recommendations based on audio features
    recommended_track_ids = recommend(track_id, metadata_features)

    # Fetch detailed track names and artists
    recommendations = get_track_details(recommended_track_ids, metadata)
    seed_track_details = get_track_details([track_id], metadata).iloc[0]

    print(
        f"Recommendations similar to '{seed_track_details['track_name']}' by {seed_track_details['artist_name']}:"
    )
    for _, row in recommendations.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']} ({row['track_id']})")

