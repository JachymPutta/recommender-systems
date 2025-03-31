import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from common import load_playlists, get_track_details

load_dotenv()


class RankingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def prepare_features(metadata_features):
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(metadata_features), index=metadata_features.index
    )


def retrieve_candidates(track_ids, feature_tensor, track_to_idx, top_k=20):
    indices = [track_to_idx[tid] for tid in track_ids if tid in track_to_idx]
    playlist_embedding = feature_tensor[indices].mean(axis=0, keepdims=True)
    similarities = cosine_similarity(playlist_embedding, feature_tensor)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [list(track_to_idx.keys())[i] for i in top_indices]


def rank_candidates(candidate_ids, feature_tensor, track_to_idx, model):
    indices = [track_to_idx[tid] for tid in candidate_ids]
    candidate_features = feature_tensor[indices]
    model.eval()
    with torch.no_grad():
        scores = model(candidate_features).numpy()
    ranked_indices = np.argsort(scores)[::-1]
    return [candidate_ids[i] for i in ranked_indices[:5]]


if __name__ == "__main__":
    raw_files = os.getenv("RAW_FILES")
    metadata_file = os.getenv("METADATA_FILE")

    df, unique_tracks, metadata = load_playlists(raw_files)
    features_df = pd.read_csv(metadata_file).set_index("track_id").dropna()

    common_tracks = features_df.index.intersection(metadata["track_id"])
    metadata_features = features_df.loc[common_tracks]
    prepared_features = prepare_features(metadata_features)
    feature_tensor = torch.tensor(prepared_features.values, dtype=torch.float32)

    track_to_idx = {tid: idx for idx, tid in enumerate(prepared_features.index)}

    # Simple ranking model training (illustrative)
    ranking_model = RankingModel(input_dim=feature_tensor.shape[1])
    optimizer = torch.optim.Adam(ranking_model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    ranking_model.train()
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        for _, row in df.sample(50).iterrows():
            playlist_tracks = [
                t["track_uri"].split("spotify:track:")[1]
                for t in row["tracks"]
                if t["track_uri"].split("spotify:track:")[1] in track_to_idx
            ]
            if len(playlist_tracks) < 2:
                continue
            positive_idxs = [track_to_idx[tid] for tid in playlist_tracks]
            negative_idxs = np.random.choice(
                len(feature_tensor), len(positive_idxs), replace=False
            )

            pos_feats = feature_tensor[positive_idxs]
            neg_feats = feature_tensor[negative_idxs]

            inputs = torch.cat([pos_feats, neg_feats])
            labels = torch.cat(
                [torch.ones(len(pos_feats)), torch.zeros(len(neg_feats))]
            )

            preds = ranking_model(inputs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Pick random playlist for recommendation
    sample_playlist = df.sample(1).iloc[0]
    playlist_tracks = [
        t["track_uri"].split("spotify:track:")[1] for t in sample_playlist["tracks"]
    ]

    # Retrieval
    candidate_track_ids = retrieve_candidates(
        playlist_tracks, prepared_features.values, track_to_idx
    )

    # Ranking
    recommended_track_ids = rank_candidates(
        candidate_track_ids, feature_tensor, track_to_idx, ranking_model
    )

    original_tracks = get_track_details(playlist_tracks[:5], metadata)
    recommendations = get_track_details(recommended_track_ids, metadata)

    print(f"\nSample tracks from playlist '{sample_playlist['name']}':")
    for _, row in original_tracks.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']}")

    print("\nRecommended tracks after ranking:")
    for _, row in recommendations.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']}")
