import os
import torch
import pandas as pd
from torch import nn
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from common import load_playlists, get_track_details

load_dotenv()


class DualTowerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        return self.embedding_layer(x)


def prepare_features(metadata_features):
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(metadata_features), index=metadata_features.index
    )


def embed_playlist(playlist_track_ids, feature_tensor, track_to_idx, model):
    indices = [track_to_idx[tid] for tid in playlist_track_ids if tid in track_to_idx]
    if not indices:
        return None
    track_features = feature_tensor[indices]
    with torch.no_grad():
        playlist_embedding = model(track_features).mean(dim=0, keepdim=True)
    return playlist_embedding


def recommend_tracks(playlist_embedding, all_track_embeddings, track_ids, top_n=5):
    scores = torch.cosine_similarity(playlist_embedding, all_track_embeddings)
    top_indices = scores.argsort(descending=True)[:top_n]
    return [track_ids[i] for i in top_indices]


if __name__ == "__main__":
    raw_files = os.getenv("RAW_FILES")
    metadata_file = os.getenv("METADATA_FILE")

    df, unique_tracks, metadata = load_playlists(raw_files)
    features_df = pd.read_csv(metadata_file).set_index("track_id").dropna()

    # Prepare features
    common_tracks = features_df.index.intersection(metadata["track_id"])
    metadata_features = features_df.loc[common_tracks]
    prepared_features = prepare_features(metadata_features)
    feature_tensor = torch.tensor(prepared_features.values, dtype=torch.float32)

    track_to_idx = {tid: idx for idx, tid in enumerate(prepared_features.index)}
    idx_to_track = {idx: tid for tid, idx in track_to_idx.items()}

    # Model setup
    model = DualTowerModel(input_dim=prepared_features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CosineEmbeddingLoss()

    # Simple training loop (illustrative, minimal)
    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for _, row in df.sample(n=100).iterrows():
            playlist_tracks = [
                t["track_uri"].split("spotify:track:")[1]
                for t in row["tracks"]
                if t["track_uri"].split("spotify:track:")[1] in track_to_idx
            ]
            if len(playlist_tracks) < 2:
                continue
            idxs = [track_to_idx[tid] for tid in playlist_tracks]
            embeddings = model(feature_tensor[idxs])
            target = torch.ones(len(idxs) - 1)
            loss = loss_fn(embeddings[:-1], embeddings[1:], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Choose random playlist
    sample_playlist = df.sample(1).iloc[0]
    sample_playlist_tracks = [
        t["track_uri"].split("spotify:track:")[1] for t in sample_playlist["tracks"]
    ]

    playlist_embedding = embed_playlist(
        sample_playlist_tracks, feature_tensor, track_to_idx, model
    )
    if playlist_embedding is None:
        print("No valid tracks in playlist for embedding.")
        exit()

    # Recommend tracks
    model.eval()
    with torch.no_grad():
        all_track_embeddings = model(feature_tensor)
        recommended_track_ids = recommend_tracks(
            playlist_embedding, all_track_embeddings, prepared_features.index.tolist()
        )

    original_tracks = get_track_details(sample_playlist_tracks[:5], metadata)
    recommendations = get_track_details(recommended_track_ids, metadata)

    print(f"\nSample tracks from playlist '{sample_playlist['name']}':")
    for _, row in original_tracks.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']}")

    print("\nRecommended tracks:")
    for _, row in recommendations.iterrows():
        print(f"- {row['track_name']} by {row['artist_name']}")
