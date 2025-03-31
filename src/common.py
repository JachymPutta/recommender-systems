import json
import glob
import concurrent.futures
import pandas as pd
import numpy as np
import os


def load_metadata(metadata_file):
    return pd.read_csv(metadata_file)


def load_playlists(raw_files):
    filenames = glob.glob(raw_files)
    track_uris = []
    track_metadata = {}

    def load_file(filename):
        with open(filename) as f:
            json_data = json.load(f)
            for playlist in json_data["playlists"]:
                for track in playlist["tracks"]:
                    track_id = track["track_uri"].split("spotify:track:")[1]
                    track_uris.append(track_id)
                    if track_id not in track_metadata:
                        track_metadata[track_id] = {
                            "track_name": track["track_name"],
                            "artist_name": track["artist_name"],
                        }
            return pd.DataFrame(json_data["playlists"])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        dfs = list(executor.map(load_file, filenames))

    df = pd.concat(dfs)
    unique_tracks = np.unique(track_uris)

    # Convert track_metadata dictionary to DataFrame
    metadata_df = (
        pd.DataFrame.from_dict(track_metadata, orient="index")
        .reset_index()
        .rename(columns={"index": "track_id"})
    )

    return df, unique_tracks, metadata_df


def get_track_details(track_ids, metadata):
    metadata_indexed = metadata.set_index("track_id")
    available_ids = metadata_indexed.index.intersection(track_ids)
    missing_ids = set(track_ids) - set(available_ids)

    details = metadata_indexed.loc[available_ids][["track_name", "artist_name"]]
    details.reset_index(inplace=True)

    # Handle missing metadata gracefully
    if missing_ids:
        missing_df = pd.DataFrame(
            {
                "track_id": list(missing_ids),
                "track_name": ["Unknown track"] * len(missing_ids),
                "artist_name": ["Unknown artist"] * len(missing_ids),
            }
        )
        details = pd.concat([details, missing_df], ignore_index=True)

    return details
