
import os
import glob
import numpy as np
import pandas as pd
import librosa
from spafe.features.lfcc import lfcc
from tqdm import tqdm


def process_audio_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        if len(y_trimmed) < 16000:
            y_trimmed = np.pad(
                y_trimmed,
                (0, 16000 - len(y_trimmed)),
                mode="constant"
            )

        lfccs = lfcc(
            sig=y_trimmed,
            fs=sr,
            num_ceps=20,
            pre_emph=0.97
        )

        return np.mean(lfccs, axis=0)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def build_dataset():

    data_rows = []

    # Correct folder paths
    fake_paths = glob.glob("data/fake/*")
    real_paths = glob.glob("data/real/*")

    # Process fake audio files (Label: 0)
    print(f"Found {len(fake_paths)} fake audio files. Processing...")

    for file_path in tqdm(fake_paths):

        if os.path.isfile(file_path) and file_path.lower().endswith((".wav", ".mp3", ".flac")):

            features = process_audio_file(file_path)

            if features is not None:
                data_rows.append(list(features) + [0])

    # Process real audio files (Label: 1)
    print(f"Found {len(real_paths)} real audio files. Processing...")

    for file_path in tqdm(real_paths):

        if os.path.isfile(file_path) and file_path.lower().endswith((".wav", ".mp3", ".flac")):

            features = process_audio_file(file_path)

            if features is not None:
                data_rows.append(list(features) + [1])

    if len(data_rows) == 0:
        print("Error: No audio files found! Check folder locations.")
        return

    col_names = [f"feat_{i}" for i in range(20)] + ["label"]

    df = pd.DataFrame(data_rows, columns=col_names)

    df.to_csv("dataset.csv", index=False)

    print(f"\nSuccess! Processed {len(data_rows)} audio files.")
    print("Dataset saved as dataset.csv")

    print("\nLabel distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    build_dataset()
