import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def load_and_preprocess_data(
        file_path,
        sequence_length=30,
        save_objects=True):

    # Load the Excel dataset
    df = pd.read_excel(file_path)

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # Sort data so each person's records remain in time order
    df = df.sort_values(
        by=["person_id", "time_sec"]
    ).reset_index(drop=True)

    # Encode categorical columns

    condition_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()

    df["condition_encoded"] = condition_encoder.fit_transform(
        df["condition"]
    )

    df["activity_encoded"] = activity_encoder.fit_transform(
        df["activity"]
    )

    # Features used by the LSTM
    feature_columns = [
        "noise_dB",
        "light_lux",
        "humidity_pct",
        "temp_C",
        "HR_bpm",
        "HRV_ms",
        "GSR_uS",
        "condition_encoded",
        "activity_encoded"
    ]

    # Target to predict
    target_column = "stress_score"

    # Store sequences
    X_sequences = []
    y_sequences = []

    # Create sequences separately for every person
    for person_id in df["person_id"].unique():

        person_data = df[
            df["person_id"] == person_id
        ].reset_index(drop=True)

        features = person_data[
            feature_columns
        ].values

        targets = person_data[
            target_column
        ].values

        # Create sliding-window sequences
        for i in range(
                len(person_data) - sequence_length):

            X_sequences.append(
                features[
                    i:i + sequence_length
                ]
            )

            y_sequences.append(
                targets[
                    i + sequence_length
                ]
            )

    # Convert to NumPy arrays
    X = np.array(X_sequences)
    y = np.array(y_sequences)

    print("\nSequences created successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Get dimensions
    samples, timesteps, features_count = X.shape

    # Reshape before scaling
    X_reshaped = X.reshape(
        -1,
        features_count
    )

    # Standardize sensor features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_reshaped
    )

    # Convert back to LSTM format
    X_scaled = X_scaled.reshape(
        samples,
        timesteps,
        features_count
    )

    # Save preprocessing objects
    if save_objects:

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            scaler,
            "models/feature_scaler.pkl"
        )

        joblib.dump(
            condition_encoder,
            "models/condition_encoder.pkl"
        )

        joblib.dump(
            activity_encoder,
            "models/activity_encoder.pkl"
        )

    return X_scaled, y


if __name__ == "__main__":

    X, y = load_and_preprocess_data(
        "data/stress_dataset.xlsx",
        sequence_length=30
    )

    print("\nPreprocessing completed successfully.")
