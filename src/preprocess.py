import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# 1. LOAD DATA AND RANDOMLY SELECT 150 PERSONS


def load_data(
        file_path,
        n_persons=150,
        random_state=42):

    # Load dataset
    df = pd.read_excel(file_path)

    print("Original dataset shape:", df.shape)
    print("Total persons available:",
          df["person_id"].nunique())

    # Randomly select 150 persons
    rng = np.random.default_rng(random_state)

    all_persons = df["person_id"].unique()

    selected_persons = rng.choice(
        all_persons,
        size=n_persons,
        replace=False
    )

    # Keep complete records of selected persons
    df = df[
        df["person_id"].isin(selected_persons)
    ].copy()

    # Sort by person and time
    df = df.sort_values(
        by=["person_id", "time_sec"]
    ).reset_index(drop=True)

    print("\nSelected persons:", len(selected_persons))
    print("Dataset shape after selection:", df.shape)

    return df, selected_persons


# ============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_features(df):

    condition_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()

    df["condition_encoded"] = (
        condition_encoder.fit_transform(
            df["condition"]
        )
    )

    df["activity_encoded"] = (
        activity_encoder.fit_transform(
            df["activity"]
        )
    )

    return (
        df,
        condition_encoder,
        activity_encoder
    )


# ============================================================
# 3. SPLIT PERSONS INTO TRAIN / VALIDATION / TEST
# ============================================================

def split_persons(
        persons,
        random_state=42):

    rng = np.random.default_rng(random_state)

    # Shuffle persons
    persons = np.array(persons)
    rng.shuffle(persons)

    total = len(persons)

    # 70% training
    train_end = int(total * 0.70)

    # 15% validation
    val_end = train_end + int(total * 0.15)

    train_persons = persons[:train_end]
    val_persons = persons[train_end:val_end]
    test_persons = persons[val_end:]

    print("\nPERSON-WISE SPLIT")
    print("Training persons:", len(train_persons))
    print("Validation persons:", len(val_persons))
    print("Testing persons:", len(test_persons))

    return (
        train_persons,
        val_persons,
        test_persons
    )


# ============================================================
# 4. CREATE SLIDING WINDOW SEQUENCES
# ============================================================

def create_sequences(
        df,
        persons,
        feature_columns,
        target_column,
        sequence_length=15):

    X_sequences = []
    y_sequences = []

    for person_id in persons:

        # Get one person's complete time-series data
        person_data = df[
            df["person_id"] == person_id
        ].sort_values(
            "time_sec"
        ).reset_index(drop=True)

        features = person_data[
            feature_columns
        ].values

        targets = person_data[
            target_column
        ].values

        # Create sliding windows
        for i in range(
                len(person_data) - sequence_length):

            # Input: previous 15 time steps
            X_sequences.append(
                features[
                    i:i + sequence_length
                ]
            )

            # Target: next stress score
            y_sequences.append(
                targets[
                    i + sequence_length
                ]
            )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# ============================================================
# 5. SCALE DATA WITHOUT DATA LEAKAGE
# ============================================================

def scale_data(
        X_train,
        X_val,
        X_test):

    # Get feature count
    n_features = X_train.shape[2]

    # Create scaler
    scaler = StandardScaler()

    # Reshape training data
    train_reshaped = X_train.reshape(
        -1,
        n_features
    )

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(
        train_reshaped
    )

    # Transform validation data
    val_reshaped = X_val.reshape(
        -1,
        n_features
    )

    X_val_scaled = scaler.transform(
        val_reshaped
    )

    # Transform test data
    test_reshaped = X_test.reshape(
        -1,
        n_features
    )

    X_test_scaled = scaler.transform(
        test_reshaped
    )

    # Reshape back to LSTM format

    X_train_scaled = X_train_scaled.reshape(
        X_train.shape
    )

    X_val_scaled = X_val_scaled.reshape(
        X_val.shape
    )

    X_test_scaled = X_test_scaled.reshape(
        X_test.shape
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# 6. COMPLETE PREPROCESSING PIPELINE
# ============================================================

def prepare_lstm_data(
        file_path,
        n_persons=150,
        sequence_length=15,
        random_state=42):

    # Load and select persons
    df, selected_persons = load_data(
        file_path,
        n_persons,
        random_state
    )

    # Encode categorical data
    df, condition_encoder, activity_encoder = (
        encode_features(df)
    )

    # Features used for prediction
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

    # Target
    target_column = "stress_score"

    # Split PERSONS before sequence creation
    (
        train_persons,
        val_persons,
        test_persons

    ) = split_persons(
        selected_persons,
        random_state
    )

    # Create sequences

    X_train, y_train = create_sequences(
        df,
        train_persons,
        feature_columns,
        target_column,
        sequence_length
    )

    X_val, y_val = create_sequences(
        df,
        val_persons,
        feature_columns,
        target_column,
        sequence_length
    )

    X_test, y_test = create_sequences(
        df,
        test_persons,
        feature_columns,
        target_column,
        sequence_length
    )

    # Scale data
    (
        X_train,
        X_val,
        X_test,
        scaler

    ) = scale_data(
        X_train,
        X_val,
        X_test
    )

    # Create models folder
    os.makedirs(
        "models",
        exist_ok=True
    )

    # Save preprocessing objects
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

    # Print final shapes
    print("\nFINAL LSTM DATA")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "y_val:",
        y_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# RUN PREPROCESSING
# ============================================================

if __name__ == "__main__":

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test

    ) = prepare_lstm_data(

        file_path="data/stress_dataset.xlsx",

        n_persons=150,

        sequence_length=15,

        random_state=42
    )

    print(
        "\nPreprocessing completed successfully!"
    )
