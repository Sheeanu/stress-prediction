import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)


# ============================================================
# LOAD AND SELECT PERSONS
# ============================================================

def load_data(
        file_path,
        n_persons=150,
        random_state=42):

    df = pd.read_excel(file_path)

    print("Original dataset shape:", df.shape)
    print(
        "Total persons available:",
        df["person_id"].nunique()
    )

    rng = np.random.default_rng(random_state)

    all_persons = df["person_id"].unique()

    selected_persons = rng.choice(
        all_persons,
        size=n_persons,
        replace=False
    )

    df = df[
        df["person_id"].isin(selected_persons)
    ].copy()

    df = df.sort_values(
        ["person_id", "time_sec"]
    ).reset_index(drop=True)

    print(
        "\nRandomly selected persons:",
        len(selected_persons)
    )

    print(
        "Selected dataset shape:",
        df.shape
    )

    return df, selected_persons


# ============================================================
# ENCODE CATEGORICAL FEATURES
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
# PERSON-WISE SPLIT
# ============================================================

def split_persons(
        persons,
        random_state=42):

    rng = np.random.default_rng(random_state)

    persons = np.array(persons)

    rng.shuffle(persons)

    total = len(persons)

    train_end = int(total * 0.70)

    val_end = (
        train_end +
        int(total * 0.15)
    )

    train_persons = persons[:train_end]

    val_persons = persons[
        train_end:val_end
    ]

    test_persons = persons[val_end:]

    print("\nPERSON-WISE SPLIT")

    print(
        "Training persons:",
        len(train_persons)
    )

    print(
        "Validation persons:",
        len(val_persons)
    )

    print(
        "Testing persons:",
        len(test_persons)
    )

    return (
        train_persons,
        val_persons,
        test_persons
    )


# ============================================================
# CREATE SLIDING WINDOWS
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

        for i in range(
                len(person_data)
                - sequence_length):

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

    return (
        np.array(
            X_sequences,
            dtype=np.float32
        ),

        np.array(
            y_sequences,
            dtype=np.float32
        )
    )


# ============================================================
# SCALE INPUT FEATURES
# ============================================================

def scale_features(
        X_train,
        X_val,
        X_test):

    n_features = X_train.shape[2]

    scaler = StandardScaler()

    train_reshaped = X_train.reshape(
        -1,
        n_features
    )

    X_train_scaled = scaler.fit_transform(
        train_reshaped
    )

    val_reshaped = X_val.reshape(
        -1,
        n_features
    )

    X_val_scaled = scaler.transform(
        val_reshaped
    )

    test_reshaped = X_test.reshape(
        -1,
        n_features
    )

    X_test_scaled = scaler.transform(
        test_reshaped
    )

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
# SCALE TARGET
# ============================================================

def scale_targets(
        y_train,
        y_val,
        y_test):

    target_scaler = StandardScaler()

    y_train_scaled = (
        target_scaler.fit_transform(
            y_train.reshape(-1, 1)
        )
        .flatten()
    )

    y_val_scaled = (
        target_scaler.transform(
            y_val.reshape(-1, 1)
        )
        .flatten()
    )

    y_test_scaled = (
        target_scaler.transform(
            y_test.reshape(-1, 1)
        )
        .flatten()
    )

    return (
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        target_scaler
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def prepare_lstm_data(
        file_path,
        n_persons=150,
        sequence_length=15,
        random_state=42):

    df, selected_persons = load_data(
        file_path,
        n_persons,
        random_state
    )

    (
        df,
        condition_encoder,
        activity_encoder

    ) = encode_features(df)

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

    target_column = "stress_score"

    (
        train_persons,
        val_persons,
        test_persons

    ) = split_persons(
        selected_persons,
        random_state
    )

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

    (
        X_train,
        X_val,
        X_test,
        feature_scaler

    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    (
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        target_scaler

    ) = scale_targets(
        y_train,
        y_val,
        y_test
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        feature_scaler,
        "models/feature_scaler.pkl"
    )

    joblib.dump(
        target_scaler,
        "models/target_scaler.pkl"
    )

    joblib.dump(
        condition_encoder,
        "models/condition_encoder.pkl"
    )

    joblib.dump(
        activity_encoder,
        "models/activity_encoder.pkl"
    )

    print("\nFINAL DATA SHAPES")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    return (
        X_train,
        y_train,
        y_train_scaled,

        X_val,
        y_val,
        y_val_scaled,

        X_test,
        y_test,
        y_test_scaled,

        target_scaler
    )
