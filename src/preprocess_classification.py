import numpy as np
import pandas as pd


# ============================================================
# LOAD DATASET AND PREPARE LSTM SEQUENCES
# ============================================================

def prepare_classification_data(
        file_path="data/LSTM_ready_stable_dataset.xlsx",
        sequence_length=15
):

    print("\nLoading classification dataset...")

    df = pd.read_excel(file_path)

    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())


    # ========================================================
    # REMOVE RARE CRITICAL CLASS
    # ========================================================

    df = df[df["target"] != 4].copy()

    print(
        "\nDataset shape after removing Critical class:",
        df.shape
    )


    # ========================================================
    # FEATURE COLUMNS
    # ========================================================

    feature_columns = [

        "noise_z",
        "light_z",
        "humidity_z",
        "temperature_z",

        "heart_rate_z",
        "hrv_z",
        "gsr_z",

        "activity_code"
    ]


    # ========================================================
    # CREATE 60-SECOND SEQUENCES
    # ========================================================

    X_train = []
    y_train = []

    X_val = []
    y_val = []

    X_test = []
    y_test = []


    # ========================================================
    # PROCESS EACH PERSON
    # ========================================================

    for person_id, person_data in df.groupby("person_id"):

        person_data = person_data.sort_values("time_sec")

        features = person_data[
            feature_columns
        ].values

        targets = person_data[
            "target"
        ].values


        # Skip incomplete sequences
        if len(person_data) < sequence_length:
            continue


        # Create sliding windows
        for i in range(
                len(person_data) - sequence_length + 1
        ):

            X_sequence = features[
                i:i + sequence_length
            ]

            y_label = targets[
                i + sequence_length - 1
            ]


            # Get split from person's first row
            split = person_data[
                "split"
            ].iloc[0]


            if split == "train":

                X_train.append(X_sequence)

                y_train.append(y_label)


            elif split == "validation":

                X_val.append(X_sequence)

                y_val.append(y_label)


            elif split == "test":

                X_test.append(X_sequence)

                y_test.append(y_label)


    # ========================================================
    # CONVERT TO NUMPY ARRAYS
    # ========================================================

    X_train = np.array(
        X_train,
        dtype=np.float32
    )

    y_train = np.array(
        y_train,
        dtype=np.int32
    )


    X_val = np.array(
        X_val,
        dtype=np.float32
    )

    y_val = np.array(
        y_val,
        dtype=np.int32
    )


    X_test = np.array(
        X_test,
        dtype=np.float32
    )

    y_test = np.array(
        y_test,
        dtype=np.int32
    )


    # ========================================================
    # PRINT FINAL SHAPES
    # ========================================================

    print("\nFINAL DATA SHAPES")

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)


    print("\nCLASS DISTRIBUTION")

    print(
        "Train:",
        np.bincount(y_train)
    )

    print(
        "Validation:",
        np.bincount(y_val)
    )

    print(
        "Test:",
        np.bincount(y_test)
    )


    return (
        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test
    )
