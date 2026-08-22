import numpy as np
import pandas as pd


# ============================================================
# LOAD DATASET AND PREPARE LSTM CLASSIFICATION SEQUENCES
# ============================================================

def prepare_classification_data(
        file_path="data/LSTM_ready_stable_dataset.xlsx",
        sequence_length=15
):

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    print("\nLoading classification dataset...")

    df = pd.read_excel(file_path)

    print("Original dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())


    # ========================================================
    # 2. REMOVE RARE CRITICAL CLASS
    # ========================================================

    df = df[df["target"] != 4].copy()

    print(
        "\nDataset shape after removing Critical class:",
        df.shape
    )


    # ========================================================
    # 3. BASE SENSOR AND ENVIRONMENT FEATURES
    # ========================================================

    feature_columns = [

        "noise_dB_z",
        "light_lux_z",
        "humidity_pct_z",
        "temp_C_z",

        "HR_bpm_z",
        "HRV_ms_z",
        "GSR_uS_z"
    ]


    # ========================================================
    # 4. ONE-HOT ENCODE ACTIVITY
    # ========================================================

    activity_dummies = pd.get_dummies(
        df["activity_code"],
        prefix="activity",
        dtype=np.float32
    )

    df = pd.concat(
        [df, activity_dummies],
        axis=1
    )

    # Get names of the one-hot encoded activity columns
    activity_columns = activity_dummies.columns.tolist()

    # Add activity columns to the input features
    feature_columns = (
        feature_columns +
        activity_columns
    )


    # ========================================================
    # 5. PRINT FEATURES USED
    # ========================================================

    print("\nFeatures used:")

    for feature in feature_columns:

        print("-", feature)


    print(
        "\nTotal number of input features:",
        len(feature_columns)
    )


    # ========================================================
    # 6. CREATE EMPTY DATA CONTAINERS
    # ========================================================

    X_train = []
    y_train = []

    X_val = []
    y_val = []

    X_test = []
    y_test = []


    # ========================================================
    # 7. PROCESS EACH PERSON SEPARATELY
    # ========================================================

    for person_id, person_data in df.groupby("person_id"):

        # Sort data chronologically
        person_data = person_data.sort_values(
            "time_sec"
        )


        # Extract features
        features = person_data[
            feature_columns
        ].values.astype(np.float32)


        # Extract stress targets
        targets = person_data[
            "target"
        ].values.astype(np.int32)


        # Get person's split
        split = person_data[
            "split"
        ].iloc[0]


        # Skip person if there are not enough
        # time steps to form a sequence
        if len(person_data) < sequence_length:

            continue


        # ====================================================
        # 8. CREATE SLIDING LSTM WINDOWS
        # ====================================================

        for i in range(
            len(person_data) - sequence_length + 1
        ):

            # Input sequence
            X_sequence = features[
                i:i + sequence_length
            ]


            # Target = stress class at the final time step
            y_label = targets[
                i + sequence_length - 1
            ]


            # ================================================
            # ADD TO CORRECT SPLIT
            # ================================================

            if split == "train":

                X_train.append(
                    X_sequence
                )

                y_train.append(
                    y_label
                )


            elif split == "validation":

                X_val.append(
                    X_sequence
                )

                y_val.append(
                    y_label
                )


            elif split == "test":

                X_test.append(
                    X_sequence
                )

                y_test.append(
                    y_label
                )


    # ========================================================
    # 9. CONVERT LISTS TO NUMPY ARRAYS
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
    # 10. PRINT FINAL DATA SHAPES
    # ========================================================

    print("\n" + "=" * 55)

    print("FINAL DATA SHAPES")

    print("=" * 55)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print()

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print()

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)


    # ========================================================
    # 11. PRINT CLASS DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 55)

    print("CLASS DISTRIBUTION")

    print("=" * 55)

    print(
        "\nTrain:",
        np.bincount(
            y_train,
            minlength=4
        )
    )

    print(
        "Validation:",
        np.bincount(
            y_val,
            minlength=4
        )
    )

    print(
        "Test:",
        np.bincount(
            y_test,
            minlength=4
        )
    )


    # ========================================================
    # 12. RETURN PREPARED DATA
    # ========================================================

    return (
        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test
    )
