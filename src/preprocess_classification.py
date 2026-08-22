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

    "noise_dB_z",
    "light_lux_z",
    "humidity_pct_z",
    "temp_C_z",

    "HR_bpm_z",
    "HRV_ms_z",
    "GSR_uS_z"
]


# ========================================================
# ONE-HOT ENCODE ACTIVITY
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

activity_columns = activity_dummies.columns.tolist()

feature_columns = (
    feature_columns +
    activity_columns
)


# ========================================================
# CREATE EMPTY DATA LISTS
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

        # Sort data according to time
        person_data = person_data.sort_values("time_sec")

        # Get sensor features
        features = person_data[
            feature_columns
        ].values

        # Get stress labels
        targets = person_data[
            "target"
        ].values


        # Get person's dataset split
        split = person_data[
            "split"
        ].iloc[0]


        # Skip if insufficient data
        if len(person_data) < sequence_length:
            continue


        # ====================================================
        # CREATE SLIDING LSTM WINDOWS
        # ====================================================

        for i in range(
                len(person_data) - sequence_length + 1
        ):

            X_sequence = features[
                i:i + sequence_length
            ]

            # Label of the final time step
            y_label = targets[
                i + sequence_length - 1
            ]


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
    # PRINT FINAL INFORMATION
    # ========================================================

    print("\n" + "=" * 50)
    print("FINAL DATA SHAPES")
    print("=" * 50)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)


    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)

    print(
        "\nTrain:",
        np.bincount(y_train, minlength=4)
    )

    print(
        "Validation:",
        np.bincount(y_val, minlength=4)
    )

    print(
        "Test:",
        np.bincount(y_test, minlength=4)
    )


    return (
        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test
    )
