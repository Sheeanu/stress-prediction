import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import prepare_lstm_data


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 15
N_PERSONS = 150
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading and preprocessing data...")

(
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

) = prepare_lstm_data(

    file_path="data/stress_dataset.xlsx",

    n_persons=N_PERSONS,

    sequence_length=SEQUENCE_LENGTH,

    random_state=RANDOM_STATE
)


# ============================================================
# FLATTEN 60-SECOND SEQUENCES FOR RANDOM FOREST
# ============================================================

print("\nPreparing data for Random Forest...")

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_val_flat = X_val.reshape(
    X_val.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


# ============================================================
# COMBINE TRAINING AND VALIDATION DATA
# ============================================================

X_rf_train = np.concatenate([
    X_train_flat,
    X_val_flat
])

y_rf_train = np.concatenate([
    y_train,
    y_val
])


print("\nRandom Forest Training Shape:")
print(X_rf_train.shape)


# ============================================================
# BUILD RANDOM FOREST
# ============================================================

rf_model = RandomForestRegressor(

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

rf_model.fit(
    X_rf_train,
    y_rf_train
)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print(
    "\nEvaluating Random Forest on unseen test persons..."
)

y_pred_rf = rf_model.predict(
    X_test_flat
)


# ============================================================
# CALCULATE METRICS
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    y_pred_rf
)

rf_mse = mean_squared_error(
    y_test,
    y_pred_rf
)

rf_rmse = np.sqrt(
    rf_mse
)

rf_r2 = r2_score(
    y_test,
    y_pred_rf
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 50)

print("RANDOM FOREST BASELINE PERFORMANCE")

print("=" * 50)

print(
    f"MAE  : {rf_mae:.4f}"
)

print(
    f"MSE  : {rf_mse:.4f}"
)

print(
    f"RMSE : {rf_rmse:.4f}"
)

print(
    f"R²   : {rf_r2:.4f}"
)

print("=" * 50)


# ============================================================
# SAVE MODEL AND RESULTS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


joblib.dump(
    rf_model,
    "models/random_forest_baseline.pkl"
)


results = pd.DataFrame({

    "Model": [
        "Random Forest"
    ],

    "MAE": [
        rf_mae
    ],

    "MSE": [
        rf_mse
    ],

    "RMSE": [
        rf_rmse
    ],

    "R2": [
        rf_r2
    ]
})


results.to_csv(
    "results/random_forest_results.csv",
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred_rf,
    alpha=0.6
)

min_value = min(
    y_test.min(),
    y_pred_rf.min()
)

max_value = max(
    y_test.max(),
    y_pred_rf.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title(
    "Random Forest: Actual vs Predicted Stress Score"
)

plt.xlabel(
    "Actual Stress Score"
)

plt.ylabel(
    "Predicted Stress Score"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/random_forest_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "\nRandom Forest model and results saved successfully!"
)
