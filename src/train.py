import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import prepare_lstm_data


# ============================================================
# 1. SETTINGS
# ============================================================

SEQUENCE_LENGTH = 15
N_PERSONS = 700
RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ============================================================
# 2. SET RANDOM SEEDS
# ============================================================

np.random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. LOAD AND PREPROCESS DATA
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
# 4. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 5. BUILD IMPROVED LSTM MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    # First LSTM layer
    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    # Second LSTM layer
    tf.keras.layers.LSTM(
        32
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    # Dense layers
    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    # Output layer
    tf.keras.layers.Dense(
        1
    )
])


# ============================================================
# 6. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="mse",

    metrics=[
        "mae"
    ]
)


print("\nLSTM MODEL ARCHITECTURE")

model.summary()


# ============================================================
# 7. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True,

    verbose=1
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath="models/best_stress_lstm.keras",

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=0.00001,

    verbose=1
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nStarting Improved LSTM training...")

history = model.fit(

    X_train,
    y_train_scaled,

    validation_data=(
        X_val,
        y_val_scaled
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 9. SAVE FINAL MODEL
# ============================================================

model.save(
    "models/final_stress_lstm.keras"
)

print(
    "\nFinal model saved successfully!"
)


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

print(
    "\nEvaluating model on unseen test persons..."
)

y_pred_scaled = model.predict(
    X_test
).flatten()


# Convert predictions back to original stress score scale

y_pred_lstm = target_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()


# ============================================================
# 11. CALCULATE PERFORMANCE METRICS
# ============================================================

lstm_mae = mean_absolute_error(
    y_test,
    y_pred_lstm
)

lstm_mse = mean_squared_error(
    y_test,
    y_pred_lstm
)

lstm_rmse = np.sqrt(
    lstm_mse
)

lstm_r2 = r2_score(
    y_test,
    y_pred_lstm
)


# ============================================================
# 12. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 50)

print("FINAL MODEL PERFORMANCE")

print("=" * 50)

print(
    f"MAE  : {lstm_mae:.4f}"
)

print(
    f"MSE  : {lstm_mse:.4f}"
)

print(
    f"RMSE : {lstm_rmse:.4f}"
)

print(
    f"R²   : {lstm_r2:.4f}"
)

print("=" * 50)


# ============================================================
# 13. SAVE METRICS
# ============================================================

with open(
    "results/metrics.txt",
    "w"
) as file:

    file.write(
        "Improved LSTM Stress Prediction Results\n"
    )

    file.write(
        "=" * 45 + "\n\n"
    )

    file.write(
        f"MAE: {lstm_mae:.4f}\n"
    )

    file.write(
        f"MSE: {lstm_mse:.4f}\n"
    )

    file.write(
        f"RMSE: {lstm_rmse:.4f}\n"
    )

    file.write(
        f"R2 Score: {lstm_r2:.4f}\n"
    )


# ============================================================
# 14. TRAINING VS VALIDATION LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Improved LSTM: Training vs Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Scaled MSE Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. ACTUAL VS PREDICTED STRESS SCORE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_test,
    y_pred_lstm,
    alpha=0.6
)


# Perfect prediction reference line

min_value = min(
    y_test.min(),
    y_pred_lstm.min()
)

max_value = max(
    y_test.max(),
    y_pred_lstm.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title(
    "Improved LSTM: Actual vs Predicted Stress Score"
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
    "results/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. ACTUAL VS PREDICTED OVER TEST SAMPLES
# ============================================================

plt.figure(
    figsize=(14, 6)
)


# Display first 200 samples

n_samples = min(
    200,
    len(y_test)
)


plt.plot(
    y_test[:n_samples],
    label="Actual Stress Score"
)

plt.plot(
    y_pred_lstm[:n_samples],
    label="Predicted Stress Score"
)

plt.title(
    "Actual vs Predicted Stress Score Across Test Samples"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Stress Score"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/stress_prediction_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

import pandas as pd


prediction_results = pd.DataFrame({

    "Actual_Stress_Score":
        y_test,

    "Predicted_Stress_Score":
        y_pred_lstm,

    "Absolute_Error":
        np.abs(
            y_test -
            y_pred_lstm
        )
})


prediction_results.to_csv(

    "results/predictions.csv",

    index=False
)


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print(
    "\nResults saved successfully!"
)

print(
    "\nGenerated files:"
)

print(
    "- results/metrics.txt"
)

print(
    "- results/training_validation_loss.png"
)

print(
    "- results/actual_vs_predicted.png"
)

print(
    "- results/stress_prediction_comparison.png"
)

print(
    "- results/predictions.csv"
)

print(
    "- models/best_stress_lstm.keras"
)

print(
    "- models/final_stress_lstm.keras"
)
