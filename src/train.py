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
N_PERSONS = 150
RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ============================================================
# 2. LOAD PREPROCESSED DATA
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
# 3. BUILD LSTM MODEL
# ============================================================

model = tf.keras.Sequential([

    # Input layer
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

    tf.keras.layers.Dropout(0.2),

    # Second LSTM layer
    tf.keras.layers.LSTM(
        32
    ),

    tf.keras.layers.Dropout(0.2),

    # Dense layer
    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    # Output: predicted stress score
    tf.keras.layers.Dense(
        1
    )
])


# ============================================================
# 4. COMPILE MODEL
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
# 5. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 6. CALLBACKS
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
# 7. TRAIN MODEL
# ============================================================

print("\nStarting LSTM training...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
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
# 8. SAVE FINAL MODEL
# ============================================================

model.save(
    "models/final_stress_lstm.keras"
)

print(
    "\nFinal model saved successfully!"
)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

print(
    "\nEvaluating model on unseen test persons..."
)

y_pred = model.predict(
    X_test
).flatten()


# ============================================================
# 10. CALCULATE PERFORMANCE METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 50)
print("FINAL MODEL PERFORMANCE")
print("=" * 50)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("=" * 50)


# ============================================================
# 11. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 12. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. PLOT ACTUAL VS PREDICTED STRESS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.xlabel(
    "Actual Stress Score"
)

plt.ylabel(
    "Predicted Stress Score"
)

plt.title(
    "Actual vs Predicted Stress Score"
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.grid(True)

plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. SAVE METRICS
# ============================================================

with open(
        "results/metrics.txt",
        "w"
) as file:

    file.write(
        "LSTM Stress Prediction Results\n"
    )

    file.write(
        "=" * 40 + "\n\n"
    )

    file.write(
        f"MAE: {mae:.4f}\n"
    )

    file.write(
        f"MSE: {mse:.4f}\n"
    )

    file.write(
        f"RMSE: {rmse:.4f}\n"
    )

    file.write(
        f"R2 Score: {r2:.4f}\n"
    )
    # ============================================================
# TRAINING VS VALIDATION LOSS GRAPH
# ============================================================

import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Mean Squared Error")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "\nResults saved in the results folder!"
)
# ============================================================
# ACTUAL VS PREDICTED STRESS SCORE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction reference line
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Stress Score")

plt.xlabel("Actual Stress Score")

plt.ylabel("Predicted Stress Score")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ============================================================
# ACTUAL VS PREDICTED STRESS OVER TEST SAMPLES
# ============================================================

plt.figure(figsize=(14, 6))

# Show first 200 test samples
n_samples = min(200, len(y_test))

plt.plot(
    y_test[:n_samples],
    label="Actual Stress Score"
)

plt.plot(
    y_pred[:n_samples],
    label="Predicted Stress Score"
)

plt.title(
    "Actual vs Predicted Stress Score Across Test Samples"
)

plt.xlabel("Test Sample")

plt.ylabel("Stress Score")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/stress_prediction_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
