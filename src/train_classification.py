import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

from sklearn.utils.class_weight import compute_class_weight

from preprocess_classification import prepare_classification_data
# ============================================================
# 1. SETTINGS
# ============================================================

SEQUENCE_LENGTH = 15

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001

RANDOM_STATE = 42


# ============================================================
# 2. SET RANDOM SEED
# ============================================================

np.random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

print("\nLoading and preprocessing classification data...")

(
    X_train,
    y_train,

    X_val,
    y_val,

    X_test,
    y_test

) = prepare_classification_data(

    file_path="data/LSTM_ready_stable_dataset.xlsx",

    sequence_length=SEQUENCE_LENGTH
)

# ============================================================
# 4. PRINT INPUT INFORMATION
# ============================================================

print("\n" + "=" * 50)
print("INPUT INFORMATION")
print("=" * 50)

print("Sequence Length:", X_train.shape[1])

print("Number of Features:", X_train.shape[2])

print("Number of Classes:", len(np.unique(y_train)))

# ============================================================
# 4A. CALCULATE CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(
        classes,
        class_weights_array
    )
)

print("\nCLASS WEIGHTS:")
print(class_weights)


# ============================================================
# 5. BUILD LSTM MODEL
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

    tf.keras.layers.Dropout(0.3),


    # Second LSTM layer
    tf.keras.layers.LSTM(
        32
    ),

    tf.keras.layers.Dropout(0.3),


    # Dense layer
    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),


    # Output layer
    tf.keras.layers.Dense(
        4,
        activation="softmax"
    )

])


# ============================================================
# 6. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\n" + "=" * 50)
print("LSTM CLASSIFICATION MODEL")
print("=" * 50)

model.summary()


# ============================================================
# 7. CREATE DIRECTORIES
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
# 8. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True,

    verbose=1
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath="models/best_stress_classifier.keras",

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
# 9. TRAIN MODEL
# ============================================================

print("\nStarting LSTM classification training...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================

model.save(
    "models/final_stress_classifier.keras"
)

print("\nFinal classification model saved successfully!")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

print("\nEvaluating model on test data...")

y_probabilities = model.predict(X_test)

y_pred = np.argmax(
    y_probabilities,
    axis=1
)


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(

    y_test,
    y_pred,

    average="weighted",

    zero_division=0
)


print("\n" + "=" * 50)
print("FINAL MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("=" * 50)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

class_names = [

    "Normal",
    "Mild",
    "Moderate",
    "High"
]


report = classification_report(

    y_test,
    y_pred,

    target_names=class_names,

    zero_division=0
)


print("\nCLASSIFICATION REPORT\n")

print(report)


# Save classification report

with open(
        "results/classification_report.txt",
        "w"
) as file:

    file.write(report)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(
    figsize=(8, 6)
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)

display.plot()

plt.title(
    "Confusion Matrix - Stress Classification"
)

plt.savefig(

    "results/confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. TRAINING VS VALIDATION LOSS
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
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(

    "results/classification_training_loss.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. TRAINING VS VALIDATION ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(

    "results/classification_training_accuracy.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. SAVE METRICS
# ============================================================

with open(
        "results/classification_metrics.txt",
        "w"
) as file:

    file.write(
        "LSTM Stress Classification Results\n"
    )

    file.write(
        "=" * 40 + "\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n"
    )


print(
    "\nResults saved successfully in the results folder!"
)
