import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

from data import IMG_SIZE, MODEL_PATH, load_splits, check_leakage, make_dataset

# Phase 1: train the new classification head on frozen MobileNetV2 features
HEAD_EPOCHS = 30
HEAD_LR = 1e-3

# Phase 2: fine-tune the top of MobileNetV2 at a much lower learning rate
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 1e-5
FINE_TUNE_FROM = 100  # MobileNetV2 has 154 layers; layers from this index onward are unfrozen

EARLY_STOP_PATIENCE = 5


def compute_class_weights(train_df):
    """Square root of the 'balanced' weights.

    Fully balanced weights pushed the model to over-predict the rare BBPS 1 class
    (18 predictions, only 4 correct). Taking the square root keeps the push toward
    rare classes but halves its strength on a log scale.
    """
    classes = np.unique(train_df['label'].values)
    balanced = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'].values)
    class_weight_dict = {int(c): float(np.sqrt(w)) for c, w in zip(classes, balanced)}
    for c in [0, 1, 2, 3]:
        class_weight_dict.setdefault(c, 1.0)
    print("Class weights (sqrt-balanced):", {c: round(w, 3) for c, w in sorted(class_weight_dict.items())})
    return class_weight_dict


def build_model():
    base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    # MobileNetV2's ImageNet weights expect pixels in [-1, 1]. Stored in the model so
    # anything that loads the saved model gets the right scaling automatically.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    # training=False keeps BatchNorm layers in inference mode, including during
    # fine-tuning. Updating BatchNorm statistics on a small dataset destroys the
    # pretrained features.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(4, activation='softmax')(x)

    return models.Model(inputs, outputs), base_model


def early_stopping():
    # Stops when validation loss stops improving and rolls back to the best epoch
    return callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )


def main():
    train_df, val_df, test_df = load_splits()
    check_leakage(train_df, val_df, test_df)

    class_weight_dict = compute_class_weights(train_df)

    train_ds = make_dataset(train_df, augment=True, shuffle=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    model, base_model = build_model()

    # --- PHASE 1: HEAD ONLY ---
    print("\n=== Phase 1: training classification head (base frozen) ===")
    model.compile(optimizer=optimizers.Adam(HEAD_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stopping()],
    )
    head_best_val_loss = min(head_history.history['val_loss'])
    head_weights = model.get_weights()
    print(f"Phase 1 best val_loss: {head_best_val_loss:.4f}")

    # --- PHASE 2: FINE-TUNE TOP OF BASE ---
    print(f"\n=== Phase 2: fine-tuning MobileNetV2 layers {FINE_TUNE_FROM}+ ===")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_FROM]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    trainable_count = sum(layer.trainable for layer in base_model.layers)
    print(f"Unfrozen base layers: {trainable_count} of {len(base_model.layers)}")

    # Recompile so the trainable change and lower learning rate take effect
    model.compile(optimizer=optimizers.Adam(FINE_TUNE_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stopping()],
    )
    tune_best_val_loss = min(tune_history.history['val_loss'])
    print(f"Phase 2 best val_loss: {tune_best_val_loss:.4f}")

    # Keep fine-tuning only if it actually improved validation loss
    if tune_best_val_loss >= head_best_val_loss:
        print("Fine-tuning did not improve validation loss. Reverting to Phase 1 weights.")
        model.set_weights(head_weights)
    else:
        print("Fine-tuning improved validation loss. Keeping fine-tuned weights.")

    # --- FINAL EVALUATION ---
    print("\nEvaluating model against held-out test folder...")
    _, test_acc = model.evaluate(test_ds)
    print(f"\nFinal Test Set Accuracy Score: {test_acc * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"Model pipeline complete. Saved as '{MODEL_PATH}'")
    print("Run evaluate_model.py for QWK, per-class and adequacy metrics.")


if __name__ == "__main__":
    main()