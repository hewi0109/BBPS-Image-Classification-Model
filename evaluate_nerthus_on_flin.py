"""External validation: run the Nerthus-trained model on the original BBPS Pictures dataset.

The Nerthus model never saw any BBPS Pictures images, so every image in all four
folders (Left, Right, Transverse, Test) is a valid test image. Results are
reported for all images combined.

Needs data.py (original dataset loader) and evaluate_nerthus.py in the same folder.
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from data import TRAIN_DIRS, TEST_DIR, build_dataset_df, make_dataset
from data_nerthus import CLASSES, MODEL_PATH
from evaluate_nerthus import print_metrics


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded Nerthus model: {MODEL_PATH}")

    folders = TRAIN_DIRS + [TEST_DIR]
    parts = []
    for folder in folders:
        df = build_dataset_df([folder])
        df['folder'] = folder.split('/')[-1]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        raise ValueError("No BBPS Pictures images found. Run this from the folder that contains 'BBPS Pictures'.")

    counts = df['label'].value_counts().reindex(CLASSES, fill_value=0)
    print("\n--- BBPS Pictures images per class ---")
    for c in CLASSES:
        print(f"BBPS {c}: {counts[c]} images")

    # Both models scale pixels inside the network, so raw 0-255 images are correct input here
    ds = make_dataset(df, augment=False, shuffle=False)
    probs = model.predict(ds)
    df['predicted'] = np.argmax(probs, axis=1)

    line = "=" * 64
    print("\n" + line)
    print("            NERTHUS MODEL ON BBPS PICTURES: OVERALL")
    print(line)
    print_metrics(df['label'].values, df['predicted'].values, 'image')
    print(line)

    df[['image_path', 'folder', 'label', 'predicted']].to_csv('nerthus_on_bbps_predictions.csv', index=False)
    print("Per-image predictions saved to nerthus_on_bbps_predictions.csv")


if __name__ == "__main__":
    main()