"""Shared data loading for the BBPS model.

Imported by main.py (training) and evaluate_model.py (evaluation).
Nothing in this module runs on import.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from tensorflow.keras import layers, Sequential

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VAL_FRACTION = 0.20
MODEL_PATH = 'custom_bbps_png_model.keras'

TRAIN_DIRS = ['BBPS Pictures/Left', 'BBPS Pictures/Right', 'BBPS Pictures/Transverse']
TEST_DIR = 'BBPS Pictures/Test'

# Regex with one capture group that pulls a patient or procedure ID out of a filename.
# Filenames carry no patient or procedure ID (the "V85" in "BBPS_3_V85.PNG" is only
# the image's index in its folder), so each image is its own group.
# If patient IDs become available, set a regex with one capture group here.
GROUP_ID_PATTERN = None

# Max differing bits (out of 64) for two images to be flagged as near-duplicates.
NEAR_DUPLICATE_DISTANCE = 3


# --- FILENAME PARSING ---
def extract_bbps_score(filename):
    filename_lower = filename.lower()
    for score in (0, 1, 2, 3):
        if f'bbps_{score}' in filename_lower:
            return score
    return None


def extract_group_id(filename):
    if GROUP_ID_PATTERN is None:
        return None
    match = re.search(GROUP_ID_PATTERN, filename, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def build_dataset_df(directories):
    rows = []
    seen = set()
    for folder in directories:
        if not os.path.exists(folder):
            print(f"Warning: Folder not found at -> {folder}")
            continue
        for ext in ('*.PNG', '*.png'):
            for img_path in glob.glob(os.path.join(folder, ext)):
                # On Windows and macOS, glob is case-insensitive, so both patterns
                # return the same file. Without this check every image is loaded
                # twice and copies can land in both training and validation.
                key = os.path.normcase(os.path.abspath(img_path))
                if key in seen:
                    continue
                seen.add(key)

                filename = os.path.basename(img_path)
                score = extract_bbps_score(filename)
                if score is None:
                    continue
                group = extract_group_id(filename)
                rows.append({
                    'image_path': img_path,
                    'label': score,
                    'group': group if group is not None else f'img:{key}',
                    'has_group_id': group is not None,
                })
    return pd.DataFrame(rows, columns=['image_path', 'label', 'group', 'has_group_id'])


# --- GROUP-AWARE SPLIT ---
def split_train_val(df):
    """Split so that every image from one patient/procedure stays on the same side."""
    if GROUP_ID_PATTERN is not None and not df['has_group_id'].all():
        missing = int((~df['has_group_id']).sum())
        print(f"\nWarning: {missing} image(s) did not match GROUP_ID_PATTERN "
              "and are split individually.")

    n_splits = round(1 / VAL_FRACTION)
    if df['label'].value_counts().min() >= 2:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        train_idx, val_idx = next(splitter.split(df, df['label'], groups=df['group']))
    else:
        print("\nWarning: a class has fewer than 2 images. Using an unstratified group split.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
        train_idx, val_idx = next(splitter.split(df, groups=df['group']))

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def load_splits():
    train_val_df = build_dataset_df(TRAIN_DIRS)
    test_df = build_dataset_df([TEST_DIR])
    if len(train_val_df) == 0 or len(test_df) == 0:
        raise ValueError("No matching .PNG files found. Double check your folder paths.")

    print("\n--- Training Folders Class Distribution ---")
    counts = train_val_df['label'].value_counts()
    for cls in sorted(counts.index):
        print(f"BBPS Class {cls}: {counts[cls]} images")

    train_df, val_df = split_train_val(train_val_df)
    print(f"Split Summary -> Training: {len(train_df)} | Validation: {len(val_df)} | Testing: {len(test_df)}")
    return train_df, val_df, test_df


# --- LEAKAGE CHECKS ---
def _dhash(path, hash_size=8):
    """64-bit difference hash. Near-identical images give near-identical hashes."""
    with Image.open(path) as img:
        img = img.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).flatten()


def _near_duplicates(df_a, df_b):
    hashes_a = np.array([_dhash(p) for p in df_a['image_path']])
    hashes_b = np.array([_dhash(p) for p in df_b['image_path']])
    pairs = []
    for i, h in enumerate(hashes_a):
        distances = (hashes_b != h).sum(axis=1)
        for j in np.flatnonzero(distances <= NEAR_DUPLICATE_DISTANCE):
            pairs.append((df_a['image_path'].iloc[i], df_b['image_path'].iloc[j], int(distances[j])))
    return pairs


def check_leakage(train_df, val_df, test_df):
    """Report shared patients/procedures and near-duplicate images across splits."""
    print("\n--- Leakage Check ---")
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    clean = True

    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        groups_a = set(splits[a].loc[splits[a]['has_group_id'], 'group'])
        groups_b = set(splits[b].loc[splits[b]['has_group_id'], 'group'])
        shared = groups_a & groups_b
        if shared:
            clean = False
            print(f"{len(shared)} patient/procedure ID(s) appear in both {a} and {b}: {sorted(shared)[:10]}")

        pairs = _near_duplicates(splits[a], splits[b])
        if pairs:
            clean = False
            print(f"{len(pairs)} near-duplicate image pair(s) between {a} and {b} (review manually):")
            for path_a, path_b, dist in pairs[:10]:
                print(f"  [{dist} bits] {path_a}  <->  {path_b}")

    if clean:
        print("No shared IDs or near-duplicate images found across splits.")
    return clean


# --- TF.DATA PIPELINE ---
# Augmentation runs on raw [0, 255] pixels. Scaling to MobileNetV2's [-1, 1] range
# happens inside the model (see main.py), so the saved model handles it at inference too.
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
])


def _load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)  # float32, still in [0, 255]
    return image, tf.one_hot(label, depth=4)


def make_dataset(df, augment=False, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, df['label'].values))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=SEED)
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)