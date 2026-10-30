"""Data loading for the Nerthus BBPS dataset.

Expected layout:
    nerthus-dataset-frames/
        0/  bowel_<video>_score_<score>-<n>_<frame>.jpg
        1/
        2/
        3/

Nerthus has no separate test folder, so this module splits the frames into
train / validation / test itself. Frames from one video are near-identical, so
the split is made by VIDEO: every frame of a video lands in the same subset.
Splitting by frame would put near-copies of test frames into training and
inflate every result.

Nothing in this module runs on import.
"""
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# --- CONFIGURATION ---
DATA_DIR = 'nerthus-dataset-frames'
MODEL_PATH = 'nerthus_bbps_model.keras'
SPLIT_PATH = 'nerthus_splits.csv'  # written by mainNerthus.py, read by evaluate_nerthus.py

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASSES = [0, 1, 2, 3]

# Share of each class's VIDEOS (not frames) held out for validation and test
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# bowel_1_score_0-0_00000001 -> video 'bowel_1', score 0, frame 1
FILENAME_RE = re.compile(r'(bowel_\d+)_score_(\d)-\d+_(\d+)', re.IGNORECASE)


# --- LOADING ---
def parse_filename(filename):
    match = FILENAME_RE.search(filename)
    if match is None:
        return None
    return {
        'video': match.group(1).lower(),
        'file_score': int(match.group(2)),
        'frame': int(match.group(3)),
    }


def build_dataset_df(data_dir=DATA_DIR):
    """One row per frame. The label comes from the class folder (0 to 3)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    rows, unparsed, mismatched = [], [], []
    for label in CLASSES:
        class_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(class_dir):
            print(f"Warning: class folder missing -> {class_dir}")
            continue
        # os.listdir with a lower-case extension check avoids the duplicate
        # matches that '*.JPG' + '*.jpg' globbing gives on Windows and macOS
        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            info = parse_filename(filename)
            if info is None:
                unparsed.append(filename)
                continue
            if info['file_score'] != label:
                mismatched.append(f"{label}/{filename}")
            rows.append({
                'image_path': os.path.join(class_dir, filename),
                'label': label,
                'video': info['video'],
                'frame': info['frame'],
            })

    if unparsed:
        print(f"Warning: skipped {len(unparsed)} file(s) with an unrecognised name, e.g. {unparsed[:3]}")
    if mismatched:
        print(f"Warning: {len(mismatched)} file(s) have a filename score that differs from their "
              f"folder. The folder label is used. e.g. {mismatched[:3]}")

    df = pd.DataFrame(rows, columns=['image_path', 'label', 'video', 'frame'])
    if df.empty:
        raise ValueError(f"No images found under {data_dir}/0 to /3.")
    return df


def summarise(df, title):
    print(f"\n--- {title} ---")
    frames = df.groupby('label').size().reindex(CLASSES, fill_value=0)
    videos = df.groupby('label')['video'].nunique().reindex(CLASSES, fill_value=0)
    for c in CLASSES:
        print(f"BBPS {c}: {frames[c]:>5} frames from {videos[c]:>2} video(s)")
    print(f"Total:  {len(df):>5} frames from {df['video'].nunique():>2} video(s)")


# --- VIDEO-LEVEL SPLIT ---
def _video_labels(df):
    """Each video's label. Warns if a video's frames carry more than one score."""
    per_video = df.groupby('video')['label']
    mixed = per_video.nunique()
    mixed = mixed[mixed > 1]
    if len(mixed):
        print(f"Note: {len(mixed)} video(s) contain frames with more than one score "
              f"({list(mixed.index)[:5]}). Each is assigned to a split by its most common score; "
              "all its frames still keep their own labels.")
    return per_video.agg(lambda s: s.mode().iloc[0])


def split_by_video(df):
    """Assign every video to train, val or test, separately within each class.

    Splitting inside each class keeps every class represented wherever the number of
    videos allows. A class always keeps at least one video for training. It gets a
    test video once it has 2 or more videos, and a validation video once it has 3 or more.
    """
    rng = np.random.default_rng(SEED)
    video_label = _video_labels(df)
    assignment = {}

    for c in CLASSES:
        videos = sorted(video_label[video_label == c].index)
        videos = list(rng.permutation(videos)) if videos else []
        n = len(videos)
        n_test = max(1, round(n * TEST_FRACTION)) if n >= 2 else 0
        n_val = max(1, round(n * VAL_FRACTION)) if n - n_test >= 2 else 0
        for v in videos[:n_test]:
            assignment[v] = 'test'
        for v in videos[n_test:n_test + n_val]:
            assignment[v] = 'val'
        for v in videos[n_test + n_val:]:
            assignment[v] = 'train'

        if n_test == 0:
            print(f"Warning: BBPS {c} has {n} video(s), too few for a test video. "
                  "Its test metrics will be unavailable.")
        elif n_val == 0:
            print(f"Warning: BBPS {c} has {n} video(s), too few for a validation video.")

    df = df.copy()
    df['split'] = df['video'].map(assignment)
    return df


def print_split_table(df):
    print("\n--- Split by video: frames (videos) ---")
    print(f"{'':<8}" + "".join(f"{'BBPS ' + str(c):>16}" for c in CLASSES))
    for split in ['train', 'val', 'test']:
        part = df[df['split'] == split]
        cells = []
        for c in CLASSES:
            sub = part[part['label'] == c]
            cells.append(f"{len(sub)} ({sub['video'].nunique()})")
        print(f"{split:<8}" + "".join(f"{cell:>16}" for cell in cells))


def check_video_leakage(df):
    """Every video must sit in exactly one split."""
    splits_per_video = df.groupby('video')['split'].nunique()
    leaked = list(splits_per_video[splits_per_video > 1].index)
    if leaked:
        raise RuntimeError(f"Videos found in more than one split: {leaked}")
    print("\nLeakage check: every video is in exactly one split.")


def create_splits():
    """Build, split and save the dataset. Returns (train_df, val_df, test_df)."""
    df = build_dataset_df()
    summarise(df, "Nerthus dataset")
    df = split_by_video(df)
    print_split_table(df)
    check_video_leakage(df)
    df.to_csv(SPLIT_PATH, index=False)
    print(f"Split saved to {SPLIT_PATH}")
    return tuple(df[df['split'] == s].reset_index(drop=True) for s in ['train', 'val', 'test'])


def load_saved_splits():
    """Read the exact split used in training, so evaluation uses the same test videos."""
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"{SPLIT_PATH} not found. Run mainNerthus.py first.")
    df = pd.read_csv(SPLIT_PATH)
    return tuple(df[df['split'] == s].reset_index(drop=True) for s in ['train', 'val', 'test'])


# --- TF.DATA PIPELINE ---
# Augmentation runs on raw [0, 255] pixels. Scaling to MobileNetV2's [-1, 1] range
# happens inside the model, so the saved model handles it at inference too.
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
])


def _load_image(path, label):
    image = tf.io.read_file(path)
    # decode_image handles JPEG and PNG; expand_animations=False guarantees a 3D tensor
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)  # float32, still in [0, 255]
    return image, tf.one_hot(label, depth=len(CLASSES))


def make_dataset(df, augment=False, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, df['label'].values))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)