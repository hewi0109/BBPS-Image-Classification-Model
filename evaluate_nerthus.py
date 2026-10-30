import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from data_nerthus import CLASSES, MODEL_PATH, load_saved_splits, make_dataset

# A segment score of 2 or higher is the usual threshold for adequate preparation
ADEQUATE_THRESHOLD = 2


def safe_div(num, den):
    return num / den if den else float('nan')


def fmt_pct(value):
    return "n/a (no cases)" if np.isnan(value) else f"{value * 100:6.2f}%"


def binary_metrics(y_true_pos, y_pred_pos):
    """Sensitivity, specificity, PPV and NPV for a boolean 'positive' condition."""
    tp = int(np.sum(y_true_pos & y_pred_pos))
    fn = int(np.sum(y_true_pos & ~y_pred_pos))
    tn = int(np.sum(~y_true_pos & ~y_pred_pos))
    fp = int(np.sum(~y_true_pos & y_pred_pos))
    return {
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
        'sensitivity': safe_div(tp, tp + fn),
        'specificity': safe_div(tn, tn + fp),
        'ppv': safe_div(tp, tp + fp),
        'npv': safe_div(tn, tn + fn),
    }


def print_metrics(y_true, y_pred, unit):
    accuracy = np.mean(y_true == y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=CLASSES)
    majority = np.bincount(y_true, minlength=len(CLASSES)).max() / len(y_true)

    print(f"Test {unit}s:{'':<{19 - len(unit)}}{len(y_true)}")
    print(f"Accuracy:                    {accuracy * 100:.2f}%")
    print(f"Quadratic Weighted Kappa:    {qwk:.4f}  (Target: > 0.600)")
    print(f"Majority-class baseline:     {majority * 100:.2f}%  (always predicting the most common class)")

    missing = [c for c in CLASSES if c not in y_true]
    if missing:
        print(f"\nWarning: no test {unit}s for BBPS {missing}. Those rows show 0.00 "
              "because there is no data, not because performance is zero.")

    print("-" * 64)
    print("\nPer-Class Precision, Recall and F1-Score:")
    print(classification_report(y_true, y_pred, labels=CLASSES,
                                target_names=[f'BBPS {c}' for c in CLASSES], zero_division=0))

    print("-" * 64)
    print("\nPer-Class Sensitivity and Specificity (one class vs the rest):")
    print(f"{'Class':<10}{'Sensitivity':>16}{'Specificity':>16}{'Support':>10}")
    for c in CLASSES:
        m = binary_metrics(y_true == c, y_pred == c)
        print(f"BBPS {c:<5}{fmt_pct(m['sensitivity']):>16}{fmt_pct(m['specificity']):>16}{m['tp'] + m['fn']:>10}")

    adequacy = binary_metrics(y_true < ADEQUATE_THRESHOLD, y_pred < ADEQUATE_THRESHOLD)
    print("-" * 64)
    print(f"\nBowel Prep Adequacy (inadequate = BBPS < {ADEQUATE_THRESHOLD}, the positive class):")
    print(f"  Sensitivity (inadequate preps caught):     {fmt_pct(adequacy['sensitivity'])}"
          f"   [{adequacy['tp']}/{adequacy['tp'] + adequacy['fn']}]")
    print(f"  Specificity (adequate preps passed):       {fmt_pct(adequacy['specificity'])}"
          f"   [{adequacy['tn']}/{adequacy['tn'] + adequacy['fp']}]")
    print(f"  PPV (flagged inadequate that truly are):   {fmt_pct(adequacy['ppv'])}"
          f"   [{adequacy['tp']}/{adequacy['tp'] + adequacy['fp']}]")
    print(f"  NPV (passed adequate that truly are):      {fmt_pct(adequacy['npv'])}"
          f"   [{adequacy['tn']}/{adequacy['tn'] + adequacy['fn']}]")

    print("-" * 64)
    print("\nConfusion Matrix:")
    print("                 Predicted Class")
    print("                 B0    B1    B2    B3")
    for i, row in enumerate(confusion_matrix(y_true, y_pred, labels=CLASSES)):
        print(f"True BBPS {i}:    {row[0]:<5} {row[1]:<5} {row[2]:<5} {row[3]:<5}")


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_df = load_saved_splits()
    test_ds = make_dataset(test_df, augment=False, shuffle=False)

    print("Running model predictions on test videos...")
    probs = model.predict(test_ds)
    y_true = test_df['label'].values
    y_pred = np.argmax(probs, axis=1)

    line = "=" * 64
    print("\n" + line)
    print("              NERTHUS BBPS EVALUATION: PER FRAME")
    print(line)
    print_metrics(y_true, y_pred, 'frame')

    # Frames from one video are strongly correlated, so frame counts overstate how much
    # independent evidence there is. The per-video view averages each video's frame
    # probabilities into one prediction, which is closer to how a score is assigned.
    prob_cols = [f'p{c}' for c in CLASSES]
    frame_probs = pd.DataFrame(probs, columns=prob_cols)
    frame_probs['video'] = test_df['video'].values
    frame_probs['label'] = y_true
    frame_probs['correct'] = y_true == y_pred
    videos = frame_probs.groupby('video').agg(
        label=('label', lambda s: s.mode().iloc[0]),
        frames=('label', 'size'),
        frame_accuracy=('correct', 'mean'),
        **{col: (col, 'mean') for col in prob_cols},
    )
    videos['predicted'] = np.argmax(videos[prob_cols].values, axis=1)

    print("\n" + line)
    print("        NERTHUS BBPS EVALUATION: PER VIDEO (mean probability)")
    print(line)
    print(f"{'Video':<12}{'True':>6}{'Predicted':>11}{'Frames':>8}{'Frames correct':>16}")
    for video, r in videos.sort_values(['label', 'frame_accuracy']).iterrows():
        print(f"{video:<12}{int(r['label']):>6}{int(r['predicted']):>11}{int(r['frames']):>8}"
              f"{r['frame_accuracy'] * 100:>15.1f}%")
    print("-" * 64)
    print_metrics(videos['label'].values.astype(int), videos['predicted'].values.astype(int), 'video')
    print(line)


if __name__ == "__main__":
    main()