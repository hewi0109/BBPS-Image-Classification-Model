import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from data import MODEL_PATH, TEST_DIR, build_dataset_df, make_dataset

CLASSES = [0, 1, 2, 3]
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


def main():
    # Data comes from data.py, which runs nothing on import, so the saved model
    # is evaluated as-is instead of being retrained.
    model = tf.keras.models.load_model(MODEL_PATH)
    test_df = build_dataset_df([TEST_DIR])
    test_ds = make_dataset(test_df, augment=False, shuffle=False)

    print("Running model predictions on test folder...")
    y_true = test_df['label'].values
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    accuracy = np.mean(y_true == y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    conf_matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    class_report = classification_report(
        y_true, y_pred,
        labels=CLASSES,
        target_names=[f'BBPS {c}' for c in CLASSES],
        zero_division=0,
    )

    # Adequacy: positive = inadequate prep (BBPS 0 or 1), the condition to detect
    adequacy = binary_metrics(y_true < ADEQUATE_THRESHOLD, y_pred < ADEQUATE_THRESHOLD)

    line = "=" * 64
    print("\n" + line)
    print("                  BBPS MODEL EVALUATION REPORT")
    print(line)
    print(f"Test images:                 {len(y_true)}")
    print(f"Overall Test Accuracy:       {accuracy * 100:.2f}%")
    print(f"Quadratic Weighted Kappa:    {qwk:.4f}  (Target: > 0.600)")
    majority = np.bincount(y_true, minlength=4).max() / len(y_true)
    print(f"Majority-class baseline:     {majority * 100:.2f}%  (Percentage accuracy if guessing most common score (BBPS 3))")

    missing = [c for c in CLASSES if c not in y_true]
    if missing:
        print(f"\nWarning: no test images for BBPS {missing}. Those rows show 0.00 "
              "because there is no data, not because performance is zero.")

    print("-" * 64)
    print("\nPer-Class Precision, Recall and F1-Score:")
    print(class_report)

    print("-" * 64)
    print("\nPer-Class Sensitivity and Specificity (one class vs the rest):")
    print(f"{'Class':<10}{'Sensitivity':>16}{'Specificity':>16}{'Support':>10}")
    for c in CLASSES:
        m = binary_metrics(y_true == c, y_pred == c)
        print(f"BBPS {c:<5}{fmt_pct(m['sensitivity']):>16}{fmt_pct(m['specificity']):>16}{m['tp'] + m['fn']:>10}")

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
    print("                 B0   B1   B2   B3")
    for i, row in enumerate(conf_matrix):
        print(f"True BBPS {i}:    {row[0]:<4} {row[1]:<4} {row[2]:<4} {row[3]:<4}")
    print(line)


if __name__ == "__main__":
    main()