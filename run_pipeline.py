"""
run_pipeline.py
-----------------

End-to-end demo: generates a synthetic labeled claims dataset, trains a
real RandomForestClassifier with a properly held-out train/validation/
test split, tunes the decision threshold on the validation split only,
evaluates once on the untouched test split, and prints the full
result -- including the comparison against the naive default 0.5
threshold that turned out to be badly miscalibrated for this imbalanced
task.

Run with:
    python3 run_pipeline.py
"""
from __future__ import annotations

import pandas as pd

from src.generate_data import generate_claims_dataset
from src.model import feature_importances, predict_risk, train_and_evaluate


def main() -> None:
    print("=" * 70)
    print("claims-risk-classifier -- end-to-end demo")
    print("=" * 70)

    print("\nGenerating synthetic labeled claims-risk dataset...")
    rows = generate_claims_dataset(n=3000, seed=42)
    df = pd.DataFrame(rows)
    positive_rate = df["high_risk"].mean()
    print(f"  {len(df)} rows, {df['high_risk'].sum()} high_risk ({positive_rate:.1%})")
    print("  (synthetic data with a real, noisy underlying risk rule -- NOT real insurance data)")

    print("\nTraining RandomForestClassifier (train/validation/test split, stratified)...")
    trained = train_and_evaluate(df)
    e = trained.evaluation

    print(f"\n  Tuned decision threshold: {trained.decision_threshold} (chosen on validation split by F1)")
    print(f"  Test set: {e.n_test} claims, {e.n_positive_test} genuinely high_risk ({e.n_positive_test/e.n_test:.1%})")

    print("\n  --- At the tuned threshold ---")
    print(f"  Accuracy:  {e.accuracy:.4f}")
    print(f"  Precision: {e.precision:.4f}")
    print(f"  Recall:    {e.recall:.4f}")
    print(f"  F1:        {e.f1:.4f}")
    print(f"  ROC-AUC:   {e.roc_auc:.4f}  (threshold-independent)")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {e.confusion_matrix}")

    print("\n  --- Compare: at the naive default 0.5 threshold ---")
    print(f"  Accuracy: {e.accuracy_at_default_threshold:.4f}  (looks better...)")
    print(f"  F1:       {e.f1_at_default_threshold:.4f}  (...but this is worse: mostly predicting the majority class)")

    print("\n  --- Top 5 features by importance ---")
    for name, importance in feature_importances(trained, top_n=5):
        print(f"    {name}: {importance:.4f}")

    print("\nScoring 5 new (held-out-generation) claims with the trained model:")
    new_rows = generate_claims_dataset(n=5, seed=999)
    new_df = pd.DataFrame(new_rows)
    scored = predict_risk(trained, new_df)
    print(scored[["claim_id", "claim_type", "claim_amount_eur", "high_risk", "predicted_risk_probability", "predicted_high_risk"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("Notes:")
    print("- ROC-AUC (0.62ish) is the headline number: real, above-chance")
    print("  signal on a genuinely hard, imbalanced task.")
    print("- The default 0.5 threshold looks better on accuracy alone (it mostly")
    print("  predicts the majority 'not high risk' class) but is actually much")
    print("  worse at the actual job: catching real high-risk claims (recall).")
    print("  This is a common trap in imbalanced classification, which is why")
    print("  precision, recall, F1 and ROC-AUC are reported alongside accuracy.")
    print("- 'high_risk' predictions on new data are scored with the SAME feature")
    print("  encoding and the SAME tuned threshold used during evaluation, not a")
    print("  hardcoded 0.5 -- see predict_risk() in src/model.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
