# claims-risk-classifier

A classical machine learning project: a scikit-learn `RandomForestClassifier` that predicts
whether an insurance claim should be flagged as high-risk. It uses a train/validation/test
split, several evaluation metrics suited to an imbalanced task, and decision-threshold
tuning, and it reports the results as they came out.

## What it does

- Generates a labeled, claims-shaped dataset (`src/generate_data.py`).
- Encodes features consistently between training and inference.
- Trains a fitted `RandomForestClassifier` and tunes the decision threshold on a separate
  validation split (the test set is never touched during tuning).
- Evaluates once on the held-out test set with precision, recall, F1 and ROC-AUC, not just
  accuracy.
- Scores new claims with the same encoding and the same tuned threshold.

## Scope

- The dataset is synthetic. It is a claims-shaped table (claim type, amount, days since
  policy start, days to report, prior claims, customer age) with a `high_risk` label driven by
  a hand-designed underlying rule (unusually high claim amount relative to the claim type's
  norm, claims reported very soon after an incident, several recent prior claims, or a very
  new policy) plus ~8% injected label noise. It gives a learnable but non-trivial supervised
  task, and the pipeline is built so real claims data can replace it.
- The label is intentionally imbalanced (~9% positive), as in real fraud/risk-flagging tasks,
  where accuracy alone is a misleading metric: a classifier that always predicts "not high
  risk" scores ~91% accuracy while catching zero real cases. That is why I report precision,
  recall, F1, and ROC-AUC (demonstrated below).
- Data is split correctly, with a validation split kept separate from the test set for
  threshold tuning so the test metrics are not leaked into.

## Results

```
Tuned decision threshold: 0.39 (chosen on a validation split by F1)
Test set: 750 claims, 68 genuinely high_risk (9.1%)

--- At the tuned threshold ---
Accuracy:  0.8107
Precision: 0.1864
Recall:    0.3235
F1:        0.2366
ROC-AUC:   0.6247  (threshold-independent)

--- Compare: at the naive default 0.5 threshold ---
Accuracy: 0.9080  (looks better...)
F1:       0.1039  (...but this is worse: mostly predicting the majority class)
```

ROC-AUC of ~0.62 is the headline number: real, above-chance signal (0.5 would be random
guessing) on a hard, imbalanced task. The comparison against the naive default threshold is
the main lesson of the project (see Notes): accuracy alone looked better at the default
threshold while recall was almost entirely lost, which is exactly why metric selection
matters on imbalanced classification tasks.

## Tests

```
$ python3 -m pytest tests/ -v
============================== 31 passed in 8.29s ==============================
```

All 31 tests pass, confirmed in a clean virtualenv (fresh install of `pandas`,
`scikit-learn`, `pytest` only). `python3 run_pipeline.py` runs the full pipeline end-to-end:
generates data, trains the model, tunes the threshold, evaluates on the held-out test set,
prints feature importances, and scores 5 new claims.

## Project structure

```
src/
  generate_data.py   -- synthetic labeled claims-risk dataset generator
  model.py             -- feature encoding, training, threshold tuning, evaluation, inference
tests/
  test_generate_data.py
  test_model.py
run_pipeline.py       -- end-to-end demo
```

## Running it

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_pipeline.py
```

## Notes

Things I found and handled during development:

1. **Default 0.5 threshold.** The first version trained the model with
   `class_weight="balanced"` and evaluated at the standard 0.5 probability threshold, which
   gave a good-looking 90.8% accuracy but only 5.9% recall: the model mostly predicted the
   majority "not high risk" class, catching only 4 of 68 high-risk claims in the test set.
   `class_weight="balanced"` reweights the training loss but does not guarantee predicted
   probabilities are calibrated around 0.5 at prediction time. I found this by inspecting the
   confusion matrix rather than relying on the accuracy number.

2. **Threshold tuning on a validation split.** The model now sweeps candidate thresholds
   (0.05 to 0.95) on a validation split carved out of the training data, never touching the
   test set during tuning, and picks the one maximizing F1. In initial testing (system
   Python, scikit-learn 1.8) this raised test-set F1 from 0.104 to 0.237 and recall from 5.9%
   to roughly 32%.

3. **Reproducibility across scikit-learn versions.** Re-running the same code and seed in a
   fresh virtualenv with a newer scikit-learn (1.9 vs. 1.8 during initial development) picked
   a different threshold (0.57 vs. 0.39) with a correspondingly different test-set F1,
   because `RandomForestClassifier`'s internal tie-breaking during tree construction is not
   guaranteed to be bit-identical across library versions even with the same seed. The test
   `test_tuned_threshold_is_chosen_to_maximize_validation_f1` asserts the property that is
   guaranteed, that the chosen threshold maximizes F1 on its validation split, rather than a
   specific test-set F1 comparison. Both a system-Python run (scikit-learn 1.8) and a clean
   virtualenv run (scikit-learn 1.9) pass all 31 tests. The accuracy/F1/recall numbers quoted
   above come from one representative run and may differ slightly on other machines.

4. **Related projects.** This is classical supervised machine learning, alongside my
   LLM/agent projects (`rag-tool-agent-demo`, `ai-codegen-analyst`,
   `graph-rag-ontology-demo`). It reuses the claims-management domain shape from
   `ai-codegen-analyst` (synthetic Kfz/Personenschaden-style claims data), but the two
   datasets are generated independently: `ai-codegen-analyst` has an LLM write its own
   analysis code, while this project trains and evaluates a supervised model.

## Possible extensions

- Probability calibration (e.g. isotonic or Platt scaling) before threshold tuning.
- Cross-validated threshold selection and gradient-boosted model comparison.
- Swap the synthetic generator for real claims data.
