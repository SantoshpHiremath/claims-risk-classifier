"""
generate_data.py
------------------

Generates a synthetic labeled insurance-claims dataset for a binary
classification task: predicting whether a claim is high-risk (flagged
for manual review) from claim features. Not real insurer data.

The label is generated from a genuine (if simple) underlying rule with
injected noise -- not randomly assigned -- so there is a real, learnable
signal for a classifier to find, but not a perfect one (mirroring how
real fraud/risk labels are noisy and imperfect in practice, and giving
the classifier's honestly-reported accuracy somewhere below 100% to
report, rather than a suspiciously perfect score).
"""
from __future__ import annotations

import random

CLAIM_TYPES = ["Kfz-Haftpflicht", "Kfz-Kasko", "Personenschaden", "Glasschaden", "Wildschaden"]
CHANNELS = ["Online-Portal", "Telefon", "Makler", "App"]


def _severity_base(claim_type: str) -> float:
    return {
        "Kfz-Haftpflicht": 3200.0,
        "Kfz-Kasko": 2100.0,
        "Personenschaden": 8500.0,
        "Glasschaden": 450.0,
        "Wildschaden": 1600.0,
    }[claim_type]


def generate_claims_dataset(n: int = 3000, seed: int = 42) -> list:
    """Generates n labeled claim records. The high_risk label follows a
    real underlying rule (large claim amount relative to claim-type
    norm, filed very soon after the reported incident date, and/or a
    customer with several recent prior claims) with ~8% label noise
    (both false positives and false negatives) injected on top, so the
    task is learnable but not trivial -- and a classifier reporting
    100% accuracy on held-out data would be a sign something is wrong
    (e.g. label leakage), not a sign of a great model.
    """
    rng = random.Random(seed)
    rows = []

    for i in range(1, n + 1):
        claim_type = rng.choice(CLAIM_TYPES)
        channel = rng.choice(CHANNELS)
        base = _severity_base(claim_type)

        claim_amount_eur = round(max(50.0, rng.gauss(base, base * 0.45)), 2)
        days_since_policy_start = rng.randint(1, 3650)
        days_to_report = rng.randint(0, 30)
        prior_claims_last_12mo = rng.choices([0, 1, 2, 3, 4], weights=[0.55, 0.25, 0.12, 0.06, 0.02])[0]
        customer_age = rng.randint(18, 85)

        # --- underlying (noisy) risk rule ---
        amount_ratio = claim_amount_eur / base
        risk_score = 0.0
        if amount_ratio > 1.8:
            risk_score += 0.4
        if days_to_report <= 2:
            risk_score += 0.25
        if prior_claims_last_12mo >= 3:
            risk_score += 0.35
        if days_since_policy_start < 30:
            risk_score += 0.3

        true_high_risk = risk_score >= 0.5

        # Inject ~8% label noise (flip the label) so the task isn't
        # trivially separable and the classifier's reported metrics
        # reflect a genuine, imperfect learning problem.
        label = true_high_risk
        if rng.random() < 0.08:
            label = not label

        rows.append({
            "claim_id": f"C{i:06d}",
            "claim_type": claim_type,
            "channel": channel,
            "claim_amount_eur": claim_amount_eur,
            "days_since_policy_start": days_since_policy_start,
            "days_to_report": days_to_report,
            "prior_claims_last_12mo": prior_claims_last_12mo,
            "customer_age": customer_age,
            "high_risk": int(label),
        })

    return rows


def write_csv(path: str, n: int = 3000, seed: int = 42) -> int:
    import csv
    rows = generate_claims_dataset(n=n, seed=seed)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    count = write_csv("data/claims_risk.csv")
    print(f"Generated {count} labeled claim rows.")
