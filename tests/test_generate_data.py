"""Tests for src/generate_data.py -- the synthetic labeled claims-risk
dataset generator."""
from __future__ import annotations

from src.generate_data import CLAIM_TYPES, CHANNELS, generate_claims_dataset


class TestGenerateClaimsDataset:
    def test_returns_requested_row_count(self):
        rows = generate_claims_dataset(n=500, seed=1)
        assert len(rows) == 500

    def test_deterministic_for_same_seed(self):
        rows_a = generate_claims_dataset(n=200, seed=7)
        rows_b = generate_claims_dataset(n=200, seed=7)
        assert rows_a == rows_b

    def test_different_seeds_produce_different_data(self):
        rows_a = generate_claims_dataset(n=200, seed=1)
        rows_b = generate_claims_dataset(n=200, seed=2)
        assert rows_a != rows_b

    def test_all_expected_columns_present(self):
        rows = generate_claims_dataset(n=50, seed=1)
        expected = {
            "claim_id", "claim_type", "channel", "claim_amount_eur",
            "days_since_policy_start", "days_to_report",
            "prior_claims_last_12mo", "customer_age", "high_risk",
        }
        for row in rows:
            assert expected.issubset(row.keys())

    def test_high_risk_label_is_binary(self):
        rows = generate_claims_dataset(n=500, seed=1)
        for row in rows:
            assert row["high_risk"] in (0, 1)

    def test_claim_type_values_are_from_known_set(self):
        rows = generate_claims_dataset(n=500, seed=1)
        for row in rows:
            assert row["claim_type"] in CLAIM_TYPES

    def test_channel_values_are_from_known_set(self):
        rows = generate_claims_dataset(n=500, seed=1)
        for row in rows:
            assert row["channel"] in CHANNELS

    def test_label_is_imbalanced_not_50_50(self):
        # The task is meant to be a realistic imbalanced classification
        # problem (like real fraud/risk flagging), not an artificially
        # balanced one.
        rows = generate_claims_dataset(n=3000, seed=42)
        positive_rate = sum(r["high_risk"] for r in rows) / len(rows)
        assert 0.03 < positive_rate < 0.25

    def test_label_is_not_purely_random_it_correlates_with_amount_ratio(self):
        # A basic sanity check that the label carries real signal: claims
        # with an unusually high amount relative to their claim-type norm
        # should be high_risk more often than claims with a typical amount,
        # even with the injected noise.
        from src.generate_data import _severity_base

        rows = generate_claims_dataset(n=3000, seed=42)
        high_ratio_rows = [r for r in rows if r["claim_amount_eur"] / _severity_base(r["claim_type"]) > 1.8]
        low_ratio_rows = [r for r in rows if r["claim_amount_eur"] / _severity_base(r["claim_type"]) <= 1.8]

        high_ratio_rate = sum(r["high_risk"] for r in high_ratio_rows) / len(high_ratio_rows)
        low_ratio_rate = sum(r["high_risk"] for r in low_ratio_rows) / len(low_ratio_rows)
        assert high_ratio_rate > low_ratio_rate

    def test_claim_ids_are_unique(self):
        rows = generate_claims_dataset(n=500, seed=1)
        ids = [r["claim_id"] for r in rows]
        assert len(ids) == len(set(ids))
