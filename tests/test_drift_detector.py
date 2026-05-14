"""
tests/test_drift_detector.py

Tests for utils/drift_detector.py covering fix #9 (early drift detection):
  - PSI computed correctly on stable distributions
  - KS test detects obvious drift
  - DriftReport serialises to dict
  - detect_drift returns expected structure
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from utils.drift_detector import detect_drift, _psi_numeric


class TestPsiNumeric:
    def test_identical_distributions_psi_near_zero(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=1000)
        psi = _psi_numeric(data, data.copy())
        assert psi < 0.05

    def test_very_different_distributions_high_psi(self):
        rng = np.random.default_rng(42)
        expected = rng.normal(loc=0, scale=1, size=1000)
        actual   = rng.normal(loc=5, scale=1, size=1000)  # large shift
        psi = _psi_numeric(expected, actual)
        assert psi > 0.2  # significant shift


class TestDetectDrift:
    def _make_frames(self, n=500, shift=0.0, rng_seed=42):
        rng = np.random.default_rng(rng_seed)
        train = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        })
        test = pd.DataFrame({
            "a": rng.normal(shift, 1, n),
            "b": rng.normal(0, 1, n),
        })
        return train, test

    def test_no_drift_on_identical(self):
        train, _ = self._make_frames()
        report = detect_drift(train, train.copy())
        assert report.overall_severity in ("low", "none", "unknown")

    def test_detects_high_drift(self):
        train, test = self._make_frames(shift=10.0)  # extreme shift
        report = detect_drift(train, test)
        assert report.overall_severity in ("medium", "high")

    def test_report_to_dict_serialisable(self):
        import json
        train, test = self._make_frames(shift=0.5)
        report = detect_drift(train, test)
        d = report.to_dict()
        # Must be JSON-serialisable (needed for PipelineState)
        json.dumps(d)

    def test_report_has_expected_keys(self):
        train, test = self._make_frames()
        report = detect_drift(train, test)
        d = report.to_dict()
        for key in ("overall_severity", "overall_drift_score", "feature_stats"):
            assert key in d, f"Missing key: {key}"

    def test_flagged_features_is_list(self):
        train, test = self._make_frames(shift=10.0)
        report = detect_drift(train, test)
        assert isinstance(report.flagged_features, list)
