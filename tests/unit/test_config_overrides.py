"""Lightweight tests verifying the global NethoConfig override mechanism."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nethobench.utils.evaluation_constants import config
from nethobench.cross.metrics import lead_lag_peak
from nethobench.etho.metrics import stationary_score
from nethobench.neuro.metrics.sensitive import _welch_relative_psd


class TestConfigUpdate:
    def test_update_from_dict_changes_value(self):
        old = config.WELCH_SAMPLING_FREQUENCY
        try:
            config.update_from_dict({"WELCH_SAMPLING_FREQUENCY": 99.0})
            assert config.WELCH_SAMPLING_FREQUENCY == 99.0
        finally:
            config.update_from_dict({"WELCH_SAMPLING_FREQUENCY": old})

    def test_dot_notation_assignment(self):
        old = config.AUTOCORR_MAX_LAG
        try:
            config.AUTOCORR_MAX_LAG = 100
            assert config.AUTOCORR_MAX_LAG == 100
        finally:
            config.AUTOCORR_MAX_LAG = old


class TestConfigAffectsMetrics:
    def test_lead_lag_respects_max_lag_config(self):
        """A smaller LEAD_LAG_DEFAULT_MAX_LAG should constrain the detected peak."""
        old_max_lag = config.LEAD_LAG_DEFAULT_MAX_LAG
        try:
            t = np.arange(100, dtype=float)
            neural = np.sin(0.1 * t).reshape(-1, 1)
            behavior = np.sin(0.1 * (t + 2))  # behavior leads by 2

            config.update_from_dict({"LEAD_LAG_DEFAULT_MAX_LAG": 30})
            lag_default = lead_lag_peak([neural], [behavior])

            config.update_from_dict({"LEAD_LAG_DEFAULT_MAX_LAG": 1})
            lag_restricted = lead_lag_peak([neural], [behavior])

            assert lag_default == 2
            assert lag_restricted != lag_default  # constrained search can't reach true lag
        finally:
            config.update_from_dict({"LEAD_LAG_DEFAULT_MAX_LAG": old_max_lag})

    def test_stationary_score_respects_percentile_config(self):
        """Changing STATIONARY_THRESHOLD_PERCENTILE must change the score."""
        old_pct = config.STATIONARY_THRESHOLD_PERCENTILE
        try:
            # GT: first 3 frames stationary, then moving; INF: fully stationary
            df = pd.DataFrame({
                "sequenceId": [0] * 10,
                "itemPosition": range(10),
                "CENTER_X_gt": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "CENTER_Y_gt": np.zeros(10),
                "CENTER_X_inf": np.zeros(10),
                "CENTER_Y_inf": np.zeros(10),
            })

            config.update_from_dict({"STATIONARY_THRESHOLD_PERCENTILE": 0})
            score_low = stationary_score(df)[0]

            config.update_from_dict({"STATIONARY_THRESHOLD_PERCENTILE": 100})
            score_high = stationary_score(df)[0]

            assert score_low < score_high
        finally:
            config.update_from_dict({"STATIONARY_THRESHOLD_PERCENTILE": old_pct})

    def test_welch_nperseg_config(self):
        """Changing WELCH_NPERSEG must change the PSD frequency resolution."""
        old_nperseg = config.WELCH_NPERSEG
        try:
            x = np.random.default_rng(0).normal(size=512)

            config.update_from_dict({"WELCH_NPERSEG": 256})
            freqs_256, _ = _welch_relative_psd(x)

            config.update_from_dict({"WELCH_NPERSEG": 128})
            freqs_128, _ = _welch_relative_psd(x)

            assert len(freqs_128) < len(freqs_256)
        finally:
            config.update_from_dict({"WELCH_NPERSEG": old_nperseg})
