"""Check clock indexing, gap boundaries, derivatives and lag interpretation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import noise_prediction_model.analysis.landing_event_dynamics as dynamics


class LandingDynamicsTests(unittest.TestCase):
    def test_pcm_matches_direct_dc_removed_samples(self):
        audio = np.array([100, 130, -50, 70, 200, -120, 30, 90], dtype=np.int16)
        # Audio sample i has time t0 + (i+1)*slope; [2,6) selects i=1..4.
        result, valid = dynamics.pcm_rms(audio, 1., 0., np.array([2., -1.]), np.array([6., 3.]))
        x = audio[1:5].astype(float); x -= x.mean()
        expected = 20 * np.log10(np.sqrt(np.mean(x*x)) / 32768)
        self.assertTrue(valid[0]); self.assertFalse(valid[1])
        self.assertAlmostEqual(result[0], expected)
        self.assertTrue(np.isnan(result[1]))

    def test_centered_rate_linear_and_gap(self):
        times = np.arange(40) * dynamics.DT
        segments = np.r_[np.zeros(20, int), np.ones(20, int)]
        rate = dynamics.centered_rate(3. * times + 2., segments)
        np.testing.assert_allclose(rate[3:17], 3.)
        self.assertTrue(np.isnan(rate[17:23]).all())
        self.assertTrue(np.isnan(rate[:3]).all())

    def test_lag_sign_for_known_delayed_noise(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(20, len(dynamics.REL_MS)))
        noise = rng.normal(size=x.shape)
        noise[:, 6:] = x[:, :-6]  # Noise follows driver by 12ms.
        lags = np.arange(-30, 32, 2)
        results = dynamics.lag_correlations(x, noise, lags_ms=lags)
        self.assertEqual(lags[np.argmax(results.mean(axis=0))], 12)
        np.testing.assert_allclose(results[:, lags == 12], 1.)

    def test_constant_correlations_undefined(self):
        result = dynamics.row_correlation(np.ones((2, 10)), np.tile(np.arange(10), (2, 1)))
        self.assertTrue(np.isnan(result).all())

    def test_within_trial_ranks_remove_group_offset(self):
        events = pd.DataFrame(dict(trial=['a']*4 + ['b']*4, leg=['FL']*8,
                                   synchronization_status=['CONSISTENT_CANDIDATE']*8,
                                   other_landings_in_extended_window=[0]*8))
        x = np.array([0, 1, 2, 3, 100, 101, 102, 103.])
        y = np.array([3, 2, 1, 0, 103, 102, 101, 100.])
        arrays = {name: np.repeat(x[:, None], len(dynamics.REL_MS), axis=1) for name in dynamics.DRIVERS}
        arrays['noise_5ms_dbfs'] = np.repeat(y[:, None], len(dynamics.REL_MS), axis=1)
        table = dynamics.time_resolved_table(events, arrays)
        self.assertTrue((table.pooled_spearman_rho > 0).all())
        np.testing.assert_allclose(table.within_trial_rank_rho, -1.)

    def test_equal_trial_profile_weighting(self):
        events = pd.DataFrame(dict(trial=['a']*9 + ['b'], leg=['FL']*10,
                                   synchronization_status=['CONSISTENT_CANDIDATE']*10,
                                   other_landings_in_extended_window=[0]*10))
        x = np.repeat(np.array([0]*9 + [10.])[:, None], len(dynamics.REL_MS), axis=1)
        arrays = {name: x for name in dynamics.SIGNALS}
        table = dynamics.profile_table(events, arrays)
        row = table[(table.subset == 'all_valid') & (table.leg == 'ALL')].iloc[0]
        self.assertEqual(row.event_median, 0.)
        self.assertEqual(row.trial_balanced_median, 5.)

    def test_extraction_preserves_pre_post_and_rejects_gaps(self):
        t = 100. + np.arange(201) * .002
        frame = pd.DataFrame(dict(time_abs=t, audio_delay_s=0., segment=0,
                                  audio_dbfs_20ms=-30., time_from_move_start=t-100))
        for leg in dynamics.LEGS:
            frame[f'ls_ff_{leg.lower()}'] = np.where(t < 100.2, 0., 60.)
            frame[f'foot_velocity_world_{leg.lower()}_z_mps'] = np.where(t < 100.2, -1., 0.)
        events = pd.DataFrame([dict(event_id='a:FL:0', trial='a', leg='FL', split='train',
                                    feature_valid=True, body_velocity_frame='body', contact_time_abs=t[100],
                                    contact_time_from_move_start=.2, synchronization_status='CONSISTENT_CANDIDATE')])
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / 'a.npz'
            np.savez(raw, audio=np.arange(10000, dtype=np.int16) % 100, audio_slope=.0001,
                     audio_t0=99.9, ls_t=t, sp_t=t)
            with patch.object(dynamics, 'add_foot_kinematics', side_effect=lambda f, **kwargs: f):
                accepted, arrays, audit, _ = dynamics.extract_trial(frame, events, raw)
                self.assertEqual(len(accepted), 1)
                self.assertTrue(audit.included.all())
                self.assertEqual(arrays['force'].shape, (1, 76))
                self.assertEqual(arrays['vz'][0, 24], -1.)
                self.assertEqual(arrays['vz'][0, 25], 0.)
                # Missing a row inside the pre-contact window must not be interpolated.
                gap = frame.drop(index=90).reset_index(drop=True)
                accepted, _, audit, _ = dynamics.extract_trial(gap, events, raw)
                self.assertTrue(accepted.empty)
                self.assertEqual(audit.exclusion_reason.iloc[0], 'expanded_window_or_derivative_crosses_gap')


if __name__ == '__main__':
    unittest.main()
