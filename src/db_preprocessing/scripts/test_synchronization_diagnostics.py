"""Known-delay synthetic PCM/contact tests, including held-out validation failures."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from time_synchronization import (SyncConfig, estimate_footfall_delay, apply_audio_delay,
                                  select_consensus, summarize_window)
from synchronization_diagnostics import write_json
from preprocessing import discover_split_inputs


class SyntheticTrial:
    name = 'synthetic'

    def __init__(self, end_shift=0., silent=False, echo=False):
        t = np.arange(0, 18, .002)
        force = np.tile([65., 68., 77., 76.], (len(t), 1))
        contacts = np.arange(2.25, 16, .25)
        for i, at in enumerate(contacts):
            legs = (0, 3) if i % 2 else (1, 2)
            for leg in legs:
                force[(t >= at-.10) & (t < at), leg] = [5, 8, 17, 16][leg]
        fs = 8000
        audio = np.random.default_rng(11).normal(0, .1, int(18*fs))
        if not silent:
            for at in contacts:
                onset = at+.018+(end_shift if at >= 11 else 0)
                k = int(round(onset*fs))
                n = np.arange(int(.04*fs))
                burst = 500*np.exp(-n/(fs*.012))*np.sin(2*np.pi*1800*n/fs)
                audio[k:k+len(n)] += burst
                if echo:
                    echo_start=k+int(.030*fs)
                    audio[echo_start:echo_start+len(n)] += 4*burst
        self.data = dict(ls_t=t, ls_ff=force, audio=audio, audio_slope=np.array(1/fs),
                         audio_t0=np.array(-1/fs), ev_label=np.array(['move_start', 'move_stop']),
                         ev_t=np.array([2., 16.]))

    def load(self, key):
        return self.data[key]


class SynchronizationTests(unittest.TestCase):
    def test_known_positive_delay_with_different_force_baselines(self):
        summary, events, _ = estimate_footfall_delay(SyntheticTrial())
        self.assertTrue(summary['passed'], summary)
        self.assertAlmostEqual(summary['applied_audio_delay_s'], .018, delta=.003)
        self.assertLess(abs(summary['end_residual_ms']), 3)
        self.assertEqual(set(leg for event in events for leg in event['legs']), {'fr', 'fl', 'rr', 'rl'})
        self.assertGreater(summary['contact_thresholds']['rr']['off'], 17)

    def test_louder_late_resonance_does_not_move_first_acoustic_onset(self):
        plain, _, _ = estimate_footfall_delay(SyntheticTrial())
        echoed, _, _ = estimate_footfall_delay(SyntheticTrial(echo=True))
        self.assertTrue(echoed['passed'])
        self.assertAlmostEqual(plain['applied_audio_delay_s'], echoed['applied_audio_delay_s'], delta=.002)

    def test_end_changes_cannot_change_calibration_but_fail_validation(self):
        normal, _, _ = estimate_footfall_delay(SyntheticTrial())
        drifted, _, _ = estimate_footfall_delay(SyntheticTrial(end_shift=.030))
        self.assertEqual(normal['applied_audio_delay_s'], drifted['applied_audio_delay_s'])
        self.assertEqual(normal['contact_thresholds'], drifted['contact_thresholds'])
        self.assertFalse(drifted['passed'])
        self.assertEqual(drifted['status'], 'END_RESIDUAL_REVIEW')
        self.assertAlmostEqual(drifted['end_residual_ms'], 30., delta=3.)

    def test_silent_audio_cannot_be_certified_or_shifted(self):
        summary, _, _ = estimate_footfall_delay(SyntheticTrial(silent=True))
        self.assertFalse(summary['passed'])
        self.assertFalse(summary['correction_accepted'])
        self.assertEqual(summary['applied_audio_delay_s'], 0.)

    def test_competing_modes_are_rejected_instead_of_nearest_zero(self):
        events = [dict(group_id=i, matched=False, candidates=[
            dict(audio_time=i+.01, lag_s=.01, prominence=100.),
            dict(audio_time=i+.05, lag_s=.05, prominence=100.)]) for i in range(8)]
        config = SyncConfig()
        self.assertTrue(select_consensus(events, config))
        self.assertFalse(summarize_window(events, config)['usable'])

    def test_positive_shift_and_edges_preserve_robot_validity(self):
        t = np.arange(10)*.002
        data = dict(time_abs=t, audio_valid=np.ones(10, bool), robot_keys=['ls_ff'],
                    ls_ff_valid=np.ones(10, bool), valid=np.ones(10, bool))
        for key in ('audio_rms_20ms', 'audio_dbfs_20ms', 'audio_peak_2ms'):
            data[key] = np.arange(10, dtype=float)
        result = apply_audio_delay(data, .004)
        np.testing.assert_allclose(result['audio_peak_2ms'][:8], np.arange(2, 10))
        self.assertFalse(result['valid'][-1])
        np.testing.assert_array_equal(result['ls_ff_valid'], data['ls_ff_valid'])

    def test_exclusion_and_split_overlap_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, names in [('train', ['a', 'no_vel_1']), ('test', ['b'])]:
                (root/split).mkdir()
                for name in names:
                    (root/split/f'{name}.npz').touch()
            self.assertEqual([p.stem for p in discover_split_inputs(root)['train']], ['a'])
            (root/'test/a.npz').touch()
            with self.assertRaises(ValueError):
                discover_split_inputs(root)

    def test_json_report_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'
            write_json(path, dict(missing=np.nan, scalar=np.int64(3)))
            self.assertEqual(json.loads(path.read_text()), dict(missing=None, scalar=3))


if __name__ == '__main__':
    unittest.main()
