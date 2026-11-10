"""Sign, support, tie and held-out calibration checks for timing diagnostics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import unittest
import numpy as np
import pandas as pd

import noise_prediction_model.analysis.audit_contact_alignment_impact as audit
from noise_prediction_model.analysis.summarize_contact_alignment_impact import fractional_rho


class AlignmentImpactTests(unittest.TestCase):
    def test_sample_shift_advances_trace_and_preserves_core(self):
        rng = np.random.default_rng(101)
        noise = rng.normal(size=(4, len(audit.REL)))
        delayed = rng.normal(size=noise.shape)
        delayed[:, 3:] = noise[:, :-3]
        np.testing.assert_allclose(audit.waveform_rho(delayed, noise, 6), 1.)
        self.assertTrue((audit.waveform_rho(delayed, noise, 0) < .9).all())
        # All shifts use identical noise support [-20,+60], not truncated overlap.
        with self.assertRaises(ValueError):audit.waveform_rho(delayed, noise, 1)
        with self.assertRaises(ValueError):audit.waveform_rho(delayed, noise, 100)

    def test_calibration_excludes_evaluated_trial_and_test(self):
        events = pd.DataFrame(dict(split=['train','train','train','test'],
            trial=['a','a','b','c'],leg=['FL']*4,rate_delta_ms=[12.,12.,-4.,30.],
            rate_peak_at_edge=[False]*4,rate_peak_tie_width_ms=[0.]*4))
        shifts, table = audit.heldout_shifts(events)
        np.testing.assert_array_equal(shifts, [-4.,-4.,12.,4.])
        # Test has no influence on any calibration value.
        events.loc[3,'rate_delta_ms']=-30.
        np.testing.assert_array_equal(audit.heldout_shifts(events)[0],shifts)
        self.assertFalse(table.calibration_uses_audio.any())

    def test_extrema_ties_and_search_boundaries_are_explicit(self):
        values=np.ones((2,len(audit.REL)))
        values[0,(audit.REL>=4)&(audit.REL<=8)]=4.
        values[1,audit.REL==0]=4.
        time,width,edge=audit.peak_time(values,0,40)
        np.testing.assert_array_equal(time,[6.,0.])
        np.testing.assert_array_equal(width,[4.,0.])
        np.testing.assert_array_equal(edge,[False,True])

    def test_magnitude_gap_can_exist_without_rate_timing_gap(self):
        # A force rise accompanies velocity recovery; the velocity minimum
        # naturally precedes the force maximum even with identical clocks.
        t=audit.REL.astype(float)
        ramp=np.clip((t+4)/12,0,1)[None,:]
        force=80*ramp
        velocity=-1+1.0*ramp
        fr=audit.plateau_peak(audit.rate(force,3))[0]
        vr=audit.plateau_peak(audit.rate(velocity,3))[0]
        np.testing.assert_allclose(fr,vr)
        ft=audit.peak_time(force,0,40)[0]
        vt=audit.peak_time(velocity,-20,20,minimum=True)[0]
        self.assertGreater((ft-vt)[0],0)

    def test_fractional_shift_uses_same_noise_support_and_defined_sign(self):
        rng=np.random.default_rng(13)
        signal=rng.normal(size=(3,len(audit.REL)))
        noise=rng.normal(size=signal.shape)
        # With s=+1ms the diagnostic averages current and next 2ms sample.
        noise[:,audit.CORE]=(signal[:,audit.CORE]+signal[:,audit.CORE+1])/2
        np.testing.assert_allclose(fractional_rho(signal,noise,1.),1.)
        self.assertTrue(np.isnan(fractional_rho(np.ones_like(signal),noise,1.)).all())

    def test_post40_window_can_censor_repeated_late_force_peak(self):
        time=audit.REL
        initial=50*np.exp(-((time-8)/5)**2)
        late=100*np.exp(-((time-46)/12)**2)
        force=(initial+late)[None,:]
        peak40,_,edge40=audit.peak_time(force,0,40)
        peak60,_,edge60=audit.peak_time(force,0,60)
        np.testing.assert_allclose(peak40,40.)
        np.testing.assert_allclose(peak60,46.)
        self.assertTrue(edge40[0]);self.assertFalse(edge60[0])


if __name__=='__main__':unittest.main()
