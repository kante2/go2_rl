import unittest
import numpy as np
from contact_noise_visualization import detect_contact


class ContactDetectionTests(unittest.TestCase):
    def test_onset_is_backdated_but_causal_state_waits_for_confirmation(self):
        t=np.arange(15)*.002;force=np.r_[np.zeros(5),np.ones(10)*20]
        estimated,causal,events=detect_contact(t,force,np.zeros(15),confirm_ms=6)
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events[0]['onset_time_abs'],.010)
        self.assertAlmostEqual(events[0]['confirmed_time_abs'],.016)
        self.assertEqual(estimated[5],1)
        self.assertEqual(causal[5],0)
        self.assertEqual(causal[8],1)

    def test_short_pulses_and_hysteresis_band_do_not_create_contacts(self):
        t=np.arange(20)*.002;force=np.zeros(20);force[6:8]=20;force[12:16]=7
        estimated,causal,events=detect_contact(t,force,np.zeros(20))
        self.assertEqual(events,[])
        self.assertTrue((causal[3:]==0).all())

    def test_initial_contact_and_segment_restart_are_not_new_landings(self):
        t=np.r_[np.arange(10)*.002,.2+np.arange(10)*.002]
        estimated,causal,events=detect_contact(t,np.ones(20)*20,np.r_[np.zeros(10),np.ones(10)])
        self.assertEqual(events,[])
        self.assertEqual(causal[10],-1)
        self.assertEqual(causal[13],1)

    def test_missing_force_resets_state_without_false_transition(self):
        t=np.arange(20)*.002;force=np.ones(20)*20;force[10]=np.nan
        estimated,causal,events=detect_contact(t,force,np.zeros(20))
        self.assertEqual(events,[])
        self.assertEqual(estimated[10],-1)
        self.assertEqual(causal[11],-1)


if __name__=='__main__':unittest.main()
