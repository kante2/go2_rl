"""Causal feature checks: no future access, gap resets and fair cohort matching."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_features import add_contact_features,feature_names,DEFAULT_CONFIG
from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import build_windows,select_ranked_features

class ContactFeatureTests(unittest.TestCase):
    def frame(self):
        t=np.arange(200)*.002
        frame=pd.DataFrame(dict(trial='trial',time_abs=t,time_from_move_start=t,segment=0,
            command_vx=.5,sp_vel_x=.5,audio_dbfs_20ms=np.sin(t*10)))
        for leg in ('fr','fl','rr','rl'):
            frame['ls_ff_'+leg]=np.where((t>=.04)&(t<.16),20.,0.)
            frame['sp_foot_vel_'+leg+'_z']=np.arange(len(t))*1.
        return frame

    def test_future_changes_do_not_change_past_features(self):
        frame=self.frame();whole=add_contact_features(frame)
        prefix=add_contact_features(frame.iloc[:70])
        np.testing.assert_allclose(whole[feature_names()].iloc[:70],prefix[feature_names()],equal_nan=True)
        frame.loc[70:,'ls_ff_fr']=9999;frame.loc[70:,'sp_foot_vel_fr_z']=-9999
        changed=add_contact_features(frame)
        np.testing.assert_allclose(whole[feature_names()].iloc[:70],changed[feature_names()].iloc[:70],equal_nan=True)

    def test_velocity_waits_for_confirmation_and_is_before_onset(self):
        derived=add_contact_features(self.frame())
        self.assertEqual(derived.contact_pre_valid_fr.iloc[20],0)
        self.assertEqual(derived.contact_pre_valid_fr.iloc[22],0)
        self.assertEqual(derived.contact_pre_valid_fr.iloc[23],1)
        self.assertEqual(derived.contact_pre_vz_body_fr.iloc[23],19)
        self.assertAlmostEqual(derived.contact_age_fr.iloc[23],6)
        self.assertEqual(derived.contact_pre_valid_fr.iloc[83],0)

    def test_gap_resets_cached_landing_and_physical_history(self):
        frame=self.frame();frame.loc[40:,'time_abs']+=1.;frame.loc[40:,'segment']=1
        derived=add_contact_features(frame)
        self.assertEqual(derived.contact_known_fr.iloc[40],0)
        self.assertTrue(np.isnan(derived.impact_integral20_fr.iloc[40]))
        self.assertEqual(derived.contact_on_fr.iloc[43],1)
        self.assertEqual(derived.contact_landing_valid_fr.iloc[43],0)
        self.assertEqual(derived.contact_pre_valid_fr.iloc[43],0)

    def test_force_integral_and_backward_rate(self):
        frame=self.frame();frame['ls_ff_fr']=20.
        derived=add_contact_features(frame)
        self.assertTrue(np.isnan(derived.impact_integral20_fr.iloc[9]))
        np.testing.assert_allclose(derived.impact_integral20_fr.iloc[10:],.4,atol=1e-10)
        np.testing.assert_allclose(derived.impact_rate6_fr.iloc[3:],0)
        np.testing.assert_allclose(derived.impact_rate_max20_fr.iloc[13:],0)
        derived=add_contact_features(self.frame())
        self.assertAlmostEqual(derived.impact_rate6_fr.iloc[20],20/.006)

    def test_missing_force_recovers_after_fresh_physical_history(self):
        frame=self.frame();frame["ls_ff_fr"]=20.;frame.loc[40,"ls_ff_fr"]=np.nan
        derived=add_contact_features(frame)
        self.assertTrue(np.isnan(derived.impact_integral20_fr.iloc[50]))
        self.assertAlmostEqual(derived.impact_integral20_fr.iloc[51],.4)
        self.assertEqual(derived.contact_landing_valid_fr.iloc[51],0)

    def test_only_velocity_features_need_sport_foot_velocity(self):
        frame=self.frame().drop(columns=['sp_foot_vel_fr_z'])
        add_contact_features(frame,requested=['impact_rate6_fr','contact_age_fr'])
        with self.assertRaisesRegex(ValueError,'Missing sp_foot_vel_fr_z'):
            add_contact_features(frame,requested=['contact_pre_vz_body_fr'])

    def test_matched_raw_and_derived_windows_have_identical_targets(self):
        frame=self.frame()
        raw=build_windows(frame,['ls_ff_fr'],contact_config=DEFAULT_CONFIG)
        aug=build_windows(frame,['ls_ff_fr']+feature_names(),contact_config=DEFAULT_CONFIG)
        pd.testing.assert_frame_equal(raw[2],aug[2]);np.testing.assert_array_equal(raw[1],aug[1])
        self.assertAlmostEqual(raw[2].anchor_time.iloc[0],.14)
        self.assertGreater(len(raw[1]),0)

    def test_prediction_windows_do_not_require_audio(self):
        frame=self.frame().drop(columns=["audio_dbfs_20ms"])
        x,y,meta,names,_=build_windows(frame,["impact_rate_max20_fr"],with_targets=False,contact_config=DEFAULT_CONFIG)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isnan(y).all())
        self.assertEqual(len(meta),len(x))

    def test_validity_dependency_is_included_within_k_budget(self):
        candidates=['ls_acc_x','contact_pre_vz_body_fr','contact_pre_valid_fr']
        ranked=pd.DataFrame(dict(feature=candidates,mean_abs_rho=[.8,.9,.1]))
        selected=select_ranked_features(ranked,candidates,2)
        self.assertEqual(set(selected),{'contact_pre_vz_body_fr','contact_pre_valid_fr'})
        self.assertEqual(select_ranked_features(ranked,candidates,1),['ls_acc_x'])

if __name__=='__main__':unittest.main()
