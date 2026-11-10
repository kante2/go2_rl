"""Physical kinematics and event-window boundary regression checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from go2_kinematics import Go2Kinematics,world_foot_velocity,add_foot_kinematics
from contact_event_extraction import analyze_frame,pcm_window_statistics,collect_event_tables,atomic_csv


class KinematicsTests(unittest.TestCase):
    def test_neutral_pose_matches_urdf_dimensions_and_leg_sides(self):
        model=Go2Kinematics()
        for leg in ('fr','fl','rr','rl'):
            position,j=model.position_jacobian(leg,np.zeros((1,3)))
            x=.1934 if leg[0]=='f' else -.1934
            y=.142 if leg[1]=='l' else -.142
            np.testing.assert_allclose(position[0],[x,y,-.426],atol=1e-12)

    def test_all_leg_jacobians_match_independent_finite_differences(self):
        model=Go2Kinematics();rng=np.random.default_rng(42)
        q=rng.uniform([-.3,.4,-2.3],[.3,1.5,-1.1],(8,3))
        for leg in ('fr','fl','rr','rl'):
            _,jac=model.position_jacobian(leg,q)
            for joint in range(3):
                plus=q.copy();minus=q.copy();plus[:,joint]+=1e-6;minus[:,joint]-=1e-6
                numerical=(model.position_jacobian(leg,plus)[0]-model.position_jacobian(leg,minus)[0])/2e-6
                np.testing.assert_allclose(jac[:,:,joint],numerical,rtol=1e-7,atol=1e-9)

    def test_body_translation_rotation_and_joint_motion_are_each_added_once(self):
        rotation=Rotation.from_euler('z',[90],degrees=True)
        position=np.array([[1.,0,0]]);relative=np.array([[0.,0,-.4]])
        gyro=np.array([[0.,0,2.]]);velocity=np.array([[3.,0,0]])
        body=world_foot_velocity(position,relative,gyro,rotation,velocity,'body')
        np.testing.assert_allclose(body,[[-2.,3.,-.4]],atol=1e-12)
        odom=world_foot_velocity(position,relative,gyro,rotation,velocity,'odom')
        np.testing.assert_allclose(odom,[[1.,0.,-.4]],atol=1e-12)

    def test_invalid_attitude_is_rejected(self):
        f=EventTests.frame();f.loc[0,'ls_quaternion_w']=0.
        with self.assertRaisesRegex(ValueError,'quaternion'):add_foot_kinematics(f)


class EventTests(unittest.TestCase):
    @staticmethod
    def frame():
        t=np.arange(100)*.002
        f=pd.DataFrame(dict(time_abs=t,time_from_move_start=t,segment=0,trial='toy',command_vx=.3,
            audio_delay_s=0.,audio_dbfs_20ms=-20.,audio_rms_20ms=3276.8,audio_peak_2ms=5000.))
        for a in 'xyz':f[f'ls_gyro_{a}']=0.;f[f'sp_vel_{a}']=.3 if a=='x' else 0.
        for a in ('w','x','y','z'):f[f'ls_quaternion_{a}']=1. if a=='w' else 0.
        for leg in ('fr','fl','rr','rl'):
            f[f'ls_ff_{leg}']=np.where(t>=.080,30.,0.)
            for j in ('hip','thigh','calf'):
                f[f'ls_q_{leg}_{j}']={'hip':0.,'thigh':.8,'calf':-1.6}[j]
                f[f'ls_dq_{leg}_{j}']=.5 if j=='calf' else 0.
        return f

    def test_pre_velocity_uses_strictly_earlier_sample_and_not_sport_foot_velocity(self):
        f=self.frame();result=analyze_frame(f,'toy')
        rows=result['features']
        self.assertEqual(len(rows),4)
        self.assertTrue((rows.pre_velocity_time_abs<rows.contact_time_abs).all())
        self.assertTrue(np.allclose(rows.pre_velocity_offset_ms,-2.))
        self.assertTrue((rows.force_at_contact_sensor_units==30.).all())
        # No sp_foot_vel_* columns are supplied to this test.
        self.assertTrue(np.isfinite(rows.pre_speed_world_mps).all())

    def test_sensor_baseline_thresholds_are_shared_with_synchronization(self):
        f=self.frame()
        thresholds={}
        for leg,baseline in zip(('fr','fl','rr','rl'),(5.,8.,17.,16.)):
            f[f'ls_ff_{leg}']+=baseline
            thresholds[leg]=dict(on=baseline+10.,off=baseline+5.)
        result=analyze_frame(f,'toy',sync_summary=dict(contact_thresholds=thresholds))
        self.assertEqual(len(result['features']),4)
        for row in result['features'].to_dict('records'):
            self.assertEqual(row['contact_on_threshold'],thresholds[row['leg'].lower()]['on'])
        self.assertTrue(np.allclose(result['features'].contact_time_abs,.080))

    def test_future_joint_and_body_velocity_cannot_change_pre_velocity(self):
        f=self.frame();first=analyze_frame(f,'toy')['features']
        future=f.time_abs>=.080
        for col in f:
            if col.startswith(('ls_dq_','sp_vel_')):f.loc[future,col]=50.
        second=analyze_frame(f,'toy')['features']
        np.testing.assert_allclose(first.pre_speed_world_mps,second.pre_speed_world_mps)

    def test_window_crossing_missing_segment_is_flagged_and_samples_do_not_bridge(self):
        f=self.frame();f=f[(f.time_abs<.100)|(f.time_abs>=.150)]
        result=analyze_frame(f,'toy')
        self.assertTrue((~result['features'].window_complete).all())
        self.assertTrue((~result['features'].feature_valid).all())
        self.assertTrue((result['samples'].time_abs<.100).all())

    def test_other_feet_are_marked_in_shared_microphone_window(self):
        result=analyze_frame(self.frame(),'toy')
        self.assertTrue((result['features'].other_leg_contact_count_in_window==3).all())
        self.assertTrue((~result['features'].isolated_foot_event).all())

    def test_no_force_transition_produces_no_events(self):
        f=self.frame()
        for leg in ('fr','fl','rr','rl'):f[f'ls_ff_{leg}']=30.
        result=analyze_frame(f,'toy')
        self.assertTrue(result['features'].empty);self.assertTrue(result['samples'].empty)

    def test_pcm_window_has_recorded_delay_and_no_centered_rms_leakage(self):
        # Samples occur at .01,.02,.03,.04,.05. Positive delay shifts them earlier.
        context=dict(audio_slope=.01,audio_t0=0.,audio=np.array([0,0,100,-100,999]))
        rms,peak,valid=pcm_window_statistics(context,.015,.035,.01)
        self.assertTrue(valid)
        self.assertAlmostEqual(rms,20*np.log10(100/32768.))
        self.assertAlmostEqual(peak,rms)
        self.assertFalse(pcm_window_statistics(context,-1.,.1,0.)[2])

    def test_split_collection_is_idempotent_and_does_not_mix_test_into_train(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';fig=root/'figure'
            for split,name in [('train','a'),('test','b')]:
                atomic_csv(pd.DataFrame({'time_abs':[0.]}),data/split/f'{name}.csv')
                atomic_csv(pd.DataFrame(dict(event_id=[name+':FR:0'],trial=[name],split=[split],leg=['FR'])),fig/name/'contact_event_features.csv')
                atomic_csv(pd.DataFrame(dict(event_id=[name+':FR:0'],trial=[name],split=[split],leg=['FR'],time_abs=[0.])),fig/name/'contact_event_samples.csv')
            collect_event_tables(data,fig);collect_event_tables(data,fig)
            for split,name in [('train','a'),('test','b')]:
                f=pd.read_csv(data/split/'contact_event_features.csv')
                self.assertEqual(f.trial.tolist(),[name])

    def test_mixed_window_settings_do_not_overwrite_existing_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'train';fig=root/'figure'
            existing=data/'contact_event_features.csv'
            atomic_csv(pd.DataFrame({'marker':['keep']}),existing)
            before=existing.read_bytes()
            for name,after in [('a',40.),('b',80.)]:
                atomic_csv(pd.DataFrame({'time_abs':[0.]}),data/f'{name}.csv')
                atomic_csv(pd.DataFrame(dict(event_id=[name],trial=[name],split=['train'],leg=['FR'],window_after_ms=[after])),fig/name/'contact_event_features.csv')
            with self.assertRaisesRegex(ValueError,'mixed extraction settings'):collect_event_tables(data,fig)
            self.assertEqual(existing.read_bytes(),before)


if __name__=='__main__':unittest.main()
