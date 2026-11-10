"""Verify full-motor preservation, causal resampling, gap-safe windows,
held-out preprocessing and grouped permutations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"preprocessing"))
from common_time import create_common_time, interpolate_previous, recorded_commands
from preprocessing import iter_rows, csv_columns, write_csv
from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import build_windows, raw_features, select_from_windows, select_input_features
from noise_prediction_model.experiments.feature_selection.train import estimator, fit, tune
from noise_prediction_model.experiments.feature_selection.test import validate_test_split
from noise_prediction_model.experiments.feature_selection.model_evaluation import block_permutation, evaluate_saved_predictions, validate_predictions


class FakeTrial:
    name = "fake"
    path = Path("fake.npz")
    meta = {"vx": .5}

    def __init__(self):
        t = np.arange(.02, 1.9, .002)
        self.data = {"ls_t":t, "sp_t":t, "audio":np.sin(np.arange(4000)*.4)*1000,
                     "audio_slope":np.array(.0005), "audio_t0":np.array(0.),
                     "ev_label":np.array(["move_start","move_stop"]), "ev_t":np.array([.2,1.7]),
                     "ev_api":np.array([1008,1003])}
        for key,width in [("ff",4),("acc",3),("gyro",3),("rpy",3),("quaternion",4),("q",12),("dq",12),("tau",12)]:
            self.data[f"ls_{key}"] = np.tile(np.arange(width)+1., (len(t),1))
        self.data["ls_tick"] = np.arange(len(t))
        self.data["sp_vel"] = np.tile([.5,0.,0.],(len(t),1))
        self.data["sp_foot_vel"] = np.zeros((len(t),12))

    @property
    def keys(self): return list(self.data)
    def load(self,key): return self.data[key]


class TimeseriesTests(unittest.TestCase):
    def test_split_preprocessing_rejects_empty_or_overlapping_raw_trials(self):
        from preprocessing import process_split_dataset
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for split in ('train','test'):(root/split).mkdir()
            with self.assertRaisesRegex(ValueError,'disjoint'):
                process_split_dataset(SimpleNamespace(data_dir=root))
            for split in ('train','test'):(root/split/'0.3_1.npz').write_bytes(b'not needed: reject IDs before loading')
            with self.assertRaisesRegex(ValueError,'disjoint'):
                process_split_dataset(SimpleNamespace(data_dir=root))

    def test_preprocessing_contract_rejects_non_20ms_targets(self):
        from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import validate_preprocessing_metadata
        validate_preprocessing_metadata(dict(robot_interpolation='previous',audio_rms_window_s=.020))
        for window in [.001,None]:
            with self.assertRaisesRegex(ValueError,'20ms RMS'):
                validate_preprocessing_metadata(dict(robot_interpolation='previous',audio_rms_window_s=window))
        with self.assertRaisesRegex(ValueError,'previous-value'):
            validate_preprocessing_metadata(dict(robot_interpolation='linear',audio_rms_window_s=.020))

    def samples(self):
        t=np.arange(0,1,.002)
        return pd.DataFrame(dict(trial="t1",command_vx=.5,time_abs=t,time_from_move_start=t,
                                 segment=0,state=t,audio_dbfs_20ms=t*10,sp_vel_x=.5))

    def test_full_12_motors_and_tick_preserved(self):
        d=create_common_time(FakeTrial())
        for field in ["q","dq","tau"]:
            self.assertEqual(d[f"ls_{field}"].shape[1],12)
            np.testing.assert_allclose(d[f"ls_{field}"][d["valid"]][0],np.arange(12)+1.)
        self.assertIn("ls_tick",d)
        self.assertEqual(len([c for c in csv_columns() if c.startswith('ls_q_')]),12)

    def test_previous_never_reads_future_and_respects_dropout(self):
        a,valid=interpolate_previous([0.,.01,.5],np.array([1.,2.,999.]),np.array([.005,.015,.1,.5]),.02)
        np.testing.assert_allclose(a[[0,1,3]],[1.,2.,999.])
        self.assertFalse(valid[2]);self.assertTrue(np.isnan(a[2]))

    def test_walking_retains_nonsteady_rows(self):
        trial=FakeTrial();d=create_common_time(trial);d['audio_delay_s']=np.array(0.)
        n=len(d['time_abs'])
        state={'steady':np.zeros(n,dtype=bool),'move_mask':(d['time_abs']>=.2)&(d['time_abs']<=1.7),
               'sp_vel_x_smooth':np.full(n,.5),'sp_acc_x':np.zeros(n)}
        walking=list(iter_rows(trial,d,state,'walking'))
        self.assertGreater(len(walking),500)
        self.assertEqual(len(list(iter_rows(trial,d,state,'steady'))),0)
        self.assertEqual(len(walking[0]),len(csv_columns()))

    def test_lags_past_only_target_future(self):
        x,y,meta,names,_=build_windows(self.samples(),['state'],[0,20,100])
        np.testing.assert_allclose(x[:,0],meta.anchor_time,atol=1e-7)
        np.testing.assert_allclose(x[:,1],meta.anchor_time-.02,atol=1e-7)
        np.testing.assert_allclose(x[:,2],meta.anchor_time-.1,atol=1e-7)
        np.testing.assert_allclose(y,meta.target_time*10)

    def test_recorded_direction_overrides_legacy_speed_metadata_without_future_commands(self):
        trial = FakeTrial()
        trial.data['cmd_t'] = np.array([.2, .4, .6])
        trial.data['cmd_vel'] = np.array([[0., .3, 0.], [0., 0., -1.], [0., 0., 0.]])
        np.testing.assert_allclose(recorded_commands(trial, [.1, .2, .399, .4, .7]),
            [[0., 0., 0.], [0., .3, 0.], [0., .3, 0.], [0., 0., -1.], [0., 0., 0.]])

    def test_lateral_and_turning_commands_are_not_standing(self):
        for field in ('command_vy', 'command_vyaw'):
            data = self.samples()
            data['command_vx'] = 0.
            data['command_vy'] = 0.
            data['command_vyaw'] = 0.
            data.loc[data.time_abs >= .1, field] = -.3
            x, _, meta, _, coverage = build_windows(data, ['state'])
            self.assertGreater(len(x), 0)
            self.assertEqual(coverage.reason.iloc[0], 'included')
            np.testing.assert_allclose(meta[field], -.3)
        data[['command_vx', 'command_vy', 'command_vyaw']] = 0.
        with self.assertRaisesRegex(ValueError, 'No valid whole-sequence windows'):
            build_windows(data, ['state'])

    def test_windows_cannot_cross_gap(self):
        d=self.samples();d=d[~d.time_abs.between(.4,.5)]
        x,y,meta,_,_=build_windows(d,['state'],[0,100])
        self.assertFalse(meta.anchor_time.between(.4,.6).any())
        self.assertFalse(((meta.anchor_time<.4)&(meta.target_time>=.4)).any())

    def test_prediction_does_not_require_audio_or_future_robot(self):
        d=self.samples().drop(columns='audio_dbfs_20ms')
        x,y,meta,_,_=build_windows(d,['state'],[0,100],with_targets=False)
        self.assertTrue(np.isnan(y).all())
        self.assertGreater(meta.target_time.max(),d.time_abs.max())

    def test_permutation_stays_in_trial_and_segment(self):
        m=pd.DataFrame({'trial':['a']*20+['b']*20,'segment':[0]*10+[1]*10+[0]*20})
        order=block_permutation(m,3,np.random.default_rng(42))
        self.assertEqual(sorted(order.tolist()),list(range(40)))
        np.testing.assert_array_equal(m.trial,m.trial.iloc[order])
        np.testing.assert_array_equal(m.segment,m.segment.iloc[order])

    def test_scaler_uses_training_only(self):
        x=np.arange(10.).reshape(-1,1);y=2*x[:,0]
        model=fit(estimator('ridge',1),x,y,np.array(['a']*5+['b']*5))
        mean=model.named_steps['standardscaler'].mean_.copy()
        model.predict(np.array([[10000.]]))
        np.testing.assert_array_equal(mean,model.named_steps['standardscaler'].mean_)
        self.assertAlmostEqual(mean[0],4.5)

    def test_top_k_selects_training_correlated_feature(self):
        x=np.column_stack([np.arange(40.),np.tile([0.,1.],20)])
        meta=pd.DataFrame({'trial':['a']*20+['b']*20})
        selected=select_from_windows(x,x[:,0],meta,['ls_q_a__lag0ms','ls_q_b__lag0ms'],['ls_q_a','ls_q_b'],1)
        self.assertEqual(selected,['ls_q_a'])

    def test_test_split_rejects_training_or_unreserved_trials(self):
        cfg={'training_trials':['a','b'],'excluded_test_trials':['c']}
        validate_test_split(cfg,['c'])
        with self.assertRaises(ValueError): validate_test_split(cfg,['a'])
        with self.assertRaises(ValueError): validate_test_split(cfg,['d'])

    def test_feature_schema_rejects_audio_and_duplicates(self):
        with self.assertRaises(ValueError): select_input_features(['ls_q_a'],['audio_dbfs_20ms'])
        with self.assertRaises(ValueError): select_input_features(['ls_q_a'],['ls_q_a','ls_q_a'])

    def test_inner_tuning_supports_independent_feature_selection(self):
        x=np.column_stack([np.arange(40.), np.sin(np.arange(40.)), np.arange(40.)**2])
        y=x[:,0]; groups=np.repeat(['a','b','c','d'],10)
        inner=(np.arange(30),np.arange(30,40))
        model,param,score=tune('ridge',x,y,groups,inner,indices=[0,1],inner_indices=[2])
        self.assertEqual(model.n_features_in_,2)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(len(model.predict(x[:,[0,1]])),40)

    def test_evaluation_needs_raw_predictions_only_and_preserves_them(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            predictions=pd.DataFrame({'trial':['c']*4,'actual_dbfs':[-10.,-20.,-10.,-20.],
                                      'mean_baseline':[-15.]*4,'speed_lowstate_hist':[-10.,-20.,-10.,-20.]})
            path=folder/'test_predictions.csv';predictions.to_csv(path,index=False)
            original=path.read_bytes()
            config={'train_trials':['a'],'test_trials':['c'],'model_names':['mean_baseline','speed_lowstate_hist'],
                    'selected_model':'speed_lowstate_hist','horizon_ms':20,'loud_threshold_train_dbfs':-12.}
            (folder/'run.json').write_text(json.dumps(config))
            rows=evaluate_saved_predictions(folder,'holdout',no_plots=True)
            perfect=rows[rows.model=='speed_lowstate_hist'].iloc[0]
            self.assertEqual(perfect.rmse_db,0.)
            self.assertEqual(perfect.r2,1.)
            self.assertEqual(perfect.n_loud_windows,2)
            self.assertTrue((folder/'test_metrics.csv').exists())
            self.assertTrue((folder/'test_summary.csv').exists())
            self.assertEqual(original,path.read_bytes())

    def test_evaluation_rejects_nonfinite_predictions(self):
        predictions=pd.DataFrame({'trial':['c'],'actual_dbfs':[-20.],'model':[np.inf]})
        with self.assertRaises(ValueError): validate_predictions(predictions,['model'])

    def test_failed_csv_write_preserves_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'dataset.csv'
            write_csv(path,['value'],[[1]])
            original=path.read_bytes()
            def broken_rows():
                yield [2]
                raise RuntimeError('interrupted producer')
            with self.assertRaises(RuntimeError):
                write_csv(path,['value'],broken_rows())
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(directory).glob('*.tmp.*')),[])


if __name__=='__main__': unittest.main()
