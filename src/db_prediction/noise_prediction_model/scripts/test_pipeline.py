"""Causality, gap isolation, training isolation and GP posterior math tests."""
from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from noise_prediction_model.scripts import features as f
from noise_prediction_model.scripts import train_evaluate as t
from noise_prediction_model.scripts.predict import predict_frame
from noise_prediction_model.scripts.sparse_gp import SparseGaussianProcessRegressor
from noise_prediction_model.scripts.torch_models import TorchRidge, TorchHGB, save_bundle, load_bundle


def frame(n=220):
    times = 100+np.arange(n)*.002
    values = dict(time_abs=times, segment=0, audio_dbfs_20ms=-40+times-100)
    for axis in 'xyz':
        values[f'ls_gyro_{axis}'] = 0.
    for axis in 'wxyz':
        values[f'ls_quaternion_{axis}'] = 1. if axis == 'w' else 0.
    for leg in f.LEGS:
        values[f'ls_ff_{leg}'] = 20+3*(times-100)
        for joint in ('hip', 'thigh', 'calf'):
            values[f'ls_q_{leg}_{joint}'] = 0.
            values[f'ls_dq_{leg}_{joint}'] = .1
    return pd.DataFrame(values)


class PipelineTests(unittest.TestCase):
    def test_causal_features_ignore_future_and_audio(self):
        source = frame()
        arrays, segments = f.physical_arrays(source)
        x, y, meta, names, _ = f.causal_windows(source, arrays, segments, 'one', 'train')
        anchor = np.searchsorted(source.time_abs, meta.anchor_time.iloc[4]-1e-8)
        changed = source.copy()
        cols = [c for c in changed if c.startswith(('ls_ff_', 'ls_q_', 'ls_dq_'))]
        changed.loc[anchor+1:, cols] += 10
        changed['audio_dbfs_20ms'] += 15
        aa, ss = f.physical_arrays(changed)
        xx, yy, *_ = f.causal_windows(changed, aa, ss, 'one', 'train')
        np.testing.assert_allclose(x[:5], xx[:5])
        np.testing.assert_allclose(yy-y, 15.)
        without_audio = source.drop(columns='audio_dbfs_20ms')
        xa, _, ma, *_ = f.causal_windows(without_audio, arrays, segments, 'one', 'inference', with_targets=False)
        np.testing.assert_allclose(x, xa[:len(x)])
        self.assertEqual(len(names), 64)
        self.assertTrue((ma.target_support_start > ma.anchor_time).all())

    def test_explicit_dataset_keeps_review_trials_and_separates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'experiment'
            (source/'train').mkdir(parents=True)
            (source/'test').mkdir()
            (source/'split.json').write_text(json.dumps(dict(train_trials=['good','review'],test_trials=['heldout'])))
            for split, trial, accepted in [('train','good',True),('train','review',False),('test','heldout',False)]:
                data=frame()
                data['command_vx']=.4
                data['synchronization_passed']=accepted
                data['synchronization_status']='CONSISTENT_CANDIDATE' if accepted else 'CALIBRATION_REVIEW'
                data.to_csv(source/split/f'{trial}.csv',index=False)
            dataset, protocol=f.prepare(source,output)
            self.assertEqual(set(dataset['meta'].trial),{'good','review','heldout'})
            self.assertEqual(set(dataset['meta'].loc[dataset['meta'].trial.eq('review'),'sync_group']),{'sync_review'})
            self.assertEqual(dataset['x'].shape[1],64)
            self.assertTrue((output/'datasets/causal_forecast.npz').exists())
            self.assertEqual(protocol['source'],str(source.resolve()))
            self.assertFalse((source/'datasets').exists())

    def test_history_and_target_never_cross_gap(self):
        source = frame().drop(index=100).reset_index(drop=True)
        arrays, segments = f.physical_arrays(source)
        _, _, meta, *_ = f.causal_windows(source, arrays, segments, 'one', 'train')
        self.assertFalse(((meta.anchor_time-.046 < 100.2) & (meta.target_support_stop > 100.2)).any())
        self.assertTrue(np.isnan(arrays['force_change'][100:103]).all())

    def test_gp_matches_exact_gp_when_inducing_cover_training(self):
        x = torch.linspace(-2, 2, 12, dtype=torch.float64)[:, None]
        y = x[:, 0].sin()
        gp = SparseGaussianProcessRegressor(n_features=1, n_inducing=len(x), batch_size=5).double()
        gp.x_mean, gp.x_scale = x.mean(0), x.std(0, correction=0)
        xx = (x-gp.x_mean)/gp.x_scale
        gp.inducing = xx.clone()
        gp.lengthscale = torch.tensor([.8], dtype=torch.float64)
        gp.outputscale = torch.tensor(1.2, dtype=torch.float64)
        gp.noise_variance = torch.tensor(.05, dtype=torch.float64)
        gp.condition_on_all_training(xx, y)
        query = torch.linspace(-3, 3, 23, dtype=torch.float64)[:, None]
        k = gp.kernel((query-gp.x_mean)/gp.x_scale, xx)
        gram = gp.kernel(xx, xx)+.05*torch.eye(len(x), dtype=x.dtype)
        expected_mean = k@torch.linalg.solve(gram, y)
        expected_var = 1.2-(k*torch.linalg.solve(gram, k.T).T).sum(1)
        mean, latent, observation = gp.predict_distribution(query)
        torch.testing.assert_close(mean, expected_mean, atol=.002, rtol=0)
        torch.testing.assert_close(latent**2, expected_var, atol=.002, rtol=0)
        torch.testing.assert_close(observation**2-latent**2, torch.full_like(latent, .05))
        np.testing.assert_allclose(gp.predict(query), mean, atol=1e-12)
        # Native checkpoint inference is float32. Test serialization in that dtype.
        gp = gp.float()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_bundle(path, gp, dict(model_family='03_sparse_gpr'))
            restored = load_bundle(path)['model']
            for batch in (0, 1, 7, 23):
                value = query[:batch].float()
                torch.testing.assert_close(restored(value), gp(value))
                for actual, expected in zip(restored.predict_distribution(value), gp.predict_distribution(value)):
                    torch.testing.assert_close(actual, expected)
            value = torch.tensor([[.25]], requires_grad=True)
            restored(value).sum().backward()
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_ridge_torch_solve_and_gradients(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(40, 3))
        y = -30+2*x[:, 0]-x[:, 1]
        model = TorchRidge(alpha=3.).fit(x, y)
        mean, scale = x.mean(0), x.std(0)
        xx = (x-mean)/scale
        weight = np.linalg.solve(xx.T@xx+3*np.eye(3), xx.T@(y-y.mean()))
        expected = y.mean()+xx@weight
        np.testing.assert_allclose(model.predict(x), expected, atol=1e-5)
        value = torch.tensor(x[:2], dtype=torch.float32, requires_grad=True)
        model(value).sum().backward()
        torch.testing.assert_close(value.grad, model.coefficient.expand(2, -1))

    def test_hgb_threshold_missing_routes_and_checkpoint(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(9)
        x = rng.normal(size=(257, 3))
        y = -40+3*(x[:, 0] > .2)-2*x[:, 1]
        reference = HistGradientBoostingRegressor(max_iter=13, max_leaf_nodes=7,
                    min_samples_leaf=3, learning_rate=.13, early_stopping=False, random_state=9).fit(x, y)
        model = TorchHGB.from_estimator(reference)
        probes = [x]
        for stage in reference._predictors:
            for node in stage[0].nodes:
                if not node['is_leaf']:
                    # Include exact thresholds and adjacent representable doubles.
                    row = np.repeat(x[:1], 4, axis=0)
                    threshold = node['num_threshold']
                    row[:, node['feature_idx']] = [np.nextafter(threshold, -np.inf), threshold,
                                                  np.nextafter(threshold, np.inf), np.nan]
                    probes.append(row)
        probes = np.concatenate(probes)
        with torch.inference_mode():
            np.testing.assert_allclose(model(torch.from_numpy(probes)), reference.predict(probes), atol=1e-10, rtol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_bundle(path, model, dict(model_family='02_hgb'))
            restored = load_bundle(path)['model']
            self.assertEqual(restored.threshold.dtype, torch.float64)
            self.assertEqual(restored.feature.dtype, torch.int64)
            for batch in (0, 1, 17, len(probes)):
                value = torch.from_numpy(probes[:batch])
                torch.testing.assert_close(restored(value), model(value))
            # The loaded inference model has no sklearn reference object.
            self.assertFalse(hasattr(restored, 'reference_estimator_'))
        fitted = TorchHGB(max_iter=5, min_samples_leaf=3, early_stopping=False).fit(x, y)
        self.assertLess(fitted.check_reference(x), 1e-10)

    def test_hgb_constant_leaf_trees(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        x = np.ones((40, 3))
        y = np.full(40, -35.)
        reference = HistGradientBoostingRegressor(max_iter=3, early_stopping=False).fit(x, y)
        model = TorchHGB.from_estimator(reference)
        self.assertEqual(model.max_depth, 0)
        np.testing.assert_allclose(model.predict(x), reference.predict(x), atol=1e-12)

    def test_gp_train_and_distribution_are_torch(self):
        x = np.linspace(-2, 2, 24)[:, None]
        y = np.sin(x[:, 0])
        gp = SparseGaussianProcessRegressor(n_inducing=6, epochs=2, batch_size=8).fit(x, y)
        result = gp.predict_distribution(torch.tensor(x, dtype=torch.float32))
        self.assertTrue(all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in result))
        self.assertTrue((result[2] >= result[1]).all())

    def test_selection_fits_only_fold_training_rows(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 3))
        y = 2*x[:, 0]-x[:, 1]
        meta = pd.DataFrame(dict(trial=np.repeat([f'train{i}' for i in range(6)], 10)))
        folds = list(t.GroupKFold(n_splits=3).split(x, y, meta.trial))
        original = t.make_model
        fitted_sizes = []
        def factory(key, parameters):
            model = original(key, parameters)
            original_fit = model.fit
            def fit(xx, yy):
                self.assertLess(abs(yy).max(), 100)
                fitted_sizes.append(len(yy))
                return original_fit(xx, yy)
            model.fit = fit
            return model
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            with patch.object(t, 'candidates', return_value=[dict(alpha=1.)]), patch.object(t, 'make_model', side_effect=factory):
                bundle, (oof, _, _) = t.select_and_fit('01_ridge_regression', 'Ridge', x, y, meta, folds,
                    np.array(['a', 'b', 'c']), 1, Path(directory))
            self.assertEqual(fitted_sizes, [40, 40, 40, 60])
            self.assertEqual(set(bundle['fit_trials']), set(meta.trial))
            self.assertTrue(np.isfinite(oof).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
