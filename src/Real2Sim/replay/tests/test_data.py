"""Numeric data contract for the local replay UI."""
from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_data import metric, rolling_rmse, BASE


def test_initial_state_can_be_excluded_from_metrics():
    a = np.array([0., 0., 0.]); b = np.array([100., 1., 1.])
    result = metric(a, b, np.array([False, True, True]))
    assert result['rmse'] == result['mae'] == result['bias'] == 1.
    assert result['n'] == 2
    assert result['correlation'] is None


def test_rolling_rmse_excludes_invalid_initialization_and_uses_past_only():
    real = np.zeros((5, 2)); sim = np.array([[100, 100], [1, 3], [3, 1], [5, 5], [7, 7]])
    error, combined = rolling_rmse(real, sim, [False, True, True, True, True], samples=2)
    assert np.isnan(error[0]).all()
    np.testing.assert_allclose(error[1], [1, 3])
    np.testing.assert_allclose(error[2], np.sqrt([5, 5]))
    np.testing.assert_allclose(combined[1:3], np.sqrt([5, 5]))
    changed = sim.copy(); changed[4] = 999
    future, _ = rolling_rmse(real, changed, [False, True, True, True, True], samples=2)
    np.testing.assert_allclose(error[:4], future[:4], equal_nan=True)


def test_exported_mapping_keeps_sources_and_same_evaluation_samples():
    from build_data import PROJECT, sha
    directory = BASE / 'web/data'
    manifest = json.loads((directory / 'index.json').read_text())
    assert manifest['schema'] == 'go2-actor-mapping-v4'
    assert manifest['dataset'] == '1116_RL_base'
    assert [e['id'] for e in manifest['trials']] == [f'0.2_{i}' for i in range(1,6)]
    assert manifest['units']['realForce'] == 'counts'
    assert manifest['units']['simForce'] == 'N'
    for entry in manifest['trials']:
        trial = json.loads((directory / entry['file']).read_text())
        assert entry['realSha256'] == sha(Path(entry['realPath']))
        assert entry['simSha256'] == sha(Path(entry['simPath']))
        before, after = trial['before'], trial['after']
        assert before['kind'] == 'before' and after['kind'] == 'after'
        assert before['t'] == after['t']
        assert before['evaluationMask'] == after['evaluationMask']
        assert before['real']['ddq'] == after['real']['ddq']
        assert before['real']['q'] != after['real']['q']
        assert before['units']['q'] == 'rad' and after['units']['q'] == '무차원'
        assert before['force']['real_counts'] == after['force']['real_counts']
        assert before['force']['sim_N'] == after['force']['sim_N']
        assert before['force']['zscoreRmse'] == after['force']['zscoreRmse']
        assert before['sim']['ddq'] == after['sim']['ddq']
        assert before['sim']['q'] != after['sim']['q']
        assert trial['mapping']['time_alignment_in_both_views']
        assert trial['mapping']['deployment']['policy_inference_performed'] is False
        d=trial['mapping']['deployment']
        for q,left in (('q',11),('dq',23)):
            denom=np.empty(12);denom[d['policy_to_sdk']]=d['normalization_denominator'][left:left+12]
            np.testing.assert_allclose([m['rmse'] for m in after['metrics'][q]], np.asarray([m['rmse'] for m in before['metrics'][q]])/denom, atol=1e-12)
        for key in ('real_mean','real_std','sim_mean','sim_std'):
            assert before['force'][key] == after['force'][key]
        assert trial['mapping']['simulation_executed'] is False
        assert trial['mapping']['action_scale_applied'] is False
        for view in (before, after):
            assert len(view['metrics']['q']) == 12
            assert len(view['errors']['quantities']['q']['joint']) == len(view['t'])
            assert np.shape(view['force']['sim_N']) == (len(view['t']),4)
            assert view['t'][-1]-view['t'][0] > 24
            assert all(m['n'] == sum(view['evaluationMask']) for m in view['metrics']['q'])


def test_mapping_is_interpolation_of_existing_simulation_not_a_new_trajectory():
    from build_data import PROJECT, load, map_pair, prepare_pair, named_mapping
    raw, rm = load(PROJECT / 'data/real/1116_RL_base/0.2_3.npz')
    sim, sm = load(PROJECT / 'data/sim/1116_RL_base/0.2_3.npz')
    original = sim['ls_q'].copy()
    before, after, _, sync, _, contract, coordinates = map_pair(raw, rm, sim, sm)
    r, s, _ = named_mapping(raw, rm, sim, sm)
    source, _ = prepare_pair(r, s, 15., .005)
    query = after['t'] + sync['applied_lag_s']
    for key in ('sim_ddq','sim_foot_fz_N'):
        expected = np.column_stack([np.interp(query, source['t'], source[key][:,j]) for j in range(source[key].shape[1])])
        np.testing.assert_allclose(after[key],expected)
    np.testing.assert_array_equal(sim['ls_q'],original)
    for domain in ('real','sim'):
        d=contract['deployment'];order=d['policy_to_sdk'];mean=np.asarray(d['normalization_mean']);den=np.asarray(d['normalization_denominator'])
        expected=(before[domain+'_q'][:,order]-d['default_joint_pos']-mean[11:23])/den[11:23]
        np.testing.assert_allclose(after[domain+'_q'][:,order],expected)
        np.testing.assert_allclose(after[domain+'_dq'][:,order],(before[domain+'_dq'][:,order]-mean[23:35])/den[23:35])
        assert not np.allclose(before[domain+'_q'],after[domain+'_q'])
    assert contract['time_alignment_in_both_views']
    assert contract['deployment']['joint_observation_parity_max_error'] < 1e-6
    assert contract['deployment']['real_target_roundtrip_max_error_rad'] < 1e-14
    d=contract['deployment']; order=d['policy_to_sdk']
    np.testing.assert_allclose(coordinates['real_q_policy_relative'],before['real_q'][:,order]-d['default_joint_pos'])
    np.testing.assert_allclose(coordinates['real_logged_target_action_coordinate'],(raw['lc_q'][:,order]-d['default_joint_pos'])/d['action_scale'])
    np.testing.assert_array_equal(before['sim_foot_fz_N'],after['sim_foot_fz_N'])


def test_named_mapping_handles_reordered_simulation_channels():
    from build_data import named_mapping, JOINTS, LEGS
    ji = list(reversed(range(12))); fi = [2,3,0,1]
    sim = {key: np.arange(24).reshape(2,12)[:,ji] for key in ('ls_q','ls_dq','ls_tau')}
    sim['sim_foot_fz_N'] = np.arange(8).reshape(2,4)[:,fi]
    sm = dict(joint_order=[JOINTS[j] for j in ji],foot_order=[LEGS[j] for j in fi],force_semantics='World Fz')
    _, mapped, _ = named_mapping({}, {'joint_order':'SDK: FR, FL, RR, RL; hip, thigh, calf per leg'}, sim, sm)
    np.testing.assert_array_equal(mapped['ls_q'],np.arange(24).reshape(2,12))
    np.testing.assert_array_equal(mapped['sim_foot_fz_N'],np.arange(8).reshape(2,4))


def test_every_before_signal_uses_the_same_time_grid():
    directory = BASE / 'web/data'
    for path in directory.glob('0.2_*.json'):
        data = json.loads(path.read_text())['before']
        assert np.all(np.diff(data['t']) > 0)
        for domain in ('real', 'sim'):
            for quantity in ('q', 'dq', 'ddq', 'tau'):
                assert np.shape(data[domain][quantity]) == (len(data['t']), 12)
                assert np.isfinite(data[domain][quantity]).all()


def test_dataset_selection_ignores_other_speeds_and_sessions(tmp_path):
    from build_data import paired_paths, TRIALS
    import pytest
    for domain in ('real','sim'):
        folder=tmp_path/domain/'1116_RL_base'
        folder.mkdir(parents=True)
        for name in (*TRIALS,'0.2_6','0.4_1','1.0_1'):
            (folder/f'{name}.npz').touch()
        (tmp_path/domain/'0.2_1.npz').touch()
        other=tmp_path/domain/'another_session';other.mkdir()
        (other/'0.2_1.npz').touch()
    pairs=paired_paths(tmp_path)
    assert [r.stem for r,s in pairs] == list(TRIALS)
    assert all(r.parent.name == s.parent.name == '1116_RL_base' for r,s in pairs)
    (tmp_path/'sim/1116_RL_base/0.2_3.npz').unlink()
    with pytest.raises(FileNotFoundError,match='sim/1116_RL_base/0.2_3.npz'):
        paired_paths(tmp_path)


def test_torque_uses_logged_values_and_same_alignment_without_action_scaling():
    from build_data import PROJECT, load, map_pair, prepare_pair
    folder=PROJECT/'data'
    raw,rm=load(folder/'real/1116_RL_base/0.2_1.npz')
    sim,sm=load(folder/'sim/1116_RL_base/0.2_1.npz')
    baseline,mapped,_,sync,_,contract,_=map_pair(raw,rm,sim,sm)
    original,_=prepare_pair(raw,sim,15.,.005)
    for j in range(12):
        np.testing.assert_allclose(baseline['real_tau'][:,j],np.interp(baseline['t'],original['t'],original['real_tau'][:,j]))
        np.testing.assert_allclose(baseline['sim_tau'][:,j],np.interp(baseline['t']+sync['applied_lag_s'],original['t'],original['sim_tau'][:,j]))
    np.testing.assert_array_equal(mapped['real_tau'],baseline['real_tau'])
    np.testing.assert_array_equal(mapped['sim_tau'],baseline['sim_tau'])
    assert contract['deployment']['action_scale_applied_to_measured_q'] is False
