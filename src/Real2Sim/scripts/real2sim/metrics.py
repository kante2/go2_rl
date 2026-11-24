"""Evaluate physical and policy-coordinate arrays once in the analysis pipeline."""
import numpy as np

def metric(real, sim, valid):
    a, b = np.asarray(real)[valid], np.asarray(sim)[valid]
    if not len(a): return dict(n=0, rmse=None, mae=None, bias=None, correlation=None)
    error = b - a
    r = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-10 else None
    return dict(n=len(a), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(abs(error))), bias=float(error.mean()), correlation=r)

def rolling_rmse(real, sim, valid, samples=100):
    """Trailing window; invalid samples do not enter either sum or count."""
    error = np.asarray(sim) - np.asarray(real)
    good = np.asarray(valid, bool) & np.isfinite(error).all(axis=1)
    sums = np.vstack((np.zeros((1, error.shape[1])), np.cumsum(np.where(good[:, None], error ** 2, 0.), axis=0)))
    counts = np.r_[0, np.cumsum(good)]
    end = np.arange(len(error)) + 1; start = np.maximum(0, end - samples)
    count = counts[end] - counts[start]
    mean_squared = np.full_like(error, np.nan, dtype=float)
    np.divide(sums[end] - sums[start], count[:, None], out=mean_squared, where=count[:, None] > 0)
    return np.sqrt(mean_squared), np.sqrt(np.mean(mean_squared, axis=1))

def evaluate_view(data, kind, source, mask, ddq_mask=None, rolling_mask=None, force_reference=None, rolling_arrays=None):
    result = dict(kind=kind, source=source, metrics={}, errors=dict(windowSeconds=.5))
    for quantity in ('q', 'dq', 'ddq', 'tau'):
        a, b = data['real_' + quantity], data['sim_' + quantity]
        valid = mask if quantity != 'ddq' or ddq_mask is None else ddq_mask
        result['metrics'][quantity] = [metric(a[:, j], b[:, j], valid) for j in range(12)]
        rolling_valid = valid if rolling_mask is None else rolling_mask
        per_joint, combined = rolling_rmse(a, b, rolling_valid, samples=100)
        if rolling_arrays is not None:
            rolling_arrays[f'{kind}_{quantity}_rmse_joint'] = per_joint
            rolling_arrays[f'{kind}_{quantity}_rmse_combined'] = combined
    a, b = data['sensor_counts'], data['sim_fz_N']
    reference = force_reference or dict(
        real_mean=a[mask].mean(axis=0).tolist(), real_std=a[mask].std(axis=0).tolist(),
        sim_mean=b[mask].mean(axis=0).tolist(), sim_std=b[mask].std(axis=0).tolist())
    force_shape_error = []
    for foot in range(4):
        x, y = a[mask, foot], b[mask, foot]
        if min(reference['real_std'][foot], reference['sim_std'][foot]) > 1e-10:
            x = (x - reference['real_mean'][foot]) / reference['real_std'][foot]
            y = (y - reference['sim_mean'][foot]) / reference['sim_std'][foot]
            force_shape_error.append(float(np.sqrt(np.mean((x - y) ** 2))))
        else:
            force_shape_error.append(None)
    result['force'] = dict(zscoreRmse=force_shape_error,
        correlation=[metric(a[:, j], b[:, j], mask)['correlation'] for j in range(4)], **reference)
    result['metricScope'] = 'same recording-interior evaluation samples before/after; same filter and derivative'
    return result
