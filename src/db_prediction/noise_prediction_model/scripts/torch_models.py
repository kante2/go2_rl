"""Native PyTorch regressors, CPU/NumPy adapters and state-dict persistence.

Call model(features) directly for resident CPU/CUDA tensors. The predict methods
are conveniences for CSV evaluation and existing NumPy streaming consumers.
"""
from pathlib import Path
import torch
from torch import nn


class TensorRegressor(nn.Module):
    def input_tensor(self, x):
        reference = next(self.buffers())
        tensor = torch.as_tensor(x, dtype=reference.dtype, device=reference.device)
        if tensor.ndim != 2 or tensor.shape[1] != self.n_features or not torch.isfinite(tensor).all():
            raise ValueError('Expected finite [batch, n_features] inputs')
        return tensor

    def predict(self, x, return_std=False):
        with torch.inference_mode():
            tensor = self.input_tensor(x)
            if return_std:
                mean, _, observation = self.predict_distribution(tensor)
                return mean.cpu().numpy(), observation.cpu().numpy()
            return self(tensor).cpu().numpy()

    def predict_distribution_numpy(self, x):
        with torch.inference_mode():
            return tuple(value.cpu().numpy() for value in self.predict_distribution(self.input_tensor(x)))


class TorchRidge(TensorRegressor):
    """Closed-form L2 regression fitted with torch.linalg in float64."""
    def __init__(self, coefficient=None, intercept=0., alpha=1., n_features=64):
        super().__init__()
        self.alpha = alpha
        coefficient = torch.zeros(n_features) if coefficient is None else torch.as_tensor(coefficient)
        self.n_features = len(coefficient)
        self.register_buffer('coefficient', coefficient.clone())
        self.register_buffer('intercept', torch.as_tensor(intercept, dtype=coefficient.dtype).clone())

    def fit(self, x, y):
        tx, ty = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
        mean = tx.mean(0)
        scale = tx.std(0, correction=0)
        scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        xx = (tx-mean)/scale
        centered = ty-ty.mean()
        weight = torch.linalg.solve(xx.T@xx + self.alpha*torch.eye(xx.shape[1], dtype=tx.dtype), xx.T@centered)
        coefficient = weight/scale
        self.n_features = tx.shape[1]
        self.coefficient = coefficient.float()
        self.intercept = (ty.mean()-coefficient@mean).float()
        return self

    def forward(self, x):
        return x@self.coefficient+self.intercept


class TorchHGB(TensorRegressor):
    """Histogram boosting trained by sklearn; exact numeric trees run in torch.

    All trees traverse together for every input batch. Thresholds and leaf
    values retain float64 precision; node indices are integer buffers.
    """
    def __init__(self, n_features=64, n_trees=1, n_nodes=1, max_depth=0, **parameters):
        super().__init__()
        self.n_features, self.max_depth = n_features, max_depth
        self.parameters_config = parameters
        self.register_buffer('baseline', torch.tensor(0., dtype=torch.float64))
        for name in ('feature', 'left', 'right'):
            self.register_buffer(name, torch.zeros(n_trees, n_nodes, dtype=torch.long))
        for name in ('threshold', 'value'):
            self.register_buffer(name, torch.zeros(n_trees, n_nodes, dtype=torch.float64))
        self.register_buffer('missing_left', torch.zeros(n_trees, n_nodes, dtype=torch.bool))

    @classmethod
    def from_estimator(cls, estimator):
        import numpy as np
        if estimator.loss != 'squared_error' or any(len(stage) != 1 for stage in estimator._predictors):
            raise ValueError('Only scalar squared-error HGB regression is supported')
        trees = [stage[0].nodes for stage in estimator._predictors]
        if any(tree['is_categorical'].any() for tree in trees):
            raise ValueError('Categorical HGB splits are unsupported')
        model = cls(n_features=estimator.n_features_in_, n_trees=len(trees),
                    n_nodes=max(map(len, trees)), max_depth=max(int(tree['depth'].max()) for tree in trees))
        model.baseline.fill_(float(estimator._baseline_prediction[0, 0]))
        for index, tree in enumerate(trees):
            n = len(tree)
            leaves = tree['is_leaf'].astype(bool)
            for name, field in [('feature', 'feature_idx'), ('threshold', 'num_threshold'),
                                ('value', 'value'), ('missing_left', 'missing_go_to_left')]:
                getattr(model, name)[index, :n] = torch.as_tensor(tree[field].copy())
            # Absorb leaf nodes, so shallower trees stay at their leaves.
            for name in ('left', 'right'):
                child = np.where(leaves, np.arange(n), tree[name])
                getattr(model, name)[index, :n] = torch.as_tensor(child.copy())
        return model.eval()

    def fit(self, x, y):
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
        reference = HistGradientBoostingRegressor(**self.parameters_config).fit(x, y)
        converted = self.from_estimator(reference)
        self.n_features, self.max_depth = converted.n_features, converted.max_depth
        for name, value in converted.named_buffers():
            setattr(self, name, value)
        # This transient training reference is never written to model.pt.
        self.reference_estimator_ = reference
        self.history_ = [dict(iteration=i+1, train_rmse_db=float(np.sqrt(np.mean((prediction-y)**2))))
                         for i, prediction in enumerate(reference.staged_predict(x))]
        self.train_conversion_max_abs_error_db_ = self.check_reference(x)
        return self.eval()

    def check_reference(self, x):
        import numpy as np
        expected = self.reference_estimator_.predict(x)
        actual = self.predict(x)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        return float(np.max(np.abs(actual-expected))) if len(actual) else 0.

    def predict(self, x, return_std=False):
        if return_std:
            raise ValueError('HGB does not provide predictive standard deviations')
        with torch.inference_mode():
            tensor = self.input_tensor(x)
            # Bound CPU evaluation memory; direct tensor callers set their batch.
            return torch.cat([self(batch) for batch in tensor.split(2048)]).cpu().numpy()

    def forward(self, x):
        # Rows are trees; columns are batch elements. No per-sample Python loop.
        # Convert once and gather directly into tree-major storage. Transposing
        # each gathered batch creates strided intermediates at every depth.
        transposed = x.to(self.threshold.dtype).T
        nodes = torch.zeros((self.feature.shape[0], x.shape[0]), dtype=torch.long, device=x.device)
        for _ in range(self.max_depth):
            features = self.feature.gather(1, nodes)
            samples = transposed.gather(0, features)
            thresholds = self.threshold.gather(1, nodes)
            go_left = torch.where(torch.isnan(samples), self.missing_left.gather(1, nodes), samples <= thresholds)
            nodes = torch.where(go_left, self.left.gather(1, nodes), self.right.gather(1, nodes))
        return self.baseline+self.value.gather(1, nodes).sum(0)


def save_bundle(path, model, metadata):
    """Save plain metadata and tensors, readable with weights_only=True."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, TorchHGB):
        metadata = dict(metadata, tree_max_depth=model.max_depth, n_features=model.n_features)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(dict(metadata, backend='pytorch', format_version=1, state_dict=state), path)


def load_bundle(path, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('backend') != 'pytorch' or bundle.get('format_version') != 1:
        raise ValueError('Expected a native PyTorch noise model checkpoint')
    state = bundle.pop('state_dict')
    family = bundle['model_family']
    if family == '01_ridge_regression':
        model = TorchRidge(n_features=len(state['coefficient']))
    elif family == '02_hgb':
        model = TorchHGB(n_features=bundle['n_features'], n_trees=state['feature'].shape[0],
                         n_nodes=state['feature'].shape[1], max_depth=bundle['tree_max_depth'])
    elif family == '03_sparse_gpr':
        from noise_prediction_model.scripts.sparse_gp import SparseGaussianProcessRegressor
        model = SparseGaussianProcessRegressor(n_features=len(state['x_mean']), n_inducing=len(state['alpha']))
    else:
        raise ValueError(f'Unknown model family: {family}')
    model.load_state_dict(state)
    bundle['model'] = model.to(device).eval().requires_grad_(False)
    return bundle


def load_model(path, device='cpu'):
    """Load once, then call model(float32_features) for batch tensor inference."""
    return load_bundle(path, device)['model']
