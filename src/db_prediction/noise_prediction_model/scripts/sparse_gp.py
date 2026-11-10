"""PyTorch sparse variational GP: training, all-train conditioning and inference.

Matérn-3/2 ARD kernel, Gaussian likelihood, minibatch ELBO optimization followed
by the optimal Gaussian variational posterior at fixed kernel/inducing points.
"""
import math
import time
import torch
from torch.nn import Parameter
from noise_prediction_model.scripts.torch_models import TensorRegressor


def matern(a, b, lengthscale, outputscale):
    r = math.sqrt(3.)*torch.cdist(a/lengthscale, b/lengthscale,
                                compute_mode='donot_use_mm_for_euclid_dist')
    return outputscale*(1+r)*torch.exp(-r)


class SparseGaussianProcessRegressor(TensorRegressor):
    def __init__(self, n_inducing=128, epochs=30, batch_size=2048,
                 learning_rate=.025, random_state=20261027, n_features=64):
        super().__init__()
        self.n_inducing, self.epochs, self.batch_size = n_inducing, epochs, batch_size
        self.learning_rate, self.random_state, self.n_features = learning_rate, random_state, n_features
        values = dict(x_mean=torch.zeros(n_features), x_scale=torch.ones(n_features),
                      inducing=torch.zeros(n_inducing, n_features), lengthscale=torch.ones(n_features),
                      alpha=torch.zeros(n_inducing), variance_matrix=torch.zeros(n_inducing, n_inducing),
                      outputscale=torch.tensor(1.), noise_variance=torch.tensor(.15),
                      y_mean=torch.tensor(0.), y_scale=torch.tensor(1.))
        for name, value in values.items():
            self.register_buffer(name, value)

    def kernel(self, a, b):
        return matern(a, b, self.lengthscale, self.outputscale)

    def initial_centers(self, x, m, generator):
        # Torch-only, bounded-memory minibatch k-means initialization.
        centers = x[torch.randperm(len(x), generator=generator)[:m]].clone()
        for _ in range(20):
            batch = x[torch.randperm(len(x), generator=generator)[:min(4096, len(x))]]
            labels = torch.cdist(batch, centers).argmin(1)
            totals = torch.zeros_like(centers).index_add_(0, labels, batch)
            counts = torch.bincount(labels, minlength=m)
            active = counts > 0
            centers[active] = totals[active]/counts[active, None]
        return centers

    def fit(self, x, y):
        torch.manual_seed(self.random_state)
        started = time.perf_counter()
        tx, ty = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        if tx.ndim != 2 or ty.ndim != 1 or len(tx) != len(ty) or not torch.isfinite(tx).all() or not torch.isfinite(ty).all():
            raise ValueError('Need finite X and y of matching lengths')
        self.n_features = tx.shape[1]
        self.x_mean = tx.mean(0)
        scale = tx.std(0, correction=0)
        self.x_scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        self.y_mean, self.y_scale = ty.mean(), ty.std(correction=0).clamp_min(1e-8)
        xx, yy = (tx-self.x_mean)/self.x_scale, (ty-self.y_mean)/self.y_scale
        n, d = xx.shape
        m = min(self.n_inducing, n)
        generator = torch.Generator().manual_seed(self.random_state)
        z = Parameter(self.initial_centers(xx, m, generator))
        log_length = Parameter(torch.full((d,), math.log(math.sqrt(d))))
        log_output, log_noise = Parameter(torch.tensor(0.)), Parameter(torch.tensor(math.log(.15)))
        q_mean, q_raw = Parameter(torch.zeros(m)), Parameter(torch.zeros(m, m))
        eye = torch.eye(m)
        parameters = [log_length, log_output, log_noise, q_mean, q_raw]
        optimizer = torch.optim.Adam([dict(params=parameters, lr=self.learning_rate), dict(params=[z], lr=.005)])

        def kernel(a, b):
            # Squared-distance form supports training through identical points.
            aa, bb = a/log_length.exp(), b/log_length.exp()
            squared = (aa.square().sum(1)[:, None]+bb.square().sum(1)[None, :]-2*aa@bb.T).clamp_min(1e-10)
            r = (3*squared).sqrt()
            return log_output.exp()*(1+r)*(-r).exp()

        self.history_ = []
        for epoch in range(self.epochs):
            objective = 0.
            for ids in torch.randperm(n, generator=generator).split(self.batch_size):
                optimizer.zero_grad()
                lzz = torch.linalg.cholesky(kernel(z, z)+1e-4*eye)
                a = torch.linalg.solve_triangular(lzz, kernel(z, xx[ids]), upper=False)
                diagonal = q_raw.diagonal().clamp(-7, 4).exp()
                lq = torch.tril(q_raw, diagonal=-1)+torch.diag(diagonal)
                mean = a.T@q_mean
                variance = (log_output.exp()-a.square().sum(0)+(lq.T@a).square().sum(0)).clamp_min(1e-7)
                ell = -.5*(math.log(2*math.pi)+log_noise+((yy[ids]-mean).square()+variance)/log_noise.exp()).mean()
                kl = .5*(lq.square().sum()+q_mean.square().sum()-m-2*diagonal.log().sum())
                loss = -ell+kl/n
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite sparse-GP ELBO')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters+[z], 20.)
                optimizer.step()
                with torch.no_grad():
                    log_length.clamp_(-3, 5)
                    log_output.clamp_(-5, 5)
                    log_noise.clamp_(-7, 2)
                objective += float(loss.detach())*len(ids)
            self.history_.append(dict(epoch=epoch+1, negative_elbo_per_sample=objective/n,
                                      noise_variance_standardized=float(log_noise.detach().exp())))
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'  Sparse GP M={m}, epoch={epoch+1}/{self.epochs}, ELBO loss={objective/n:.4f}', flush=True)
        self.inducing = z.detach().clone()
        self.lengthscale = log_length.detach().exp()
        self.outputscale = log_output.detach().exp()
        self.noise_variance = log_noise.detach().exp()
        self.condition_on_all_training(xx, yy)
        self.n_training_samples_ = n
        self.fit_seconds_ = time.perf_counter()-started
        return self.eval()

    @torch.no_grad()
    def condition_on_all_training(self, xx, yy):
        # Solve in float64 for posterior stability, then deploy in float32.
        dtype = self.inducing.dtype
        z, length = self.inducing.double(), self.lengthscale.double()
        output, noise = self.outputscale.double(), self.noise_variance.double()
        xx, yy = torch.as_tensor(xx, dtype=torch.float64), torch.as_tensor(yy, dtype=torch.float64)
        m = len(z)
        eye = torch.eye(m, dtype=torch.float64)
        lzz = torch.linalg.cholesky(matern(z, z, length, output)+1e-4*eye)
        gram, rhs = torch.zeros_like(eye), torch.zeros(m, dtype=torch.float64)
        for start in range(0, len(xx), self.batch_size):
            k = matern(z, xx[start:start+self.batch_size], length, output)
            a = torch.linalg.solve_triangular(lzz, k, upper=False)
            gram += a@a.T
            rhs += a@yy[start:start+self.batch_size]
        factor = torch.linalg.cholesky(eye+gram/noise)
        covariance = torch.cholesky_solve(eye, factor)
        mean = torch.cholesky_solve((rhs/noise)[:, None], factor)[:, 0]
        self.alpha = torch.linalg.solve_triangular(lzz.T, mean[:, None], upper=True)[:, 0].to(dtype)
        left = torch.linalg.solve_triangular(lzz.T, covariance-eye, upper=True)
        matrix = torch.linalg.solve_triangular(lzz.T, left.T, upper=True).T
        self.variance_matrix = ((matrix+matrix.T)/2).to(dtype)

    def forward(self, x):
        k = self.kernel((x-self.x_mean)/self.x_scale, self.inducing)
        return self.y_mean+self.y_scale*(k@self.alpha)

    def predict_distribution(self, x):
        k = self.kernel((x-self.x_mean)/self.x_scale, self.inducing)
        mean = self.y_mean+self.y_scale*(k@self.alpha)
        variance = (self.outputscale+((k@self.variance_matrix)*k).sum(1)).clamp_min(0.)
        return mean, self.y_scale*variance.sqrt(), self.y_scale*(variance+self.noise_variance).sqrt()
