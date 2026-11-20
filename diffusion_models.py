"""
Stochastic Differential Equations (SDE) and Score-Based Diffusion Models from Scratch.
Implementation of Forward Ornstein-Uhlenbeck (OU) and Reverse-Time Backward Diffusion
processes in 1D, 2D, and high-dimensional spaces (e.g., Fashion-MNIST).

Author: Jessim El Asmar
Context: Master 1 Applied Mathematics & Statistics (TER), Universite Claude Bernard Lyon 1
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.neighbors import KNeighborsRegressor


def forward_ou_diffusion(x0: np.ndarray, T: float = 5.0, steps: int = 500, return_trajectory: bool = False):
    """
    Simulate the forward Ornstein-Uhlenbeck (OU) SDE via Euler-Maruyama discretization:
        dX_t = -X_t dt + sqrt(2) dW_t

    Parameters
    ----------
    x0 : np.ndarray of shape (N,) or (N, d)
        Initial data samples at t = 0.
    T : float
        Terminal diffusion time (stationary distribution N(0, I_d) as T -> inf).
    steps : int
        Number of discretization steps.
    return_trajectory : bool
        If True, returns the full array of states across all time steps.

    Returns
    -------
    np.ndarray
        Terminal noisy samples X_T of shape x0.shape, or full trajectory of shape (steps + 1, *x0.shape).
    """
    dt = T / steps
    x = np.array(x0, dtype=np.float64, copy=True)
    if return_trajectory:
        traj = np.empty((steps + 1, *x.shape), dtype=np.float64)
        traj[0] = x

    sqrt_2dt = np.sqrt(2.0 * dt)
    for n in range(steps):
        dw = np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        x = x - x * dt + sqrt_2dt * dw
        if return_trajectory:
            traj[n + 1] = x

    return traj if return_trajectory else x


def gmm_exact_score_1d(x: np.ndarray, t: float, weights: np.ndarray, means0: np.ndarray, vars0: np.ndarray) -> np.ndarray:
    """
    Compute the exact analytical score function nabla_x log p_t(x) for a 1D Gaussian Mixture Model
    evolving under the Ornstein-Uhlenbeck forward process.

    At time t:
        m_k(t) = m_k(0) * exp(-t)
        sigma_k^2(t) = 1 + (sigma_k^2(0) - 1) * exp(-2t)
    """
    x_col = x[:, None]  # (N, 1)
    means_t = means0[None, :] * np.exp(-t)
    vars_t = 1.0 + (vars0[None, :] - 1.0) * np.exp(-2.0 * t)
    stds_t = np.sqrt(vars_t)

    # Component densities pi_k * N(x; m_k(t), sigma_k^2(t))
    comp_densities = weights[None, :] * norm.pdf(x_col, loc=means_t, scale=stds_t)
    total_density = np.sum(comp_densities, axis=1, keepdims=True) + 1e-12

    # Responsibilities r_k(x, t)
    responsibilities = comp_densities / total_density

    # Score = - sum_k r_k(x, t) * (x - m_k(t)) / sigma_k^2(t)
    score = -np.sum(responsibilities * (x_col - means_t) / vars_t, axis=1)
    return score


def backward_diffusion_exact_score(
    xT: np.ndarray,
    score_fn,
    T: float = 5.0,
    steps: int = 500,
    alpha: float = 0.0,
    return_trajectory: bool = False,
):
    """
    Simulate the reverse-time backward diffusion process using an analytical score function:
        dX_t = [X_t + (1 + alpha) * score(X_t, t)] dt + sqrt(2 * alpha) dW_t

    Setting alpha = 0 yields the deterministic Probability Flow ODE.
    Setting alpha = 1 yields the standard reverse-time SDE.
    """
    dt = T / steps
    x = np.array(xT, dtype=np.float64, copy=True)
    if return_trajectory:
        traj = np.empty((steps + 1, *x.shape), dtype=np.float64)
        traj[0] = x

    for n in range(steps, 0, -1):
        t = n * dt
        s = score_fn(x, t)
        dw = np.random.normal(loc=0.0, scale=1.0, size=x.shape) if alpha > 0 else 0.0
        x = x + dt * (x + (1.0 + alpha) * s) + np.sqrt(2.0 * alpha * dt) * dw
        if return_trajectory:
            traj[steps - n + 1] = x

    return traj if return_trajectory else x


class NonParametricKNNScoreDiffusion:
    """
    Non-parametric reverse-time diffusion baseline estimating step-wise reverse drift
    Y_hat(x, t) approx E[X_{t - dt} - X_t | X_t = x] via k-Nearest Neighbors (k-NN) regression.
    Used to empirically test the memorization vs. generalization boundary in generative modeling.
    """

    def __init__(self, T: float = 5.0, steps: int = 100, n_neighbors: int = 15):
        self.T = T
        self.steps = steps
        self.dt = T / steps
        self.n_neighbors = n_neighbors
        self.models = []

    def fit(self, x0: np.ndarray):
        """
        Generate a forward OU trajectory from x0 and train step-wise k-NN regressors
        to predict the reverse displacement X_{t - dt} - X_t from X_t.
        """
        traj = forward_ou_diffusion(x0, T=self.T, steps=self.steps, return_trajectory=True)
        self.models = []
        for n in range(1, self.steps + 1):
            x_curr = traj[n].reshape(len(x0), -1)
            target_disp = (traj[n - 1] - traj[n]).reshape(len(x0), -1)
            knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance", n_jobs=-1)
            knn.fit(x_curr, target_disp)
            self.models.append(knn)
        return self

    def sample(self, xT: np.ndarray, return_trajectory: bool = False) -> np.ndarray:
        """
        Integrate the deterministic reverse-time trajectory from noise X_T back to t = 0.
        """
        x = np.array(xT, dtype=np.float64, copy=True)
        orig_shape = x.shape
        x_flat = x.reshape(len(x), -1)

        if return_trajectory:
            traj = np.empty((self.steps + 1, *orig_shape), dtype=np.float64)
            traj[0] = x

        for idx, knn in enumerate(reversed(self.models)):
            disp = knn.predict(x_flat)
            x_flat = x_flat + disp
            if return_trajectory:
                traj[idx + 1] = x_flat.reshape(orig_shape)

        return traj if return_trajectory else x_flat.reshape(orig_shape)
