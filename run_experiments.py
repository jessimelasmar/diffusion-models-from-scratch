"""
End-to-End Reproducible Experiment Script:
1. Forward OU convergence validation on 1D Gaussian Mixture Models & 2D Spirals.
2. Exact-score reverse diffusion (Probability Flow ODE vs Reverse SDE).
3. Non-parametric k-NN score estimation & Memorization vs. Generalization diagnostic.
"""

import numpy as np
from src.diffusion_models import (
    forward_ou_diffusion,
    gmm_exact_score_1d,
    backward_diffusion_exact_score,
    NonParametricKNNScoreDiffusion,
)


def sample_1d_gmm(n_samples: int, weights: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    return np.random.normal(loc=means[components], scale=np.sqrt(vars_[components]))


def sample_2d_spiral(n_samples: int) -> np.ndarray:
    t = np.linspace(0.5, 4.0 * np.pi, n_samples)
    x = t * np.cos(t) + np.random.normal(0, 0.15, size=n_samples)
    y = t * np.sin(t) + np.random.normal(0, 0.15, size=n_samples)
    return np.column_stack([x, y])


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Forward OU convergence on 2D Spiral
    spiral_0 = sample_2d_spiral(5000)
    spiral_T = forward_ou_diffusion(spiral_0, T=8.0, steps=800)
    emp_mean = np.mean(spiral_T, axis=0)
    emp_cov = np.cov(spiral_T, rowvar=False)
    print(f"[2D Spiral Forward OU at T=8.0] Empirical Mean: {emp_mean.round(3)}")
    print(f"[2D Spiral Forward OU at T=8.0] Empirical Covariance:\n{emp_cov.round(3)}")

    # 2. Exact-score backward generation on 3-mode 1D GMM
    weights = np.array([0.4, 0.3, 0.3])
    means0 = np.array([2.0, 9.0, 17.0])
    vars0 = np.array([0.64, 2.25, 1.44])

    x_noise = np.random.normal(0.0, 1.0, size=20000)
    score_fn = lambda x, t: gmm_exact_score_1d(x, t, weights, means0, vars0)
    x_gen_exact = backward_diffusion_exact_score(x_noise, score_fn, T=5.0, steps=500, alpha=0.0)
    print(f"[1D GMM Exact Score Backward] Generated Mean: {x_gen_exact.mean():.3f} (Theoretical: {np.sum(weights * means0):.3f})")

    # 3. Non-parametric k-NN score estimation & memorization test
    train_A = sample_1d_gmm(4000, weights, means0, vars0)
    train_B = sample_1d_gmm(4000, weights, means0, vars0)

    knn_A = NonParametricKNNScoreDiffusion(T=5.0, steps=50, n_neighbors=15).fit(train_A)
    knn_B = NonParametricKNNScoreDiffusion(T=5.0, steps=50, n_neighbors=15).fit(train_B)

    test_seed = np.array([0.45, -0.32])
    traj_A = knn_A.sample(test_seed, return_trajectory=True)
    traj_B = knn_B.sample(test_seed, return_trajectory=True)
    print(f"[Memorization Diagnostic] Endpoint divergence from identical noise seed: {np.abs(traj_A[-1] - traj_B[-1]).round(3)}")
