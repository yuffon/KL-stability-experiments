from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm
from scipy.special import logsumexp
from matplotlib.lines import Line2D

def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def cov_from_eigs(lam1, lam2, theta=0.0):
    R = rotation(theta)
    return R @ np.diag([lam1, lam2]) @ R.T


def gaussian_logpdf_2d(x, mu, Sigma):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Sigma must be positive definite.")
    invS = np.linalg.inv(Sigma)
    diff = x - mu
    quad = np.einsum("ni,ij,nj->n", diff, invS, diff)
    return -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)


def kl_gaussian(mu1, S1, mu2, S2):
    mu1, mu2 = np.asarray(mu1, float), np.asarray(mu2, float)
    S1, S2 = np.asarray(S1, float), np.asarray(S2, float)
    invS2 = np.linalg.inv(S2)
    d = mu1.size
    diff = mu2 - mu1
    sign1, logdet1 = np.linalg.slogdet(S1)
    sign2, logdet2 = np.linalg.slogdet(S2)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("Covariances must be positive definite.")
    return 0.5 * (
        np.trace(invS2 @ S1)
        + diff.T @ invS2 @ diff
        - d
        + logdet2
        - logdet1
    )


class TruncatedGMM2D:
    """
    Axis-aligned 2D truncated Gaussian mixture.
    """

    def __init__(self, weights, means, sigmas, lower, upper, seed=2026):
        self.weights = np.asarray(weights, float)
        self.weights = self.weights / self.weights.sum()
        self.means = np.asarray(means, float)
        self.sigmas = np.asarray(sigmas, float)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.rng = np.random.RandomState(seed)
        self.K = len(self.weights)

        self.logZ = np.zeros(self.K)
        for k in range(self.K):
            z = 1.0
            for j in range(2):
                m, s = self.means[k, j], self.sigmas[k, j]
                z *= norm.cdf((self.upper[j] - m) / s) - norm.cdf((self.lower[j] - m) / s)
            self.logZ[k] = np.log(z)

    def sample(self, n):
        comp = self.rng.choice(self.K, size=n, p=self.weights)
        x = np.empty((n, 2), dtype=float)
        for k in range(self.K):
            idx = np.where(comp == k)[0]
            if idx.size == 0:
                continue
            for j in range(2):
                m, s = self.means[k, j], self.sigmas[k, j]
                a = (self.lower[j] - m) / s
                b = (self.upper[j] - m) / s
                x[idx, j] = truncnorm.rvs(
                    a, b, loc=m, scale=s, size=idx.size, random_state=self.rng
                )
        return x

    def logpdf(self, x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = x[None, :]
        inside = np.all((x >= self.lower) & (x <= self.upper), axis=1)

        terms = []
        for k in range(self.K):
            lp = np.log(self.weights[k]) - self.logZ[k]
            lp += norm.logpdf(x[:, 0], loc=self.means[k, 0], scale=self.sigmas[k, 0])
            lp += norm.logpdf(x[:, 1], loc=self.means[k, 1], scale=self.sigmas[k, 1])
            terms.append(lp)

        out = logsumexp(np.vstack(terms), axis=0)
        return np.where(inside, out, -np.inf)


def mc_kl(samples, logp, mu, Sigma):
    vals = logp - gaussian_logpdf_2d(samples, mu, Sigma)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(samples.shape[0])


def main():
    outdir = Path("./data2d/experiment2_kl_2d_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    P = TruncatedGMM2D(
        weights=[0.55, 0.45],
        means=[[-2.0, -1.4], [2.1, 1.5]],
        sigmas=[[0.45, 0.75], [0.65, 0.40]],
        lower=[-4.0, -4.0],
        upper=[4.0, 4.0],
        seed=2026,
    )

    #standard Gaussian N1
    mu1 = np.array([0.0, 0.0])
    S1 = np.eye(2)
    threshold = 1.0 / 12.0

    n_mc = 300_000
    samples = P.sample(n_mc)
    logp = P.logpdf(samples)
    kl_p_n1, se_p_n1 = mc_kl(samples, logp, mu1, S1)
    logn1 = gaussian_logpdf_2d(samples, mu1, S1)

    methods = {
        "mean-x": {
            "description": "mean shift along x-axis",
            "mu": lambda a: np.array([0.40 * a, 0.0]),
            "S": lambda a: np.eye(2),
        },
        "mean-diag": {
            "description": "mean shift along diagonal direction",
            "mu": lambda a: np.array([0.28 * a, -0.28 * a]),
            "S": lambda a: np.eye(2),
        },
        "cov-x-expand": {
            "description": "covariance stretch along x-axis",
            "mu": lambda a: mu1,
            "S": lambda a: cov_from_eigs(np.exp(0.35 * a), 1.0, 0.0),
        },
        "cov-y-shrink": {
            "description": "covariance shrink along y-axis",
            "mu": lambda a: mu1,
            "S": lambda a: cov_from_eigs(1.0, np.exp(-0.30 * a), 0.0),
        },
        "cov-diag-expand": {
            "description": "covariance stretch along 45-degree direction",
            "mu": lambda a: mu1,
            "S": lambda a: cov_from_eigs(np.exp(0.35 * a), 1.0, np.pi / 4),
        },
        "cov-rot-anisotropic": {
            "description": "rotated anisotropic covariance perturbation",
            "mu": lambda a: mu1,
            "S": lambda a: cov_from_eigs(np.exp(0.28 * a), np.exp(-0.24 * a), np.pi / 6),
        },
        "both-mean-cov-rot": {
            "description": "joint mean shift and rotated covariance perturbation",
            "mu": lambda a: np.array([0.22 * a, -0.18 * a]),
            "S": lambda a: cov_from_eigs(np.exp(0.22 * a), np.exp(-0.16 * a), np.pi / 5),
        },
    }

    rows = []
    for name, spec in methods.items():
        for alpha in np.linspace(0.0, 1.0, 51):
            mu2 = spec["mu"](alpha)
            S2 = spec["S"](alpha)
            eps = kl_gaussian(mu1, S1, mu2, S2)
            if eps >= threshold + 1e-12:
                continue

            logn2 = gaussian_logpdf_2d(samples, mu2, S2)
            vals2 = logp - logn2
            diff_vals = logn2 - logn1
            eigs = np.linalg.eigvalsh(S2)

            rows.append({
                "method": name,
                "description": spec["description"],
                "alpha": alpha,
                "mu2_x": mu2[0],
                "mu2_y": mu2[1],
                "Sigma2_11": S2[0, 0],
                "Sigma2_12": S2[0, 1],
                "Sigma2_22": S2[1, 1],
                "Sigma2_eig_min": eigs[0],
                "Sigma2_eig_max": eigs[1],
                "KL_N1_to_N2_closed_form": eps,
                "KL_P_to_N1_MC": kl_p_n1,
                "KL_P_to_N1_MC_SE": se_p_n1,
                "KL_P_to_N2_MC": vals2.mean(),
                "KL_P_to_N2_MC_SE": vals2.std(ddof=1) / np.sqrt(n_mc),
                "KL_P_to_N1_minus_KL_P_to_N2": diff_vals.mean(),
                "difference_MC_SE": diff_vals.std(ddof=1) / np.sqrt(n_mc),
                "sqrt_KL_N1_to_N2": np.sqrt(eps),
            })

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "results.csv", index=False)
    max_eps = results["KL_N1_to_N2_closed_form"].max()
    assert max_eps < threshold

    # Grid for contour
    grid_n = 170
    x = np.linspace(-4.5, 4.5, grid_n)
    y = np.linspace(-4.5, 4.5, grid_n)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    P_grid = np.exp(P.logpdf(pts)).reshape(grid_n, grid_n)
    N1_grid = np.exp(gaussian_logpdf_2d(pts, mu1, S1)).reshape(grid_n, grid_n)

    for name in ["cov-x-expand",
                 "cov-diag-expand",
                 "both-mean-cov-rot"]:
        mu2 = methods[name]["mu"](1.0)
        S2 = methods[name]["S"](1.0)
        eps = kl_gaussian(mu1, S1, mu2, S2)
        N2_grid = np.exp(gaussian_logpdf_2d(pts, mu2, S2)).reshape(grid_n, grid_n)

        plt.figure(figsize=(6.3, 5.5))
        plt.contour(X, Y, P_grid, levels=7, linewidths=1.6, cmap="Blues")
        plt.contour(X, Y, N1_grid, levels=7, linestyles="--", linewidths=1.4, cmap="Reds")
        plt.contour(X, Y, N2_grid, levels=7, linestyles="-.", linewidths=1.4, cmap="Greens")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(f"2D density contours: {name}\nKL($N_1||N_2$)={eps:.4f}")
        plt.legend(
            [
                Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='green', lw=2),
            ],
            ['$P$: truncated Gaussian mixture', '$N_1$', '$N_2$']
        )

        plt.axis("equal")

        plt.savefig(outdir / f"contour_{name}.png", dpi=220)
        plt.close()

    plt.figure(figsize=(5.2, 3.0))
    for name in methods:
        sub = results[results["method"] == name]
        plt.plot(
            sub["alpha"],
            sub["KL_P_to_N1_minus_KL_P_to_N2"],
            marker="x",
            markersize=2,
            linewidth=1,
            label=name,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("perturbation size alpha")
    plt.ylabel("estimated $\widehat{\mathrm{KL}}$($P||N_1$) - $\widehat{\mathrm{KL}}$($P||N_2$)")
    plt.title("2D KL difference under Gaussian perturbations")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "kl_difference_vs_perturbation_2d.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.2, 3.0))
    for name in methods:
        sub = results[results["method"] == name]
        plt.plot(
            sub["alpha"],
            sub["KL_N1_to_N2_closed_form"],
            marker="x",
            markersize=2,
            linewidth=1,
            label=name,
        )
    plt.axhline(threshold, linestyle="--", linewidth=1, label="1/12 threshold")
    plt.xlabel("perturbation size alpha")
    plt.ylabel("closed-form KL($N_1||N_2$)")
    plt.title("2D Gaussian perturbation size")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "gaussian_kl_vs_perturbation_2d.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.2, 3.0))
    for name in methods:
        sub = results[results["method"] == name]
        plt.plot(
            sub["sqrt_KL_N1_to_N2"],
            sub["KL_P_to_N1_minus_KL_P_to_N2"],
            marker="o",
            markersize=2,
            linewidth=1,
            label=name,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("$\sqrt{\mathrm{KL}(N_1||N_2)}$")
    plt.ylabel("estimated $\widehat{\mathrm{KL}}$($P||N_1$) - $\widehat{\mathrm{KL}}$($P||N_2$)")
    plt.title("2D KL difference versus square-root Gaussian KL")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "kl_difference_vs_sqrt_epsilon_2d.png", dpi=220)
    plt.close()

    endpoint = results.loc[results.groupby("method")["alpha"].idxmax()]
    endpoint = endpoint[[
        "method", "alpha", "mu2_x", "mu2_y",
        "Sigma2_11", "Sigma2_12", "Sigma2_22",
        "KL_N1_to_N2_closed_form",
        "KL_P_to_N1_MC", "KL_P_to_N2_MC",
        "KL_P_to_N1_minus_KL_P_to_N2",
        "difference_MC_SE",
    ]]

    print("Saved outputs to:", outdir)
    print(f"Monte Carlo samples: {n_mc}")
    print(f"KL(P||N1) = {kl_p_n1:.6f} ± {1.96 * se_p_n1:.6f} (approx. 95% MC interval)")
    print(f"Max KL(N1||N2) = {max_eps:.6f} < 1/12 = {threshold:.6f}")
    print("\nEndpoint summary:")
    print(endpoint.to_string(index=False))

if __name__ == "__main__":
    main()
