from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm
from scipy.special import logsumexp


class TruncatedGaussianMixture1D:
    """Mixture of two-sided truncated 1D Gaussian components."""

    def __init__(self, weights, means, sigmas, lower, upper, seed=2026):
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.means = np.asarray(means, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)
        self.lower = float(lower)
        self.upper = float(upper)
        self.rng = np.random.RandomState(seed)

        assert len(self.weights) == len(self.means) == len(self.sigmas)
        assert np.all(self.weights > 0)
        assert np.all(self.sigmas > 0)
        assert self.lower < self.upper

        self._log_norm_consts = []
        for m, s in zip(self.means, self.sigmas):
            z = norm.cdf((self.upper - m) / s) - norm.cdf((self.lower - m) / s)
            self._log_norm_consts.append(np.log(z))
        self._log_norm_consts = np.asarray(self._log_norm_consts)

    def sample(self, n):
        comp = self.rng.choice(len(self.weights), size=n, p=self.weights)
        x = np.empty(n, dtype=float)

        for k, (m, s) in enumerate(zip(self.means, self.sigmas)):
            idx = np.where(comp == k)[0]
            if idx.size == 0:
                continue
            a = (self.lower - m) / s
            b = (self.upper - m) / s
            x[idx] = truncnorm.rvs(
                a, b, loc=m, scale=s, size=idx.size, random_state=self.rng
            )
        return x

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        log_terms = []
        for w, m, s, log_z in zip(
            self.weights, self.means, self.sigmas, self._log_norm_consts
        ):
            log_terms.append(np.log(w) + norm.logpdf(x, loc=m, scale=s) - log_z)

        logp = logsumexp(np.vstack(log_terms), axis=0)
        logp = np.where((x >= self.lower) & (x <= self.upper), logp, -np.inf)
        return logp


def gaussian_logpdf(x, mu, sigma):
    return norm.logpdf(x, loc=mu, scale=sigma)


def kl_gaussian_1d(mu1, sigma1, mu2, sigma2):
    """
    Closed-form KL(N1 || N2) for 1D Gaussians
    """
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2.0 * sigma2**2) - 0.5


def mc_kl_p_to_gaussian(samples, logp_samples, mu, sigma):
    """Monte Carlo estimate of KL(P || N(mu, sigma^2))."""
    values = logp_samples - gaussian_logpdf(samples, mu, sigma)
    estimate = values.mean()
    se = values.std(ddof=1) / np.sqrt(samples.size)
    return estimate, se


def main():
    outdir = Path("./data/experiment1_kl_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    P = TruncatedGaussianMixture1D(
        weights=[0.5, 0.5],
        means=[-2.2, 2.2],
        sigmas=[0.35, 0.45],
        lower=-4.0,
        upper=4.0,
        seed=2026,
    )

    # Gaussian N1.
    mu1 = 0.0
    sigma1 = 1.0

    epsilon_threshold = 1.0 / 12.0

    # Monte Carlo samples from P.
    n_mc = 300_000
    samples = P.sample(n_mc)
    logp_samples = P.logpdf(samples)

    kl_p_n1, se_p_n1 = mc_kl_p_to_gaussian(samples, logp_samples, mu1, sigma1)


    logn1_samples = gaussian_logpdf(samples, mu1, sigma1)

    methods = {
        "mean-plus": {
            "label": "mean perturbation: mu2 = mu1 + 0.40 alpha",
            "mu": lambda alpha: mu1 + 0.40 * alpha,
            "sigma": lambda alpha: sigma1,
        },
        "mean-minus": {
            "label": "mean perturbation: mu2 = mu1 - 0.40 alpha",
            "mu": lambda alpha: mu1 - 0.40 * alpha,
            "sigma": lambda alpha: sigma1,
        },
        "sigma-expand": {
            "label": "std perturbation: sigma2 = sigma1 exp(0.30 alpha)",
            "mu": lambda alpha: mu1,
            "sigma": lambda alpha: sigma1 * np.exp(0.30 * alpha),
        },
        "sigma-shrink": {
            "label": "std perturbation: sigma2 = sigma1 exp(-0.24 alpha)",
            "mu": lambda alpha: mu1,
            "sigma": lambda alpha: sigma1 * np.exp(-0.24 * alpha),
        },
        "both-plus": {
            "label": "mean+std perturbation: mu2 = mu1 + 0.25 alpha, sigma2 = sigma1 exp(0.20 alpha)",
            "mu": lambda alpha: mu1 + 0.25 * alpha,
            "sigma": lambda alpha: sigma1 * np.exp(0.20 * alpha),
        },
        "both-minus": {
            "label": "mean+std perturbation: mu2 = mu1 - 0.25 alpha, sigma2 = sigma1 exp(-0.18 alpha)",
            "mu": lambda alpha: mu1 - 0.25 * alpha,
            "sigma": lambda alpha: sigma1 * np.exp(-0.18 * alpha),
        },
    }

    alpha_grid = np.linspace(0.0, 1.0, 51)
    rows = []

    for method_name, spec in methods.items():
        for alpha in alpha_grid:
            mu2 = float(spec["mu"](alpha))
            sigma2 = float(spec["sigma"](alpha))
            eps = kl_gaussian_1d(mu1, sigma1, mu2, sigma2)

            if eps >= epsilon_threshold + 1e-12:
                continue

            logn2_samples = gaussian_logpdf(samples, mu2, sigma2)
            kl_p_n2_values = logp_samples - logn2_samples
            kl_p_n2 = kl_p_n2_values.mean()
            se_p_n2 = kl_p_n2_values.std(ddof=1) / np.sqrt(samples.size)

            diff_values = logn2_samples - logn1_samples
            diff = diff_values.mean()
            diff_se = diff_values.std(ddof=1) / np.sqrt(samples.size)

            rows.append(
                {
                    "method": method_name,
                    "description": spec["label"],
                    "alpha": alpha,
                    "mu2": mu2,
                    "sigma2": sigma2,
                    "KL_N1_to_N2_closed_form": eps,
                    "KL_P_to_N1_MC": kl_p_n1,
                    "KL_P_to_N1_MC_SE": se_p_n1,
                    "KL_P_to_N2_MC": kl_p_n2,
                    "KL_P_to_N2_MC_SE": se_p_n2,
                    "KL_P_to_N1_minus_KL_P_to_N2": diff,
                    "difference_MC_SE": diff_se,
                    "sqrt_KL_N1_to_N2": np.sqrt(eps),
                    "K_d": diff/np.sqrt(eps),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "results.csv", index=False)

    max_eps = results["KL_N1_to_N2_closed_form"].max()
    assert max_eps < epsilon_threshold

    max_K_d = results["K_d"].max()
    print("max K_d: ", max_K_d)


    # plot
    x_grid = np.linspace(-5.0, 5.0, 1200)
    p_grid = np.exp(P.logpdf(x_grid))
    n1_grid = np.exp(gaussian_logpdf(x_grid, mu1, sigma1))

    representative_methods = ["sigma-expand", "both-plus"]

    for method_name in representative_methods:
        spec = methods[method_name]
        mu2 = float(spec["mu"](1.0))
        sigma2 = float(spec["sigma"](1.0))
        n2_grid = np.exp(gaussian_logpdf(x_grid, mu2, sigma2))
        eps = kl_gaussian_1d(mu1, sigma1, mu2, sigma2)

        plt.figure(figsize=(6.2, 4.2))
        plt.plot(x_grid, p_grid, linewidth=2, label="$P$: truncated Gaussian mixture")
        plt.plot(x_grid, n1_grid, linewidth=2, linestyle="--", label="$N_1$")
        plt.plot(
            x_grid,
            n2_grid,
            linewidth=2,
            linestyle="-.",
            label="$N_2$",
        )
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title(f"Density comparison: {method_name}\nKL($N_1||N_2$)={eps:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"density_{method_name}.png", dpi=220)
        plt.close()

    plt.figure(figsize=(5.2, 3.0))
    for method_name in methods.keys():
        if method_name == "both-minus":
            pass
        sub = results[results["method"] == method_name]
        plt.plot(
            sub["alpha"],
            sub["KL_P_to_N1_minus_KL_P_to_N2"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=method_name,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("perturbation size alpha")
    plt.ylabel("estimated $\widehat{\mathrm{KL}}$($P||N_1$) - $\widehat{\mathrm{KL}}$($P||N_2$)")
    plt.title("KL difference under Gaussian perturbations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "kl_difference_vs_perturbation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.2, 3.0))
    for method_name in methods.keys():
        sub = results[results["method"] == method_name]
        plt.plot(
            sub["alpha"],
            sub["KL_N1_to_N2_closed_form"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=method_name,
        )
    plt.axhline(epsilon_threshold, linestyle="--", linewidth=1, label="1/12 threshold")
    plt.xlabel("perturbation size alpha")
    plt.ylabel("closed-form KL($N_1||N_2$)")
    plt.title("Gaussian perturbation size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "gaussian_kl_vs_perturbation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.2, 3.0))
    for method_name in methods.keys():
        sub = results[results["method"] == method_name]
        plt.plot(
            sub["sqrt_KL_N1_to_N2"],
            sub["KL_P_to_N1_minus_KL_P_to_N2"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=method_name,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("$\sqrt{\mathrm{KL}(N_1||N_2)}$")
    plt.ylabel("estimated $\widehat{\mathrm{KL}}$($P||N_1$) - $\widehat{\mathrm{KL}}$($P||N_2$)")
    plt.title("KL difference versus square-root Gaussian KL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "kl_difference_vs_sqrt_epsilon.png", dpi=220)
    plt.close()




    endpoint = results.loc[results.groupby("method")["alpha"].idxmax()].copy()
    endpoint = endpoint[
        [
            "method",
            "alpha",
            "mu2",
            "sigma2",
            "KL_N1_to_N2_closed_form",
            "KL_P_to_N1_MC",
            "KL_P_to_N2_MC",
            "KL_P_to_N1_minus_KL_P_to_N2",
            "difference_MC_SE",
        ]
    ]

    print("Saved outputs to:", outdir)
    print(f"Monte Carlo samples: {n_mc}")
    print(f"KL(P||N1) = {kl_p_n1:.6f} ± {1.96 * se_p_n1:.6f} (approx. 95% MC interval)")
    print(f"Max KL(N1||N2) over all retained perturbations = {max_eps:.6f} < 1/12 = {epsilon_threshold:.6f}")
    print("\nEndpoint summary:")
    print(endpoint.to_string(index=False))


if __name__ == "__main__":
    main()
