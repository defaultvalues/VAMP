import numpy as np
from scipy.special import ndtr  # numerically stable normal CDF
from matplotlib import pyplot as plt

class NonnegativeVAMP:
    def __init__(self, G, sigma2=1e-2, lam=0.1):
        self.G = G
        self.M, self.N = G.shape
        self.sigma2 = sigma2
        self.lam = lam

        # SVD
        self.U, self.S, self.Vt = np.linalg.svd(G, full_matrices=False)

    def run(self, z, max_iter=30):

        U, S, Vt = self.U, self.S, self.Vt
        V = Vt.T

        # init
        x = np.zeros(self.N)
        tau_x = 1.0

        for t in range(max_iter):

            # -------------------------
            # MODULE A: Linear MMSE
            # -------------------------

            z_tilde = U.T @ z

            shrink = S / (S**2 + self.sigma2)

            x_A = V @ (shrink * z_tilde)

            tau_A = np.mean(self.sigma2 / (S**2 + self.sigma2))

            # -------------------------
            # MODULE B: Nonnegative denoiser
            # -------------------------

            r = x_A  # VAMP coupling

            tau_B = tau_A + 1e-12

            x_new = np.maximum(r - self.lam * tau_B, 0)

            # variance update
            tau_x = np.mean(x_new > 0) * tau_B

            x = x_new

        return x


class my_VAMP:
    """
    Correct VAMP for linear model z = G x + noise, x >= 0, x ~ Exp(lam).

    Two-module message-passing loop:
      Module A: Linear MMSE using SVD, outputs extrinsic (r̂₁, γ₁)
      Module B: Truncated-Gaussian MMSE denoiser (nonneg exponential prior),
                outputs extrinsic (r̂₂, γ₂)
    Both modules exchange *extrinsic* (self-removed) messages each iteration.
    """

    def __init__(self, G, sigma2=1e-2, lam=0.1, damping=0.8, prior="trunc_exp", rho=0.05):
        self.G = G
        self.M, self.N = G.shape
        self.sigma2 = sigma2
        self.lam = lam
        self.damping = damping
        self.prior = prior

        rho_arr = np.asarray(rho, dtype=float)
        if rho_arr.ndim == 0:
            self.rho_vec = np.full(self.N, float(rho_arr))
        elif rho_arr.size == self.N:
            self.rho_vec = rho_arr.reshape(self.N)
        else:
            raise ValueError("rho must be a scalar or an array of length N.")
        # Keep scalar-like API for backward compatibility in user code.
        self.rho = float(self.rho_vec[0]) if np.allclose(self.rho_vec, self.rho_vec[0]) else self.rho_vec

        # precompute SVD once
        self.U, self.S, self.Vt = np.linalg.svd(G, full_matrices=False)

        if self.prior not in ("trunc_exp", "bernoulli_exp"):
            raise ValueError("prior must be 'trunc_exp' or 'bernoulli_exp'.")
        if np.any((self.rho_vec <= 0.0) | (self.rho_vec >= 1.0)):
            raise ValueError("all rho values must be in (0, 1).")

    # ------------------------------------------------------------------
    # Module B: MMSE denoiser for p(x) = lam*exp(-lam*x), x >= 0
    #   Effective channel: r = x + N(0, 1/gamma)
    #   Posterior: x | r  ~  TruncatedNormal(mu_eff = r - lam/gamma,
    #                                         var = 1/gamma,  support [0, inf))
    # Returns:
    #   x_hat      : MMSE estimate
    #   alpha      : average Jacobian E[dg/dr] (needed for extrinsic)
    #   post_var   : element-wise posterior variance estimate
    # ------------------------------------------------------------------
    def _module_B_trunc_exp(self, r_hat, gamma):
        sigma = 1.0 / np.sqrt(np.maximum(gamma, 1e-30))
        mu_eff = r_hat - self.lam / np.maximum(gamma, 1e-30)
        t = np.clip(mu_eff / sigma, -50.0, 50.0)   # clip prevents t**2 overflow

        phi = np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)
        Phi = ndtr(t)                               # P(N(0,1) <= t)

        safe = Phi > 1e-10
        mills = np.where(safe, phi / np.where(safe, Phi, 1.0), 0.0)

        x_hat = np.maximum(np.where(safe, mu_eff + sigma * mills, 0.0), 0.0)

        # dg/dr = 1 - t*mills - mills^2  (= posterior var * gamma)
        alpha_vec = np.where(safe, 1.0 - t * mills - mills ** 2, 0.0)
        alpha = np.maximum(np.mean(alpha_vec), 1e-10)
        post_var = np.maximum(alpha_vec / np.maximum(gamma, 1e-30), 1e-20)

        return x_hat, alpha, post_var

    def _module_B_bernoulli_exp(self, r_hat, gamma):
        """
        MMSE denoiser for Bernoulli-Exponential prior:
                    p(x_n) = (1-rho_n) delta(x_n) + rho_n * lam * exp(-lam*x_n), x_n>=0
        Effective channel: r = x + N(0, 1/gamma)
        """
        tau = 1.0 / np.maximum(gamma, 1e-30)
        sigma = np.sqrt(tau)

        mu_eff = r_hat - self.lam * tau
        t = np.clip(mu_eff / sigma, -50.0, 50.0)
        phi = np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)
        Phi = np.maximum(ndtr(t), 1e-12)
        kappa = phi / Phi

        # Active branch posterior moments (truncated Gaussian)
        m1 = np.maximum(mu_eff + sigma * kappa, 0.0)
        v1 = tau * np.maximum(1.0 - t * kappa - kappa ** 2, 1e-12)

        # Stable posterior active probability w = P(active | r)
        logZ0 = -0.5 * (np.log(2.0 * np.pi * tau) + (r_hat ** 2) / tau)
        logZ1 = (
            np.log(self.lam)
            - self.lam * r_hat
            + 0.5 * (self.lam ** 2) * tau
            + np.log(Phi)
        )
        log_num = np.log(self.rho_vec) + logZ1
        log_den0 = np.log(1.0 - self.rho_vec) + logZ0
        delta = np.clip(log_den0 - log_num, -60.0, 60.0)
        w = 1.0 / (1.0 + np.exp(delta))

        x_hat = w * m1
        post_var = np.maximum(w * (v1 + m1 ** 2) - x_hat ** 2, 1e-20)

        # Posterior-mean denoiser identity for AWGN channel: dE[x|r]/dr = Var[x|r]/tau
        alpha_vec = np.clip(post_var / np.maximum(tau, 1e-30), 1e-12, 1.0)
        alpha = float(np.maximum(np.mean(alpha_vec), 1e-10))

        return x_hat, alpha, post_var

    def _module_B(self, r_hat, gamma):
        if self.prior == "trunc_exp":
            return self._module_B_trunc_exp(r_hat, gamma)
        return self._module_B_bernoulli_exp(r_hat, gamma)

    def _prior_moments(self):
        if self.prior == "trunc_exp":
            mean0 = 1.0 / self.lam
            var0 = 1.0 / (self.lam ** 2)
            return mean0, var0

        # Bernoulli-Exponential moments
        mean0 = self.rho_vec / self.lam
        second_moment = 2.0 * self.rho_vec / (self.lam ** 2)
        var0 = np.maximum(second_moment - mean0 ** 2, 1e-12)
        return mean0, var0

    def run(self, z, max_iter=50, tol=1e-7, return_info=False):
        U, S, Vt = self.U, self.S, self.Vt
        V = Vt.T
        z_tilde = U.T @ z   # project observation once: shape (M,)
        d = S ** 2 / self.sigma2  # (M,) d_i = s_i² / sigma²

        # --- Initialisation ---
        # gamma_2: precision of the prior message from Module B → Module A
        prior_mean, prior_var = self._prior_moments()
        prior_mean_vec = np.full(self.N, prior_mean) if np.ndim(prior_mean) == 0 else np.asarray(prior_mean, dtype=float)
        prior_var_vec = np.full(self.N, prior_var) if np.ndim(prior_var) == 0 else np.asarray(prior_var, dtype=float)
        prior_var_eff = float(np.maximum(np.mean(prior_var_vec), 1e-12))

        gamma_2 = 1.0 / prior_var_eff
        r_hat_2 = np.array(prior_mean_vec, copy=True)
        x_hat = np.zeros(self.N)
        post_var = np.array(prior_var_vec, copy=True)
        converged = False

        for it in range(max_iter):

            # ============================================================
            # Module A: Linear MMSE
            #   Prior on x: N(r̂₂, γ₂⁻¹ I)
            #   x̂₁ = (AᵀA/σ² + γ₂ I)⁻¹ (Aᵀz/σ² + γ₂ r̂₂)
            # Using SVD: B⁻¹ = V diag(1/(d_i+γ₂)) Vᵀ + (1/γ₂)(I - VVᵀ)
            # ============================================================
            r_tilde = Vt @ r_hat_2                  # (M,)
            coeff = (S * z_tilde / self.sigma2 + gamma_2 * r_tilde) / (d + gamma_2)
            x_hat_1 = V @ coeff + (r_hat_2 - V @ r_tilde)  # null-space += r̂₂ component

            # α₁ = mean over M row-space dims only (no null-space term).
            # Including (N-M)/γ₂ causes γ₁ → ∞ as γ₂ → ∞, leading to overflow.
            alpha_1 = max(float(np.mean(1.0 / (d + gamma_2))), 1e-10)

            # Extrinsic A → B:  remove Module B's prior contribution
            gamma_1 = np.clip(1.0 / alpha_1 - gamma_2, 1e-10, 1e8)
            r_hat_1 = np.clip(
                (x_hat_1 / alpha_1 - gamma_2 * r_hat_2) / gamma_1,
                -1e4, 1e4
            )

            # ============================================================
            # Module B: Nonneg exponential MMSE denoiser
            # ============================================================
            x_hat_2, alpha_2, post_var_2 = self._module_B(r_hat_1, gamma_1)
            alpha_2 = max(alpha_2, 1e-10)

            # Extrinsic B → A:  remove Module A's message contribution
            raw_gamma_2_new = 1.0 / alpha_2 - gamma_1
            if raw_gamma_2_new > 1e-10:
                # Normal extrinsic update
                gamma_2_new = min(raw_gamma_2_new, 1e8)
                r_hat_2_new = np.clip(
                    (x_hat_2 / alpha_2 - gamma_1 * r_hat_1) / gamma_2_new,
                    -1e4, 1e4
                )
            else:
                # Module A is more precise than Module B's posterior.
                # Dividing by ≈0 would blow up r_hat_2.  Instead, send Module B's
                # estimate back as the next prior with a weak precision.
                gamma_2_new = 1.0 / prior_var_eff
                r_hat_2_new = x_hat_2

            # Convergence
            change = np.max(np.abs(x_hat_2 - x_hat))

            # Damping on messages (not on estimate)
            gamma_2 = self.damping * gamma_2_new + (1 - self.damping) * gamma_2
            r_hat_2 = self.damping * r_hat_2_new + (1 - self.damping) * r_hat_2
            x_hat = x_hat_2
            post_var = post_var_2

            if change < tol:
                print(f"  VAMP converged at iteration {it + 1}")
                converged = True
                break

        if return_info:
            info = {
                "converged": converged,
                "iterations": it + 1,
                "posterior_var": post_var,
                "posterior_std": np.sqrt(post_var),
            }
            return x_hat, info
        return x_hat

    def plot_estimate_with_variance(self, x_hat, post_var, x_true=None, top_k=None):
        """
        Visualize estimate and posterior uncertainty.
        If top_k is set, only plot the largest top_k entries of x_hat.
        """
        idx = np.arange(x_hat.size)
        if top_k is not None and top_k < x_hat.size:
            sel = np.argsort(x_hat)[-top_k:]
            sel = np.sort(sel)
            idx = sel

        x_plot = x_hat[idx]
        std_plot = np.sqrt(np.maximum(post_var[idx], 0.0))

        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        if x_true is not None:
            plt.stem(idx, x_true[idx], linefmt='g-', markerfmt='go', basefmt=' ', label='True')
            plt.stem(idx, x_plot, linefmt='r-', markerfmt='ro', basefmt=' ', label='Estimated')
            plt.legend()
        else:
            plt.stem(idx, x_plot, linefmt='r-', markerfmt='ro', basefmt=' ', label='Estimated')
            plt.legend()
        plt.title('VAMP Estimate')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.stem(idx, post_var[idx], linefmt='b-', markerfmt='bo', basefmt=' ')
        plt.title('Posterior Variance')
        plt.xlabel('Index')
        plt.ylabel('Variance')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    M, N = 400, 400

    # sensing matrix (NONNEGATIVE as in your model)
    G = np.abs(np.random.randn(M, N))

    # 构造一个不均匀的G，使得某些列的范数较大，增加稀疏信号恢复的难度
    # col_norms = np.linalg.norm(G, axis=0)
    # scaling_factors = 1.0 + 5.0 * (col_norms - np.min(col_norms)) / (np.max(col_norms) - np.min(col_norms) + 1e-12)
    # G *= scaling_factors

    # 构造一个托普利兹矩阵作为G的特殊结构示例
    from scipy.linalg import toeplitz
    c = np.random.rand(M)
    r = np.random.rand(N)
    G = toeplitz(c, r)

    # 让G的能量集中在一些低维子空间，增加恢复难度
    # U, S, Vt = np.linalg.svd(G, full_matrices=False)
    # S = np.exp(np.linspace(4.0, -1.0, min(M, N)))  # 快速衰减的奇异值
    # G = U @ np.diag(S) @ Vt

    # sparse gamma (true signal)
    gamma_true = np.zeros(N)
    idx = np.random.choice(N, 4, replace=False)
    gamma_true[idx] = np.abs(np.random.rand(4) * 1)

    # observations
    target_snr_db = 20
    signal_power = np.mean((G @ gamma_true) ** 2)
    sigma2 = signal_power / (10 ** (target_snr_db / 10))
    noise = np.sqrt(sigma2) * np.random.randn(M)
    z = G @ gamma_true + noise

    rho_vec = np.zeros(N) + 0.01  # 设置大部分位置的先验激励概率较低
    # rho_vec[idx] = 0.5  # 设置非零位置的先验激励概率较高

    # run VAMP
    vamp = my_VAMP(G, sigma2=sigma2, lam=0.8, prior="bernoulli_exp", rho=rho_vec)
    gamma_est, info = vamp.run(z, max_iter=100, tol=1e-3, return_info=True)
    print("Converged:", info["converged"], "Iterations:", info["iterations"])
    print("Posterior variance mean:", float(np.mean(info["posterior_var"])))

    vamp.plot_estimate_with_variance(
        x_hat=gamma_est,
        post_var=info["posterior_var"],
        x_true=gamma_true,
        top_k=80,
    )