import numpy as np


def generate_block_sparse_signal(n, block_size, active_prob, sigma_x2=1.0, rng=None):
    """Generate a complex block-sparse signal with Bernoulli-Gaussian block prior."""
    if rng is None:
        rng = np.random.default_rng()

    if n % block_size != 0:
        raise ValueError("n must be divisible by block_size")

    g = n // block_size
    x = np.zeros(n, dtype=np.complex128)

    for i in range(g):
        if rng.random() < active_prob:
            start = i * block_size
            end = start + block_size
            real = rng.standard_normal(block_size)
            imag = rng.standard_normal(block_size)
            x[start:end] = np.sqrt(sigma_x2 / 2.0) * (real + 1j * imag)

    return x


def module1_block_bg(
    r2_t,
    gamma2_t,
    active_prob,
    sigma_x2,
    block_size,
):
    """
    Module 1 (denoising):
        q1^(t)(x) propto p(x) * N(x; r2^(t), (gamma2^(t))^{-1} I)

    Returns:
        x1_hat_t = E_{q1^(t)}[x]
        eta1_t   = ( (1/N) Tr(Cov_{q1^(t)}(x)) )^{-1}

    Prior per block x_i (length d):
        p(x_i) = (1-lambda) delta(x_i) + lambda CN(0, sigma_x2 I)
    Incoming message:
        CN(x; r2, gamma2^{-1} I)
    """
    n = r2_t.size
    if n % block_size != 0:
        raise ValueError("Signal length must be divisible by block_size")

    g = n // block_size
    d = block_size

    c_scalar = sigma_x2 / (1.0 + gamma2_t * sigma_x2)
    alpha = (gamma2_t * sigma_x2) / (1.0 + gamma2_t * sigma_x2)

    x1_hat_t = np.zeros_like(r2_t)
    tr_cov_sum = 0.0

    eps = 1e-16
    active_prob = float(np.clip(active_prob, eps, 1.0 - eps))

    for i in range(g):
        start = i * d
        end = start + d
        r = r2_t[start:end]
        norm2 = float(np.vdot(r, r).real)

        # Log-domain posterior active probability for numerical stability.
        # Complex-valued CN likelihood:
        # CN(r; 0, v I) = (pi v)^(-d) * exp(-||r||^2 / v)
        v0 = 1.0 / gamma2_t
        v1 = sigma_x2 + v0

        # p(r_i | inactive) = CN(r_i; 0, gamma2^{-1} I)
        log_p_inactive = d * (np.log(gamma2_t) - np.log(np.pi)) - gamma2_t * norm2
        # p(r_i | active) = CN(r_i; 0, (sigma_x2 + gamma2^{-1}) I)
        log_p_active = -d * np.log(np.pi * v1) - norm2 / v1

        log_a = np.log(active_prob) + log_p_active
        log_i = np.log(1.0 - active_prob) + log_p_inactive
        log_z = np.logaddexp(log_a, log_i)
        pi_i = float(np.exp(log_a - log_z))

        mu_i = alpha * r
        xhat_i = pi_i * mu_i

        x1_hat_t[start:end] = xhat_i

        tr_cov_i = pi_i * d * c_scalar + pi_i * (1.0 - pi_i) * float(np.vdot(mu_i, mu_i).real)
        tr_cov_sum += tr_cov_i

    avg_var = tr_cov_sum / n
    eta1_t = 1.0 / max(avg_var, 1e-16)

    return x1_hat_t, eta1_t


def module2_lmmse_svd(y, r1_t, gamma1_t, gamma_w, svd_cache):
    """
    Module 2 (linear MMSE):
        x2_hat^(t) and eta2^(t) from LMMSE with incoming N(x; r1^(t), (gamma1^(t))^{-1} I).
    """
    u, s, vh, n = svd_cache

    v = vh.conj().T
    k = s.size

    uy = u.conj().T @ y
    vr = v.conj().T @ r1_t

    den = gamma1_t + gamma_w * (s ** 2)
    coeff_y = gamma_w * s / den
    coeff_r = gamma1_t / den

    # Component in span(V)
    x_span = v @ (coeff_y * uy + coeff_r * vr)
    # Null-space component unchanged by linear measurements
    x_null = r1_t - v @ vr
    x2_hat_t = x_span + x_null

    tr_c2_t = (n - k) / gamma1_t + np.sum(1.0 / den)
    eta2_t = 1.0 / max((tr_c2_t.real / n), 1e-16)

    return x2_hat_t, eta2_t


def vamp_block_sparse(
    y,
    a,
    gamma_w,
    block_size,
    active_prob,
    sigma_x2,
    max_iter=100,
    tol=1e-6,
    gamma2_init=1.0,
    damping=1.0,
    min_precision=1e-12,
):
    """
     VAMP algorithm with block Bernoulli-Gaussian prior, matching the tutorial equations.

     Iteration t:
     1) Module 1:
         q1^(t), x1_hat^(t), eta1^(t)
     2) Extrinsic to Module 2:
         gamma1^(t) = eta1^(t) - gamma2^(t)
         r1^(t)     = (eta1^(t) x1_hat^(t) - gamma2^(t) r2^(t)) / gamma1^(t)
     3) Module 2:
         x2_hat^(t), eta2^(t)
     4) Extrinsic to Module 1:
         gamma2^(t+1) = eta2^(t) - gamma1^(t)
         r2^(t+1)     = (eta2^(t) x2_hat^(t) - gamma1^(t) r1^(t)) / gamma2^(t+1)
    """
    y = np.asarray(y)
    a = np.asarray(a)

    m, n = a.shape
    if y.shape[0] != m:
        raise ValueError("Shape mismatch: y must have length equal to A.shape[0]")
    if n % block_size != 0:
        raise ValueError("A.shape[1] must be divisible by block_size")
    if gamma_w <= 0:
        raise ValueError("gamma_w must be positive")
    if not np.iscomplexobj(a):
        raise ValueError("Complex-only mode: A must be complex-valued")
    if not np.iscomplexobj(y):
        raise ValueError("Complex-only mode: y must be complex-valued")

    dtype = np.complex128

    # Initialization in Algorithm 1.
    r2_t = np.zeros(n, dtype=dtype)
    gamma2_t = float(max(gamma2_init, min_precision))

    u, s, vh = np.linalg.svd(a, full_matrices=False)
    svd_cache = (u, s, vh, n)

    x2_hat_prev = np.zeros(n, dtype=dtype)
    history = []

    converged = False
    for t in range(max_iter):
        # Module 1: x1_hat^(t), eta1^(t)
        x1_hat_t, eta1_t = module1_block_bg(
            r2_t=r2_t,
            gamma2_t=gamma2_t,
            active_prob=active_prob,
            sigma_x2=sigma_x2,
            block_size=block_size,
        )

        # Extrinsic update to Module 2.
        gamma1_raw_t = eta1_t - gamma2_t
        gamma1_t = max(gamma1_raw_t, min_precision)
        r1_t = (eta1_t * x1_hat_t - gamma2_t * r2_t) / gamma1_t

        # Module 2: x2_hat^(t), eta2^(t)
        x2_hat_t, eta2_t = module2_lmmse_svd(
            y=y,
            r1_t=r1_t,
            gamma1_t=gamma1_t,
            gamma_w=gamma_w,
            svd_cache=svd_cache,
        )

        # Extrinsic update to Module 1.
        gamma2_raw_tp1 = eta2_t - gamma1_t
        gamma2_tp1 = damping * gamma2_raw_tp1 + (1.0 - damping) * gamma2_t
        gamma2_tp1 = max(float(gamma2_tp1), min_precision)

        r2_tp1 = (eta2_t * x2_hat_t - gamma1_t * r1_t) / gamma2_tp1

        rel_change = np.linalg.norm(x2_hat_t - x2_hat_prev) / (np.linalg.norm(x2_hat_t) + 1e-12)
        history.append(
            {
                "iter": t + 1,
                "rel_change": float(rel_change),
                "eta1": float(eta1_t),
                "eta2": float(eta2_t),
                "gamma1": float(gamma1_t),
                "gamma2": float(gamma2_tp1),
            }
        )

        x2_hat_prev = x2_hat_t.copy()
        r2_t = r2_tp1
        gamma2_t = gamma2_tp1

        if rel_change < tol:
            print(f"Converged at iteration {t+1} with relative change {rel_change:.2e}")
            converged = True
            break

    if not converged:
        print("Not Converged !")

    return {"x_hat": x2_hat_prev, "history": history}


def demo():
    rng = np.random.default_rng()

    m = 300
    n = 400
    block_size = 8

    active_prob = 0.2
    sigma_x2 = 1.0

    x_true = generate_block_sparse_signal(
        n=n,
        block_size=block_size,
        active_prob=active_prob,
        sigma_x2=sigma_x2,
        rng=rng,
    )

    a = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2.0 * m)

    snr_db = 10.0
    signal_power = np.mean(np.abs(a @ x_true) ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)

    w = noise_std * (rng.standard_normal(m) + 1j * rng.standard_normal(m))
    y = a @ x_true + w

    gamma_w = 1.0 / max(noise_power, 1e-16)

    result = vamp_block_sparse(
        y=y,
        a=a,
        gamma_w=gamma_w,
        block_size=block_size,
        active_prob=active_prob,
        sigma_x2=sigma_x2,
        max_iter=500,
        tol=1e-7,
        gamma2_init=1.0,
        damping=1.0,
    )

    x_hat = result["x_hat"]
    nmse = np.linalg.norm(x_hat - x_true) ** 2 / (np.linalg.norm(x_true) ** 2 + 1e-16)

    print("VAMP block sparse recovery demo")
    print(f"iterations: {len(result['history'])}")
    print(f"NMSE: {10 * np.log10(nmse + 1e-16):.2f} dB")

    # Compare magnitudes for complex signals.
    import matplotlib.pyplot as plt
    idx = np.arange(n)
    support_mask = np.abs(x_true) > 1e-10

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].stem(idx, np.abs(x_true), linefmt='b-', markerfmt='bo', label='|True Signal|')
    axes[0].stem(idx, np.abs(x_hat), linefmt='r-', markerfmt='ro', label='|Estimated Signal|')
    axes[0].set_title('Complex Block Sparse Signal Recovery via VAMP')
    axes[0].set_ylabel('Magnitude')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(idx[support_mask], np.angle(x_true[support_mask]), 'bo', label='angle(x_true)')
    axes[1].plot(idx[support_mask], np.angle(x_hat[support_mask]), 'r.', label='angle(x_hat)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_title('Phase Recovery on True Support')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
