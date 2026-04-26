# VAMP Implementation Guide (my_VAMP)

This document describes the full process implemented in `my_VAMP` in `GAMP.py`, including model assumptions, module updates, prior options, convergence handling, posterior variance output, and visualization.

## 1. Problem Setup

We solve a sparse nonnegative linear inverse problem:

$$
\mathbf{z} = \mathbf{G}\mathbf{x} + \mathbf{w}, \quad \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}).
$$

- Observation: $\mathbf{z} \in \mathbb{R}^M$
- Sensing matrix: $\mathbf{G} \in \mathbb{R}^{M \times N}$
- Unknown nonnegative sparse signal: $\mathbf{x} \in \mathbb{R}^N$

The class `my_VAMP` implements a two-module VAMP loop:
- Module A: linear MMSE estimator (Gaussian message)
- Module B: prior-based MMSE denoiser (non-Gaussian message)

Messages are exchanged in *extrinsic* form.

## 2. SVD Precomputation

At initialization:

$$
\mathbf{G} = \mathbf{U}\,\mathrm{diag}(\mathbf{S})\,\mathbf{V}^T
$$

This makes each Module A update efficient and numerically stable.

## 3. Prior Choices in Module B

`my_VAMP` supports two priors via `prior`:

1. `trunc_exp`
$$
p(x)=\lambda e^{-\lambda x}\mathbf{1}_{x\ge 0}
$$

2. `bernoulli_exp`
$$
p(x)=(1-\rho)\delta(x)+\rho\,\lambda e^{-\lambda x}\mathbf{1}_{x\ge 0}
$$

The Bernoulli-Exponential prior is typically better when sparsity is strong (many exact zeros).

## 4. Initialization from Prior Moments

Initialization is derived from prior moments:
- Prior mean: `prior_mean`
- Prior variance: `prior_var`

For `trunc_exp`:
$$
\mathbb{E}[x]=1/\lambda, \quad \mathrm{Var}(x)=1/\lambda^2
$$

For `bernoulli_exp`:
$$
\mathbb{E}[x]=\rho/\lambda,
\quad
\mathbb{E}[x^2]=2\rho/\lambda^2,
\quad
\mathrm{Var}(x)=\mathbb{E}[x^2]-\mathbb{E}[x]^2
$$

Then:
- `gamma_2 = 1 / prior_var`
- `r_hat_2 = prior_mean * ones(N)`

## 5. Iterative VAMP Loop

For each iteration:

### 5.1 Module A (Linear MMSE)

Given incoming Gaussian message $(\hat{\mathbf{r}}_2, \gamma_2)$:

1. Project prior mean to SVD coordinates:
$$
\tilde{\mathbf{r}} = \mathbf{V}^T \hat{\mathbf{r}}_2
$$

2. Define
$$
\mathbf{d} = \mathbf{S}^2 / \sigma^2, \quad \tilde{\mathbf{z}} = \mathbf{U}^T\mathbf{z}
$$

3. Compute linear estimate:
$$
\hat{\mathbf{x}}_1
= \mathbf{V}\left(\frac{\mathbf{S}\odot \tilde{\mathbf{z}}/\sigma^2 + \gamma_2\tilde{\mathbf{r}}}{\mathbf{d}+\gamma_2}\right)
+ \left(\hat{\mathbf{r}}_2 - \mathbf{V}\tilde{\mathbf{r}}\right)
$$

4. Average divergence surrogate:
$$
\alpha_1 = \mathrm{mean}\left(\frac{1}{d_i+\gamma_2}\right)
$$

5. Extrinsic A -> B:
$$
\gamma_1 = \frac{1}{\alpha_1} - \gamma_2,
\quad
\hat{\mathbf{r}}_1 = \frac{\hat{\mathbf{x}}_1/\alpha_1 - \gamma_2\hat{\mathbf{r}}_2}{\gamma_1}
$$

The implementation clips `gamma_1` and `r_hat_1` for stability.

### 5.2 Module B (Prior MMSE Denoiser)

Effective scalar channel per coordinate:
$$
r = x + n, \quad n\sim\mathcal{N}(0, 1/\gamma_1)
$$

Module B returns:
- posterior mean estimate `x_hat_2`
- average Jacobian `alpha_2`
- element-wise posterior variance `post_var_2`

#### A) Truncated-Exponential branch

Uses truncated-Gaussian posterior formulas with:
- `mu_eff = r - lambda/gamma`
- `sigma = 1/sqrt(gamma)`
- Mills ratio via `phi/Phi`

Outputs:
- $\hat{x}$
- $\mathrm{Var}(x|r)$
- $\alpha = \langle d\hat{x}/dr \rangle$

#### B) Bernoulli-Exponential branch

Mixture posterior:
$$
p(x|r) = (1-w)\delta(x) + w\,p_{\text{active}}(x|r)
$$

Active branch moments are from truncated-Gaussian formulas. Active probability $w$ is computed in log-domain:
- `logZ0`: inactive evidence
- `logZ1`: active evidence
- logistic transform for `w`

Posterior moments:
$$
\hat{x} = w m_1,
\quad
\mathrm{Var}(x|r) = w(v_1 + m_1^2) - \hat{x}^2
$$

Jacobian identity used in AWGN channel:
$$
\frac{d\,\mathbb{E}[x|r]}{dr} = \frac{\mathrm{Var}(x|r)}{\tau}
$$
with \(\tau=1/\gamma\), giving `alpha_vec = post_var / tau`.

### 5.3 Extrinsic B -> A

$$
\gamma_2^{new} = \frac{1}{\alpha_2} - \gamma_1
$$

- If `raw_gamma_2_new > 0`: normal extrinsic update.
- Else: fallback to weak-prior precision and pass denoised mean directly, preventing divide-by-near-zero explosion.

### 5.4 Damping and Convergence

Message damping:
$$
\gamma_2 \leftarrow \beta\gamma_2^{new} + (1-\beta)\gamma_2,
\quad
\hat{\mathbf{r}}_2 \leftarrow \beta\hat{\mathbf{r}}_2^{new} + (1-\beta)\hat{\mathbf{r}}_2
$$
where `beta = damping`.

Convergence check:
$$
\max_i |\hat{x}_{2,i} - \hat{x}_i| < tol
$$

## 6. Returned Outputs

`run(..., return_info=True)` returns:
- `x_hat`: final estimate
- `info` dictionary:
  - `converged`
  - `iterations`
  - `posterior_var` (element-wise final posterior variance)
  - `posterior_std`

This gives both point estimate and uncertainty.

## 7. Visualization

`plot_estimate_with_variance(x_hat, post_var, x_true=None, top_k=None)`:
- Subplot 1: true vs estimated amplitudes
- Subplot 2: posterior variance profile
- `top_k` optionally focuses on largest estimated coefficients

## 8. Numerical Stability Strategies Implemented

1. Clip normalized variable `t` in denoisers.
2. Floor CDF values (`Phi`) to avoid division by zero.
3. Use log-domain evidence computation in Bernoulli-Exponential branch.
4. Clip message precisions and message means.
5. Fallback when extrinsic precision becomes non-positive.

These guards are essential for stable convergence on sparse, high-dynamic-range cases.

## 9. Typical Usage

```python
vamp = my_VAMP(
    G,
    sigma2=0.01,
    lam=0.8,
    prior="bernoulli_exp",
    rho=0.05,
    damping=0.8,
)

x_hat, info = vamp.run(z, max_iter=100, tol=1e-7, return_info=True)
print(info["converged"], info["iterations"], info["posterior_var"].mean())

vamp.plot_estimate_with_variance(x_hat, info["posterior_var"], x_true=gamma_true, top_k=80)
```

## 10. Parameter Hints

- `rho`: expected sparsity level (active probability). Start near true activity ratio.
- `lam`: controls active-amplitude scale (larger `lam` -> smaller active amplitudes).
- `damping`: increase toward 0.9 if oscillatory; lower (0.6~0.8) if too slow.
- `sigma2`: should match measurement noise variance as closely as possible.

---

If needed, this implementation can be further extended with EM-style updates for `rho`, `lam`, and `sigma2` to reduce manual tuning.
