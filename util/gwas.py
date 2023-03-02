import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd

from scipy import stats


@jax.jit
def _trait_scan_ols(X, y):
    n, p = X.shape
    covar = jnp.ones((n,))
    y, results = lax.scan(ols_jax, y, X.T)
    return results


def trait_scan_ols(X, y):
    """Perform a GWAS of continous phenotype `y` over genotype matrix `X`.
    Scan is performed using ordinary least squares (i.e. OLS).

    Args:
        X: the n x p genotype matrix.
        y: the n x 1 phenotype vector.

    Returns:
        `pandas.DataFrame`: A dataframe containing  effect-size estimates, 
        standard errors, the z-score, and the log p-value.
    """
    results = _trait_scan_ols(X, y)
    return pd.DataFrame(results, columns=["beta", "se", "zscore", "log.pval"])


def _pad_variant(x):
    x = len(x)
    covar = jnp.ones((n, 1))
    X = jnp.hstack((x[:, jnp.newaxis], covar))
    return X

# OLS using QR decomp with variables (x, covar) and y
def ols_qr(y, x):
    # add constant column for mean
    X = _pad_variant(x)

    q_matrix, r_matrix = jnp.linalg.qr(X, mode="reduced")
    qty = q_matrix.T @ y
    beta_hat = jsp.linalg.solve_triangular(r_matrix, qty)
    df = q_matrix.shape[0] - q_matrix.shape[1]
    residual = y - q_matrix @ qty
    rss = jnp.sum(residual**2, axis=0)
    sigma = jnp.sqrt(jnp.sum(residual**2) / df)
    se = (
        jnp.sqrt(
            jnp.diag(
                jsp.linalg.cho_solve((r_matrix, False), jnp.eye(r_matrix.shape[0]))
            )
        )
        * sigma
    )
    t_scores = beta_hat / se
    #p_value = jnp.array(2 * stats.t.sf(abs(t_scores), df=df))
    log_p_value = jnp.log(2) + jsp.stats.norm.logcdf(-jnp.abs(t_scores))

    return y, jnp.array([beta_hat, se, t_scores, log_p_value]).T[0]

# OLS using jax built in least-squares function
def ols_jax(y, x):
    n = len(y)
    # add constant column for mean
    X = _pad_variant(x)

    beta_hat, rss, rank, svals = jnp.linalg.lstsq(X, y)

    se = jnp.sqrt(rss / (n - rank)) / svals
    z_scores = beta_hat / se
    log_p_value = jnp.log(2) + jsp.stats.norm.logcdf(-jnp.abs(t_scores))

    return y, jnp.array([beta_hat, se, t_scores, log_p_value]).T[0]


def logistic_regression(y, x):
    pass
