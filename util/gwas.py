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
    results = lax.scan(ols, y, X.T)
    return results


def trait_scan_ols(X, y):
    results = _trait_scan_ols(X, y)
    return pd.DataFrame(
        {
            "beta": results.T[0],
            "se": results.T[1],
            "tscore": results.T[2],
            "pvalue": results.T[3],
        }
    )


# OLS with variables (x, covar) and y
def ols(x, y):
    n = len(y)
    covar = jnp.ones((n,))
    X = jnp.hstack((x[:, jnp.newaxis], covar))

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
    p_value = jnp.array(2 * stats.t.sf(abs(t_scores), df=df))

    return jnp.array([beta_hat, se, t_scores, pvalue]).T[0]
