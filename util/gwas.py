import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd

from scipy import stats


# OLS with variables (x, covar) and y
def ols(x, y, covar=None):
    n = len(y)
    if covar is not None:
        X = jnp.hstack((x[:, jnp.newaxis], covar))
        X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

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
    else:
        x = (x - jnp.mean(x)) / jnp.std(x)
        beta_hat = (x.T @ y) / n  # x.T @ x ~ n
        s2 = jnp.mean((y - x * beta_hat) ** 2)
        se = jnp.sqrt(s2 / n)
        z = beta_hat / se
        pvalue = 2 * stats.norm.cdf(-jnp.abs(z))

    return beta_hat, se, t_scores, pvalue
