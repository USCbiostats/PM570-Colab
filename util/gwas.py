import jax
import jax.lax as lax
import jax.numpy as jnp
import pandas as pd

# this is local
from . import optimization as opt


@jax.jit
def _trait_scan(
    X: jnp.ndarray,
    y: jnp.ndarray,
    covar: jnp.ndarray,
    func: opt._AbstractRegressionFunc,
) -> jnp.ndarray:
    """Internal function to perform a scan over genotype data using an abstract regression function"""
    n, p = X.shape
    data, results = lax.scan(func, (y, covar), X.T)
    return results


def trait_scan_simple(X: jnp.ndarray, y: jnp.ndarray, covar: jnp.ndarray = None):
    """Perform a GWAS of continous phenotype `y` over genotype matrix `X`.
    Scan is performed using ordinary least squares (i.e. OLS).

    Args:
        X: the n x p genotype matrix.
        y: the n x 1 phenotype vector.
        covar: n x k covariate matrix (optional).

    Returns:
        `pandas.DataFrame`: A dataframe containing  effect-size estimates,
        standard errors, the z-score, and the log p-value.
    """
    n = len(y)
    if covar is None:
        covar = jnp.ones((n, 1))
    # lets wrap our OLS code to keep the covariate fixed
    results = _trait_scan(X, y, covar, opt.OLSJAX())
    return pd.DataFrame(results, columns=["beta", "se", "zscore", "log.pval"])


def disease_scan_simple(
    X: jnp.ndarray, y: jnp.ndarray, covar: jnp.ndarray = None, tol=1e-3, maxiter=100
):
    """Perform a GWAS of binary phenotype `y` over genotype matrix `X`.
    Scan is performed using logistic regression.

    Args:
        X: the n x p genotype matrix.
        y: the n x 1 phenotype vector.
        covar: n x k covariate matrix (optional).

    Returns:
        `pandas.DataFrame`: A dataframe containing  effect-size estimates (i.e. log-odds),
        standard errors, the z-score, and the log p-value.
    """
    n = len(y)
    if covar is None:
        covar = jnp.ones((n, 1))
    # lets wrap our logistic reg code to keep the covariate fixed
    results = _trait_scan(X, y, covar, opt.LogisticRegression(tol, maxiter))
    return pd.DataFrame(
        results, columns=["beta", "se", "zscore", "log.pval", "converged"]
    )
