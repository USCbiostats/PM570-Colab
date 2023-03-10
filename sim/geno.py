import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm

import pandas_plink as pdp

# naive simulation of genotype with no LD structure
def naive_sim_genotype(n_samples: int, p_snps: int, rng_key):
    """Simulate genotype with no LD.

    Args:
        n_samples: the number of samples to simulate.
        p_snps: the number of SNPs to simulate.
        rng_key: the `jax.random.PRNGKey` to sample data with

    Returns:
        `jax.ndarray`: A 0/1/2 genotype matrix of shape `n_samples` by `p_snps`.
    """
    key, f_key, h1_key, h2_key = rdm.split(rng_key, 4)
    freq = rdm.uniform(f_key, shape=(p_snps,), minval=0.01, maxval=0.5)
    h1 = rdm.bernoulli(h1_key, freq, shape=(n_samples, p_snps)).astype(float)
    h2 = rdm.bernoulli(h2_key, freq, shape=(n_samples, p_snps)).astype(float)

    X = h1 + h2

    return X


def sim_geno_from_plink(prefix: str, n_samples: int, rng_key, ld_ridge: float = 0.01):
    """Simulate approximate genotypes using real genotype data from a PLINK
    dataset. Simulated data will reflect LD  patterns in real data, but have
    continous approximations to genotype data under an MVN.

    Args:
        prefix: the path to the PLINK triplet.
        n_samples: the number of samples to generate.
        rng_key: the `jax.random.PRNGKey` to sample data with.
        ld_ridge: an offset to ensure that the LD matrix is PSD.
    """

    # return cholesky L and ldscs
    bim, fam, G = pdp.read_plink(prefix, verbose=False)
    G = jnp.asarray(G.T.compute())

    n, p = G.shape
    # estimate LD for population from PLINK data
    G = (G - jnp.mean(G, axis=0)) / jnp.std(G, axis=0)

    # regularize so that LD is PSD
    LD = jnp.dot(G.T, G) / n + jnp.eye(p) * ld_ridge

    # re-adjust to get proper correlation matrix
    LD = LD / (1 + ld_ridge)

    # compute cholesky decomp for faster sampling/simulation
    L = jnla.cholesky(LD)

    p, p = L.shape

    Z = (L @ rdm.normal(rng_key, shape=(n_samples, p)).T).T
    Z -= jnp.mean(Z, axis=0)
    Z /= jnp.std(Z, axis=0)

    return Z
