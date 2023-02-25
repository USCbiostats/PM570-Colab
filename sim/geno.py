import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import jax.random as rdm

import pandas_plink as pdp

# naive simulation of genotype with no LD structure
def naive_sim_genotype(n_samples: int, p_snps: int, rng_key):
  key, f_key, h1_key, h2_key = rdm.split(rng_key, 4)
  freq = rdm.uniform(f_key, shape=(p_snps,), minval=0.01, maxval=0.5)
  h1 = rdm.bernoulli(h1_key, freq, shape=(n_samples, p_snps)).astype(int) 
  h2 = rdm.bernoulli(h2_key, freq, shape=(n_samples, p_snps)).astype(int) 
  X = h1 + h2

  return X

def sim_geno_from_plink(prefix: str, n: int, rng_key, ld_ridge: float = 0.1):

  # return cholesky L and ldscs
  bim, fam, G = pdp.read_plink(prefix, verbose=False)
  G = jnp.asarray(G.T.compute())

  # estimate LD for population from PLINK data
  n, p = [float(x) for x in G.shape]
  p_int = int(p)
  mafs = jnp.mean(G, axis=0) / 2
  G = (G - jnp.mean(G, axis=0)) / jnp.std(G, axis=0) 

  # regularize so that LD is PSD
  LD = jnp.dot(G.T, G) / n + jnp.eye(p_int) * ld_ridge

  # re-adjust to get proper correlation matrix
  LD = LD / (1 + ld_ridge)

  # compute cholesky decomp for faster sampling/simulation
  L = jnla.cholesky(LD)

  p, p = L.shape

  Z = (L @ rdm.normal(size=(n,p)).T).T
  Z -= jnp.mean(Z, axis=0)
  Z /= jnp.std(Z, axis=0)

  return Z

