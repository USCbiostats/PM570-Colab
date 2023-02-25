import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.random as rdm
import jax.scipy.stats as stats


# naive simulation of quantitative trait
def naive_trait_sim(X: jnp.ndarray, causal_prop: float, h2g: float, rng_key):

  n_samples, p_snps = X.shape

  # split our key
  rng_key, beta_key, choice_key, env_key = rdm.split(rng_key, 4)
  
  num_causal = int(jnp.ceil(p_snps * causal_prop))

  # simulate causal effects
  beta = jnp.sqrt(h2g / num_causal) * rdm.normal(key=beta_key, shape=(num_causal,))

  # select causal SNPs
  causals = rdm.choice(choice_key, p_snps, (num_causal,), replace=False)

  # generate genetic values
  g = X[:, causals] @ beta

  s2g = jnp.var(g)
  s2e = ((1 / h2g) - 1) * s2g

  # generate phenotype/trait
  y = g + jnp.sqrt(s2e) * rdm.normal(key=env_key, shape=(n_samples,))

  return y
  

def naive_disease_sim(X: jnp.ndarray, causal_prop: float, h2g: float, prevalence: float, rng_key):

  n_samples, p_snps = X.shape

  # split our key
  rng_key, beta_key, choice_key, env_key = rdm.split(rng_key, 4)
  
  num_causal = int(jnp.ceil(p_snps * causal_prop))

  # simulate causal effects
  beta = jnp.sqrt(h2g / num_causal) * rdm.normal(key=beta_key, shape=(num_causal,))

  # select causal SNPs
  causals = rdm.choice(choice_key, p_snps, (num_causal,), replace=False)

  # generate genetic values
  g = X[:, causals] @ beta

  # rescale betas to ensure h2g matches specified h2g
  s2g = jnp.var(g)
  beta *= jnp.sqrt(h2g / s2g)

  # final g based on rescale beta
  g = X[:, causals] @ beta
  g = g - jnp.mean(g, axis=0)

  # upper quantile function to get threshold
  t = -stats.norm.ppf(prevalence, scale=jnp.sqrt(h2g))

  # if liability is past threshold, then disease = 1
  y = (g >= t).astype(int)

  return y
