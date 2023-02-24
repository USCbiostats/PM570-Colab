import jax
import jax.numpy as jnp
import jax.random as rdm


# naive simulation of genotype with no LD structure
def naive_sim_genotype(n_samples: int, p_snps: int, rng_key):
  key, f_key, h1_key, h2_key = rdm.split(rng_key, 4)
  freq = rdm.uniform(f_key, shape=(p_snps,), minval=0.01, maxval=0.5)
  h1 = rdm.bernoulli(h1_key, freq, shape=(n_samples, p_snps)).astype(int) 
  h2 = rdm.bernoulli(h2_key, freq, shape=(n_samples, p_snps)).astype(int) 
  X = h1 + h2

  return X
