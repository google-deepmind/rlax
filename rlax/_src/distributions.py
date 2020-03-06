# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""JAX functions for working with probability distributions.

Reinforcement learning algorithms often require to estimate suitably
parametrized probability distributions. In this subpackage a distribution is
represented as a collection of functions that may be used to sample from a
distribution, to evaluate its probability mass (or density) function, and
to compute statistics such as its entropy.
"""

import collections
import jax
import jax.numpy as jnp
from rlax._src import base

ArrayLike = base.ArrayLike


DiscreteDistribution = collections.namedtuple(
    "DiscreteDistribution", ["sample", "probs", "logprob", "entropy"])
ContinuousDistribution = collections.namedtuple(
    "ContinuousDistribution", ["sample", "prob", "logprob", "entropy",
                               "kl_to_standard_normal"])


def _categorical_sample(key, probs):
  """Sample from a set of discrete probabilities."""
  cpi = jnp.cumsum(probs, axis=-1)
  rnds = jax.random.uniform(key, shape=probs.shape[:-1] + (1,))
  return jnp.argmin(rnds > cpi, axis=-1)


def softmax(temperature=1.):
  """A softmax distribution."""

  def sample_fn(key: ArrayLike, logits: ArrayLike):
    probs = jax.nn.softmax(logits / temperature)
    return _categorical_sample(key, probs)

  def probs_fn(logits: ArrayLike):
    return jax.nn.softmax(logits / temperature)

  def logprob_fn(sample: ArrayLike, logits: ArrayLike):
    logprobs = jax.nn.log_softmax(logits / temperature)
    return base.batched_index(logprobs, sample)

  def entropy_fn(logits: ArrayLike):
    probs = jax.nn.softmax(logits / temperature)
    logprobs = jax.nn.log_softmax(logits / temperature)
    return -jnp.sum(probs * logprobs, axis=-1)

  return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn)


def _mix_with_uniform(probs, epsilon):
  """Mix an arbitrary categorical distribution with a uniform distribution."""
  num_actions = probs.shape[-1]
  uniform_probs = jnp.ones_like(probs) / num_actions
  return (1 - epsilon) * probs + epsilon * uniform_probs


def epsilon_softmax(epsilon, temperature):
  """An epsilon-softmax distribution."""

  def sample_fn(key: ArrayLike, logits: ArrayLike):
    probs = jax.nn.softmax(logits / temperature)
    probs = _mix_with_uniform(probs, epsilon)
    return _categorical_sample(key, probs)

  def probs_fn(logits: ArrayLike):
    probs = jax.nn.softmax(logits / temperature)
    return _mix_with_uniform(probs, epsilon)

  def log_prob_fn(sample: ArrayLike, logits: ArrayLike):
    probs = jax.nn.softmax(logits / temperature)
    probs = _mix_with_uniform(probs, epsilon)
    return base.batched_index(jnp.log(probs), sample)

  def entropy_fn(logits: ArrayLike):
    probs = jax.nn.softmax(logits / temperature)
    probs = _mix_with_uniform(probs, epsilon)
    return -jnp.sum(probs * jnp.log(probs), axis=-1)

  return DiscreteDistribution(sample_fn, probs_fn, log_prob_fn, entropy_fn)


def _argmax_with_random_tie_breaking(preferences):
  """Compute probabilities greedily with respect to a set of preferences."""
  optimal_actions = (preferences == preferences.max(axis=-1, keepdims=True))
  return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)


def greedy():
  """A greedy distribution."""

  def sample_fn(key: ArrayLike, preferences: ArrayLike):
    probs = _argmax_with_random_tie_breaking(preferences)
    return _categorical_sample(key, probs)

  def probs_fn(preferences: ArrayLike):
    return _argmax_with_random_tie_breaking(preferences)

  def log_prob_fn(sample: ArrayLike, preferences: ArrayLike):
    probs = _argmax_with_random_tie_breaking(preferences)
    return base.batched_index(jnp.log(probs), sample)

  def entropy_fn(preferences: ArrayLike):
    probs = _argmax_with_random_tie_breaking(preferences)
    return -jnp.nansum(probs * jnp.log(probs), axis=-1)

  return DiscreteDistribution(sample_fn, probs_fn, log_prob_fn, entropy_fn)


def epsilon_greedy(epsilon=None):
  """An epsilon-greedy distribution."""

  def sample_fn(key: ArrayLike, preferences: ArrayLike, epsilon=epsilon):
    probs = _argmax_with_random_tie_breaking(preferences)
    probs = _mix_with_uniform(probs, epsilon)
    return _categorical_sample(key, probs)

  def probs_fn(preferences: ArrayLike, epsilon=epsilon):
    probs = _argmax_with_random_tie_breaking(preferences)
    return _mix_with_uniform(probs, epsilon)

  def logprob_fn(sample: ArrayLike, preferences: ArrayLike, epsilon=epsilon):
    probs = _argmax_with_random_tie_breaking(preferences)
    probs = _mix_with_uniform(probs, epsilon)
    return base.batched_index(jnp.log(probs), sample)

  def entropy_fn(preferences: ArrayLike, epsilon=epsilon):
    probs = _argmax_with_random_tie_breaking(preferences)
    probs = _mix_with_uniform(probs, epsilon)
    return -jnp.nansum(probs * jnp.log(probs), axis=-1)

  return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn)


def _add_gaussian_noise(key, sample, sigma):
  noise = jax.random.normal(key, shape=sample.shape) * sigma
  return sample + noise


def gaussian_diagonal(sigma=None):
  """A gaussian distribution with diagonal covariance matrix."""

  def sample_fn(key: ArrayLike, mu: ArrayLike, sigma: ArrayLike = sigma):
    return _add_gaussian_noise(key, mu, sigma)

  def prob_fn(sample: ArrayLike, mu: ArrayLike, sigma: ArrayLike = sigma):
    # Support scalar and vector `sigma`. If vector, mu.shape==sigma.shape.
    sigma = jnp.ones_like(mu) * sigma
    # Compute pdf for multivariate gaussian.
    d = mu.shape[-1]
    det = jnp.prod(sigma ** 2, axis=-1)
    z = ((2 * jnp.pi) ** (0.5 * d)) * (det ** 0.5)
    exp = jnp.exp(-0.5 * jnp.sum(((mu - sample) / sigma)**2, axis=-1))
    return exp / z

  def logprob_fn(sample: ArrayLike, mu: ArrayLike, sigma: ArrayLike = sigma):
    # Support scalar and vector `sigma`. If vector, mu.shape==sigma.shape.
    sigma = jnp.ones_like(mu) * sigma
    # Compute logpdf for multivariate gaussian in a numerically safe way.
    d = mu.shape[-1]
    half_logdet = jnp.sum(jnp.log(sigma), axis=-1)
    logz = half_logdet + 0.5 * d * jnp.log(2 * jnp.pi)
    logexp = -0.5 * jnp.sum(((mu - sample) / sigma) ** 2, axis=-1)
    return logexp - logz

  def entropy_fn(mu: ArrayLike, sigma: ArrayLike = sigma):
    # Support scalar and vector `sigma`. If vector, mu.shape==sigma.shape.
    sigma = jnp.ones_like(mu) * sigma
    # Compute entropy in a numerically safe way.
    d = mu.shape[-1]
    half_logdet = jnp.sum(jnp.log(sigma), axis=-1)
    return half_logdet + 0.5 * d * (1 + jnp.log(2 * jnp.pi))

  def kl_to_standard_normal_fn(mu: ArrayLike, sigma: ArrayLike = sigma):
    v = jnp.clip(sigma**2, 1e-6, 1e6)
    return -0.5 * (jnp.sum(v) + jnp.sum(mu**2) - 1 - jnp.sum(jnp.log(v)))

  return ContinuousDistribution(sample_fn, prob_fn, logprob_fn, entropy_fn,
                                kl_to_standard_normal_fn)


def categorical_importance_sampling_ratios(
    pi_logits_t: ArrayLike,
    mu_logits_t: ArrayLike,
    a_t: ArrayLike
) -> ArrayLike:
  """Compute importance sampling ratios from logits.

  Args:
    pi_logits_t: unnormalized logits at time t for the target policy.
    mu_logits_t: unnormalized logits at time t for the behavior policy.
    a_t: actions at time t.

  Returns:
    importance sampling ratios.
  """
  base.type_assert([pi_logits_t, mu_logits_t, a_t], [float, float, int])

  log_pi_a_t = base.batched_index(jax.nn.log_softmax(pi_logits_t), a_t)
  log_mu_a_t = base.batched_index(jax.nn.log_softmax(mu_logits_t), a_t)
  rho_t = jnp.exp(log_pi_a_t - log_mu_a_t)
  return rho_t


def categorical_cross_entropy(
    labels: ArrayLike,
    logits: ArrayLike
) -> ArrayLike:
  """Computes the softmax cross entropy between sets of logits and labels.

  See "Deep Learning" by Goodfellow et al.
  (http://www.deeplearningbook.org/contents/prob.html).

  Args:
    labels: a valid probability distribution (non-negative, sum to 1).
    logits: unnormalized log probabilities.

  Returns:
    a scalar loss.
  """
  return -jnp.sum(labels * jax.nn.log_softmax(logits))


def categorical_kl_divergence(
    p_logits: ArrayLike,
    q_logits: ArrayLike,
    temperature: float = 1.
) -> ArrayLike:
  """Compute the KL between two categorical distributions from their logits.

  Args:
    p_logits: unnormalized logits for the first distribution.
    q_logits: unnormalized logits for the second distribution.
    temperature: the temperature for the softmax distribution, defaults at 1.

  Returns:
    the kl divergence between the distributions.
  """
  base.type_assert([p_logits, q_logits], float)

  p_logits /= temperature
  q_logits /= temperature

  p = jax.nn.softmax(p_logits)
  log_p = jax.nn.log_softmax(p_logits)
  log_q = jax.nn.log_softmax(q_logits)
  kl = jnp.sum(p * (log_p - log_q), axis=-1)
  return jax.nn.relu(kl)  # Guard against numerical issues giving negative KL.
