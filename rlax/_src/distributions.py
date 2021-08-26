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
from typing import Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
from rlax._src import base

Array = chex.Array
Numeric = chex.Numeric


DiscreteDistribution = collections.namedtuple(
    "DiscreteDistribution", ["sample", "probs", "logprob", "entropy", "kl"])
ContinuousDistribution = collections.namedtuple(
    "ContinuousDistribution", ["sample", "prob", "logprob", "entropy",
                               "kl_to_standard_normal", "kl"])


def categorical_sample(key, probs):
  """Sample from a set of discrete probabilities."""
  probs = probs / probs.sum(axis=-1, keepdims=True)
  is_valid = jnp.logical_and(jnp.all(jnp.isfinite(probs)), jnp.all(probs >= 0))
  sample = distrax.Categorical(probs=probs).sample(seed=key)
  return jnp.where(is_valid, sample, -1)


def softmax(temperature=1.):
  """A softmax distribution."""

  def sample_fn(key: Array, logits: Array):
    return distrax.Softmax(logits, temperature).sample(seed=key)

  def probs_fn(logits: Array):
    return distrax.Softmax(logits, temperature).probs

  def logprob_fn(sample: Array, logits: Array):
    return distrax.Softmax(logits, temperature).log_prob(sample)

  def entropy_fn(logits: Array):
    return distrax.Softmax(logits, temperature).entropy()

  def kl_fn(p_logits: Array, q_logits: Array):
    return categorical_kl_divergence(p_logits, q_logits, temperature)

  return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn,
                              kl_fn)


def clipped_entropy_softmax(temperature=1., entropy_clip=1.):
  """A softmax distribution with clipped entropy (1 is eq to not clipping)."""

  def sample_fn(key: Array, logits: Array, action_spec=None):
    del action_spec
    return distrax.Softmax(logits, temperature).sample(seed=key)

  def probs_fn(logits: Array, action_spec=None):
    del action_spec
    return distrax.Softmax(logits, temperature).probs

  def logprob_fn(sample: Array, logits: Array, action_spec=None):
    del action_spec
    return distrax.Softmax(logits, temperature).log_prob(sample)

  def entropy_fn(logits: Array):
    return jnp.minimum(
        distrax.Softmax(logits, temperature).entropy(),
        entropy_clip * jnp.log(logits.shape[-1]))

  def kl_fn(p_logits: Array, q_logits: Array):
    return categorical_kl_divergence(p_logits, q_logits, temperature)

  return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn,
                              kl_fn)


def _mix_with_uniform(probs, epsilon):
  """Mix an arbitrary categorical distribution with a uniform distribution."""
  num_actions = probs.shape[-1]
  uniform_probs = jnp.ones_like(probs) / num_actions
  return (1 - epsilon) * probs + epsilon * uniform_probs


def epsilon_softmax(epsilon, temperature):
  """An epsilon-softmax distribution."""

  def sample_fn(key: Array, logits: Array):
    probs = jax.nn.softmax(logits / temperature)
    probs = _mix_with_uniform(probs, epsilon)
    return categorical_sample(key, probs)

  def probs_fn(logits: Array):
    probs = jax.nn.softmax(logits / temperature)
    return _mix_with_uniform(probs, epsilon)

  def log_prob_fn(sample: Array, logits: Array):
    probs = jax.nn.softmax(logits / temperature)
    probs = _mix_with_uniform(probs, epsilon)
    return base.batched_index(jnp.log(probs), sample)

  def entropy_fn(logits: Array):
    probs = jax.nn.softmax(logits / temperature)
    probs = _mix_with_uniform(probs, epsilon)
    return -jnp.nansum(probs * jnp.log(probs), axis=-1)

  def kl_fn(p_logits: Array, q_logits: Array):
    return categorical_kl_divergence(p_logits, q_logits, temperature)

  return DiscreteDistribution(sample_fn, probs_fn, log_prob_fn, entropy_fn,
                              kl_fn)


def greedy():
  """A greedy distribution."""

  def sample_fn(key: Array, preferences: Array):
    return distrax.Greedy(preferences).sample(seed=key)

  def probs_fn(preferences: Array):
    return distrax.Greedy(preferences).probs

  def log_prob_fn(sample: Array, preferences: Array):
    return distrax.Greedy(preferences).log_prob(sample)

  def entropy_fn(preferences: Array):
    return distrax.Greedy(preferences).entropy()

  return DiscreteDistribution(sample_fn, probs_fn, log_prob_fn, entropy_fn,
                              None)


def epsilon_greedy(epsilon=None):
  """An epsilon-greedy distribution."""

  def sample_fn(key: Array, preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).sample(seed=key)

  def probs_fn(preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).probs

  def logprob_fn(sample: Array, preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).log_prob(sample)

  def entropy_fn(preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).entropy()

  return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn, None)


def safe_epsilon_softmax(epsilon, temperature):
  """Tolerantly handles the temperature=0 case."""
  egreedy = epsilon_greedy(epsilon)
  unsafe = epsilon_softmax(epsilon, temperature)

  def sample_fn(key: Array, logits: Array):
    return jax.lax.cond(temperature > 0,
                        (key, logits), lambda tup: unsafe.sample(*tup),
                        (key, logits), lambda tup: egreedy.sample(*tup))

  def probs_fn(logits: Array):
    return jax.lax.cond(temperature > 0,
                        logits, unsafe.probs,
                        logits, egreedy.probs)

  def log_prob_fn(sample: Array, logits: Array):
    return jax.lax.cond(temperature > 0,
                        (sample, logits), lambda tup: unsafe.logprob(*tup),
                        (sample, logits), lambda tup: egreedy.logprob(*tup))

  def entropy_fn(logits: Array):
    return jax.lax.cond(temperature > 0,
                        logits, unsafe.entropy,
                        logits, egreedy.entropy)

  def kl_fn(p_logits: Array, q_logits: Array):
    return categorical_kl_divergence(p_logits, q_logits, temperature)

  return DiscreteDistribution(sample_fn, probs_fn, log_prob_fn, entropy_fn,
                              kl_fn)


def _add_gaussian_noise(key, sample, sigma):
  noise = jax.random.normal(key, shape=sample.shape) * sigma
  return sample + noise


def gaussian_diagonal(sigma=None):
  """A gaussian distribution with diagonal covariance matrix."""

  def sample_fn(key: Array, mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(mu, sigma).sample(seed=key)

  def prob_fn(sample: Array, mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(mu, sigma).prob(sample)

  def logprob_fn(sample: Array, mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(mu, sigma).log_prob(sample)

  def entropy_fn(mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(mu, sigma).entropy()

  def kl_to_standard_normal_fn(mu: Array, sigma: Array = sigma):
    kl_mean, kl_cov = distrax.MultivariateNormalDiag(mu, sigma).kl_divergence(
        distrax.MultivariateNormalDiag(
            jnp.zeros_like(mu), jnp.ones_like(sigma)))
    return kl_mean + kl_cov

  def kl_fn(mu_1: Array, sigma_1: Numeric, mu_0: Array, sigma_0: Numeric):
    return distrax.MultivariateNormalDiag(mu_0, sigma_0).kl_divergence(
        distrax.MultivariateNormalDiag(mu_1, sigma_1))

  return ContinuousDistribution(sample_fn, prob_fn, logprob_fn, entropy_fn,
                                kl_to_standard_normal_fn, kl_fn)


def squashed_gaussian(sigma_min=-4, sigma_max=0.):
  """A squashed gaussian distribution with diagonal covariance matrix."""

  def minmaxvals(a, action_spec):
    # broadcasts action spec to action shape
    min_shape = action_spec.minimum.shape or (1,)
    max_shape = action_spec.maximum.shape or (1,)
    min_vals = jnp.broadcast_to(action_spec.minimum, a.shape[:-1] + min_shape)
    max_vals = jnp.broadcast_to(action_spec.maximum, a.shape[:-1] + max_shape)
    return min_vals, max_vals

  def sigma_activation(sigma, sigma_min=sigma_min, sigma_max=sigma_max):
    return jnp.exp(sigma_min + 0.5 * (sigma_max - sigma_min) *
                   (jnp.tanh(sigma) + 1.))

  def mu_activation(mu):
    return jnp.tanh(mu)

  def transform(a, action_spec):
    min_vals, max_vals = minmaxvals(a, action_spec)
    scale = (max_vals - min_vals) * 0.5
    actions = (jnp.tanh(a) + 1.0) * scale + min_vals
    return actions

  def inv_transform(a, action_spec):
    min_vals, max_vals = minmaxvals(a, action_spec)
    scale = (max_vals - min_vals) * 0.5
    actions_tanh = (a - min_vals) / scale - 1.
    return jnp.arctanh(actions_tanh)

  def log_det_jacobian(a, action_spec):
    min_vals, max_vals = minmaxvals(a, action_spec)
    scale = (max_vals - min_vals) * 0.5
    log_ = jnp.sum(jnp.log(scale))
    # computes sum log (1-tanh(a)**2)
    log_ += jnp.sum(2. * (jnp.log(2.) - a - jax.nn.softplus(-2. * a)))
    return log_

  def sample_fn(key: Array,
                mu: Array,
                sigma: Array,
                action_spec,
                ):
    mu = mu_activation(mu)
    sigma = sigma_activation(sigma)
    action = _add_gaussian_noise(key, mu, sigma)
    return transform(action, action_spec)

  def prob_fn(sample: Array, mu: Array, sigma: Array, action_spec):
    # Support scalar and vector `sigma`. If vector, mu.shape==sigma.shape.
    mu = mu_activation(mu)
    sigma = sigma_activation(sigma)
    # Compute pdf for multivariate gaussian.
    d = mu.shape[-1]
    det = jnp.prod(sigma**2, axis=-1)
    z = ((2 * jnp.pi)**(0.5 * d)) * (det**0.5)
    exp = jnp.exp(-0.5 * jnp.sum(
        ((mu - inv_transform(sample, action_spec)) / sigma)**2, axis=-1))
    det_jacobian = jnp.prod(jnp.clip(1 - sample**2, 0., 1.) + 1e-6)
    return exp / (z * det_jacobian)

  def logprob_fn(sample: Array, mu: Array, sigma: Array, action_spec):
    # Support scalar and vector `sigma`. If vector, mu.shape==sigma.shape.
    mu = mu_activation(mu)
    sigma = sigma_activation(sigma)
    # Compute logpdf for multivariate gaussian in a numerically safe way.
    d = mu.shape[-1]
    half_logdet = jnp.sum(jnp.log(sigma), axis=-1)
    logz = half_logdet + 0.5 * d * jnp.log(2 * jnp.pi)
    logexp = -0.5 * jnp.sum(
        ((mu - inv_transform(sample, action_spec)) / sigma)**2, axis=-1)
    return logexp - logz - log_det_jacobian(sample, action_spec)

  def entropy_fn(mu: Array, sigma: Array):
    # Support scalar and vector `sigma`. If vector, mu.shape==sigma.shape.
    mu = mu_activation(mu)
    sigma = sigma_activation(sigma)
    # Compute entropy in a numerically safe way.
    d = mu.shape[-1]
    half_logdet = jnp.sum(jnp.log(sigma), axis=-1)
    return half_logdet + 0.5 * d * (1 + jnp.log(2 * jnp.pi))

  def kl_to_standard_normal_fn(mu: Array,
                               sigma: Array,
                               per_dimension: bool = False):
    mu = mu_activation(mu)
    sigma = sigma_activation(sigma)
    v = jnp.clip(sigma**2, 1e-6, 1e6)
    kl = 0.5 * (v + mu**2 - jnp.ones_like(mu) - jnp.log(v))
    if not per_dimension:
      kl = jnp.sum(kl, axis=-1)
    return kl

  def kl_fn(mu_1: Array, sigma_1: Numeric, mu_0: Array, sigma_0: Numeric):
    sigma_0 = sigma_activation(sigma_0)
    mu_0 = mu_activation(mu_0)
    sigma_1 = sigma_activation(sigma_1)
    mu_1 = mu_activation(mu_1)
    return multivariate_normal_kl_divergence(mu_0, sigma_0, mu_1, sigma_1)

  return ContinuousDistribution(sample_fn, prob_fn, logprob_fn, entropy_fn,
                                kl_to_standard_normal_fn, kl_fn)


def categorical_importance_sampling_ratios(pi_logits_t: Array,
                                           mu_logits_t: Array,
                                           a_t: Array) -> Array:
  """Compute importance sampling ratios from logits.

  Args:
    pi_logits_t: unnormalized logits at time t for the target policy.
    mu_logits_t: unnormalized logits at time t for the behavior policy.
    a_t: actions at time t.

  Returns:
    importance sampling ratios.
  """
  return distrax.importance_sampling_ratios(distrax.Categorical(
      pi_logits_t), distrax.Categorical(mu_logits_t), a_t)


def categorical_cross_entropy(
    labels: Array,
    logits: Array
) -> Array:
  """Computes the softmax cross entropy between sets of logits and labels.

  See "Deep Learning" by Goodfellow et al.
  (http://www.deeplearningbook.org/contents/prob.html).

  Args:
    labels: a valid probability distribution (non-negative, sum to 1).
    logits: unnormalized log probabilities.

  Returns:
    a scalar loss.
  """
  chex.assert_rank([logits, labels], 1)
  return -jnp.sum(labels * jax.nn.log_softmax(logits))


def categorical_kl_divergence(
    p_logits: Array,
    q_logits: Array,
    temperature: float = 1.
) -> Array:
  """Compute the KL between two categorical distributions from their logits.

  Args:
    p_logits: unnormalized logits for the first distribution.
    q_logits: unnormalized logits for the second distribution.
    temperature: the temperature for the softmax distribution, defaults at 1.

  Returns:
    the kl divergence between the distributions.
  """
  return distrax.Softmax(p_logits, temperature).kl_divergence(
      distrax.Softmax(q_logits, temperature))


def decoupled_multivariate_normal_kl_divergence(
    mu_0: Array, sigma_0: Numeric, mu_1: Array, sigma_1: Numeric,
    per_dimension: bool = False
) -> Tuple[Array, Array]:
  """Compute the KL between diagonal Gaussians decomposed into mean and covar-e.

  Args:
    mu_0: array like of mean values for policy 0
    sigma_0: array like of std values for policy 0
    mu_1: array like of mean values for policy 1
    sigma_1: array like of std values for policy 1
    per_dimension: Whether to return a separate kl divergence for each dimension
      on the last axis.

  Returns:
    the kl divergence between the distributions decomposed into mean and
    covariance.
  """
  # Support scalar and vector `sigma`. If vector, mu.shape==sigma.shape.
  sigma_1 = jnp.ones_like(mu_1) * sigma_1
  sigma_0 = jnp.ones_like(mu_0) * sigma_0
  v1 = jnp.clip(sigma_1**2, 1e-6, 1e6)
  v0 = jnp.clip(sigma_0**2, 1e-6, 1e6)
  mu_diff = mu_1 - mu_0
  kl_mean = 0.5 * jnp.divide(mu_diff**2, v1)
  kl_cov = 0.5 * (jnp.divide(v0, v1) - jnp.ones_like(mu_1) + jnp.log(v1) -
                  jnp.log(v0))
  if not per_dimension:
    kl_mean = jnp.sum(kl_mean, axis=-1)
    kl_cov = jnp.sum(kl_cov, axis=-1)

  return kl_mean, kl_cov


def multivariate_normal_kl_divergence(
    mu_0: Array, sigma_0: Numeric, mu_1: Array, sigma_1: Numeric,
) -> Array:
  """Compute the KL between 2 gaussian distrs with diagonal covariance matrices.

  Args:
    mu_0: array like of mean values for policy 0
    sigma_0: array like of std values for policy 0
    mu_1: array like of mean values for policy 1
    sigma_1: array like of std values for policy 1

  Returns:
    the kl divergence between the distributions.
  """
  return distrax.MultivariateNormalDiag(mu_0, sigma_0).kl_divergence(
      distrax.MultivariateNormalDiag(mu_1, sigma_1))
