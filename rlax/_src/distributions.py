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
import warnings

import chex
import distrax
import jax.numpy as jnp

Array = chex.Array
Numeric = chex.Numeric


DiscreteDistribution = collections.namedtuple(
    "DiscreteDistribution", ["sample", "probs", "logprob", "entropy", "kl"])
ContinuousDistribution = collections.namedtuple(
    "ContinuousDistribution", ["sample", "prob", "logprob", "entropy",
                               "kl_to_standard_normal", "kl"])


def categorical_sample(key, probs):
  """Sample from a set of discrete probabilities."""
  warnings.warn(
      "Rlax categorical_sample will be deprecated. "
      "Please use distrax.Categorical.sample instead.",
      PendingDeprecationWarning, stacklevel=2
  )
  return distrax.Categorical(probs=probs).sample(seed=key)


def softmax(temperature=1.):
  """A softmax distribution."""
  warnings.warn(
      "Rlax softmax will be deprecated. "
      "Please use distrax.Softmax instead.",
      PendingDeprecationWarning, stacklevel=2
  )

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

  warnings.warn(
      "Rlax clipped_entropy_softmax will be deprecated. "
      "Please use distrax.Softmax instead.",
      PendingDeprecationWarning, stacklevel=2
  )
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


def greedy():
  """A greedy distribution."""
  warnings.warn(
      "Rlax greedy will be deprecated. "
      "Please use distrax.Greedy instead.",
      PendingDeprecationWarning, stacklevel=2
  )
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

  warnings.warn(
      "Rlax epsilon_greedy will be deprecated. "
      "Please use distrax.EpsilonGreedy instead.",
      PendingDeprecationWarning, stacklevel=2
  )
  def sample_fn(key: Array, preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).sample(seed=key)

  def probs_fn(preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).probs

  def logprob_fn(sample: Array, preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).log_prob(sample)

  def entropy_fn(preferences: Array, epsilon=epsilon):
    return distrax.EpsilonGreedy(preferences, epsilon).entropy()

  return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn, None)


def gaussian_diagonal(sigma=None):
  """A gaussian distribution with diagonal covariance matrix."""

  warnings.warn(
      "Rlax gaussian_diagonal will be deprecated. "
      "Please use distrax MultivariateNormalDiag instead.",
      PendingDeprecationWarning, stacklevel=2
  )

  def sample_fn(key: Array, mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(
        mu, jnp.ones_like(mu) * sigma).sample(seed=key)

  def prob_fn(sample: Array, mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(
        mu, jnp.ones_like(mu) * sigma).prob(sample)

  def logprob_fn(sample: Array, mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(
        mu, jnp.ones_like(mu) * sigma).log_prob(sample)

  def entropy_fn(mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(
        mu, jnp.ones_like(mu) * sigma).entropy()

  def kl_to_standard_normal_fn(mu: Array, sigma: Array = sigma):
    return distrax.MultivariateNormalDiag(
        mu, jnp.ones_like(mu) * sigma).kl_divergence(
            distrax.MultivariateNormalDiag(
                jnp.zeros_like(mu), jnp.ones_like(mu)))

  def kl_fn(mu_0: Array, sigma_0: Numeric, mu_1: Array, sigma_1: Numeric):
    return distrax.MultivariateNormalDiag(
        mu_0, jnp.ones_like(mu_0) * sigma_0).kl_divergence(
            distrax.MultivariateNormalDiag(mu_1, jnp.ones_like(mu_1) * sigma_1))

  return ContinuousDistribution(sample_fn, prob_fn, logprob_fn, entropy_fn,
                                kl_to_standard_normal_fn, kl_fn)


def squashed_gaussian(sigma_min=-4, sigma_max=0.):
  """A squashed gaussian distribution with diagonal covariance matrix."""

  warnings.warn(
      "Rlax squashed_gaussian will be deprecated. "
      "Please use distrax Transformed MultivariateNormalDiag distribution "
      "with chained Tanh/ScalarAffine bijector instead.",
      PendingDeprecationWarning, stacklevel=2
  )

  def sigma_activation(sigma, sigma_min=sigma_min, sigma_max=sigma_max):
    return jnp.exp(sigma_min + 0.5 * (sigma_max - sigma_min) *
                   (jnp.tanh(sigma) + 1.))

  def mu_activation(mu):
    return jnp.tanh(mu)

  def get_squashed_gaussian_dist(mu, sigma, action_spec=None):
    if action_spec is not None:
      scale = 0.5 * (action_spec.maximum - action_spec.minimum)
      shift = action_spec.minimum
      bijector = distrax.Chain([distrax.ScalarAffine(shift=shift, scale=scale),
                                distrax.ScalarAffine(shift=1.0),
                                distrax.Tanh()])
    else:
      bijector = distrax.Tanh()
    return distrax.Transformed(
        distribution=distrax.MultivariateNormalDiag(
            loc=mu_activation(mu), scale_diag=sigma_activation(sigma)),
        bijector=distrax.Block(bijector, ndims=1))

  def sample_fn(key: Array, mu: Array, sigma: Array, action_spec):
    return get_squashed_gaussian_dist(mu, sigma, action_spec).sample(seed=key)

  def prob_fn(sample: Array, mu: Array, sigma: Array, action_spec):
    return get_squashed_gaussian_dist(mu, sigma, action_spec).prob(sample)

  def logprob_fn(sample: Array, mu: Array, sigma: Array, action_spec):
    return get_squashed_gaussian_dist(mu, sigma, action_spec).log_prob(sample)

  def entropy_fn(mu: Array, sigma: Array):
    return get_squashed_gaussian_dist(mu, sigma).distribution.entropy()

  def kl_to_standard_normal_fn(mu: Array, sigma: Array):
    return get_squashed_gaussian_dist(mu, sigma).distribution.kl_divergence(
        distrax.MultivariateNormalDiag(
            jnp.zeros_like(mu), jnp.ones_like(mu)))

  def kl_fn(mu_0: Array, sigma_0: Numeric, mu_1: Array, sigma_1: Numeric):
    return get_squashed_gaussian_dist(mu_0, sigma_0).distribution.kl_divergence(
        get_squashed_gaussian_dist(mu_1, sigma_1).distribution)

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
  warnings.warn(
      "Rlax categorical_importance_sampling_ratios will be deprecated. "
      "Please use distrax.importance_sampling_ratios instead.",
      PendingDeprecationWarning, stacklevel=2
  )
  return distrax.importance_sampling_ratios(distrax.Categorical(
      pi_logits_t), distrax.Categorical(mu_logits_t), a_t)


def categorical_cross_entropy(
    labels: Array,
    logits: Array
) -> Array:
  """Computes the softmax cross entropy between sets of logits and labels.

  See "Deep Learning" by Goodfellow et al.
  (http://www.deeplearningbook.org/contents/prob.html). The computation is
  equivalent to:

                  sum_i (labels_i * log_softmax(logits_i))

  Args:
    labels: a valid probability distribution (non-negative, sum to 1).
    logits: unnormalized log probabilities.

  Returns:
    a scalar loss.
  """
  warnings.warn(
      "Rlax categorical_cross_entropy will be deprecated. "
      "Please use distrax.Categorical.cross_entropy instead.",
      PendingDeprecationWarning, stacklevel=2
  )
  return distrax.Categorical(probs=labels).cross_entropy(
      distrax.Categorical(logits=logits))


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
  warnings.warn(
      "Rlax categorical_kl_divergence will be deprecated. "
      "Please use distrax.Softmax.kl_divergence instead.",
      PendingDeprecationWarning, stacklevel=2
  )
  return distrax.Softmax(p_logits, temperature).kl_divergence(
      distrax.Softmax(q_logits, temperature))


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
  warnings.warn(
      "Rlax multivariate_normal_kl_divergence will be deprecated."
      "Please use distrax.MultivariateNormalDiag.kl_divergence instead.",
      PendingDeprecationWarning, stacklevel=2
  )
  return distrax.MultivariateNormalDiag(mu_0, sigma_0).kl_divergence(
      distrax.MultivariateNormalDiag(mu_1, sigma_1))
