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
"""Construct and learn from policy targets. Used by Muesli-based agents."""

import functools

import chex
import distrax
import jax
import jax.numpy as jnp
from rlax._src import base


@chex.dataclass(frozen=True)
class PolicyTarget:
  """A dataclass to hold (possibly sampled) policy targets."""
  # The sampled target actions. These may not cover all actions.
  # The shape is [N_targets, ...]
  actions: chex.Array
  # Probabilities for the corresponding actions. They may have been
  # importance weighted. The shape matches log_prob(actions).shape.
  weights: chex.Array


def constant_policy_targets(
    distribution: distrax.DistributionLike,
    rng_key: chex.PRNGKey,
    num_samples: int,
    weights_scale: float = 1.) -> PolicyTarget:
  """Create policy targets with constant weights.

  The actions will be sampled from `distribution` and will all be associated to
  a constant `weights_scale`. If `distribution` is a (discrete or continuous)
  uniform probability distribution, distilling these targets will push the
  agent's policy towards a uniform distribution. The strength of the penalty
  associated with a non-uniform policy depends on `weights_scale` (e.g. in the
  extreme case `weights_scale==0`, distillation loss is 0 for any policy).

  Args:
    distribution: a `distrax` or `tfp` distribution for sampling actions.
    rng_key: a JAX pseudo random number generator.
    num_samples: number of action sampled in the PolicyTarget.
    weights_scale: constant weight associated with each action.

  Returns:
    a PolicyTarget to learn from.
  """
  random_actions = distribution.sample(
      seed=rng_key, sample_shape=(num_samples,))
  return PolicyTarget(
      actions=random_actions,
      weights=weights_scale * jnp.ones(
          (num_samples,) + distribution.batch_shape))


zero_policy_targets = functools.partial(
    constant_policy_targets, weights_scale=0.0)


def sampled_policy_distillation_loss(
    distribution: distrax.DistributionLike,
    policy_targets: PolicyTarget,
    stop_target_gradients: bool = True,
) -> chex.Numeric:
  """Compute a sampled cross-entropy-like loss.

  The loss corresponds to taking the mean of `-weights * log_prob(actions)`,
  where the weights and actions come from a `PolicyTarget` object, and the mean
  is computed over the N samples in the `policy_target` as well as any batch
  dimension. The loss is suitable for both discrete and continuous actions.

  Args:
    distribution: a predicted `distrax` or `tfp` distribution.
    policy_targets: a policy target to learn from.
    stop_target_gradients: bool indicating whether or not to apply a stop
      gradient to the policy_targets, default True.

  Returns:
    a scalar loss.
  """
  # Optionally, stop gradients from propagating into the targets and into
  # the actions; the latter is mostly relevant in continuous control.
  weights = jax.lax.select(
      stop_target_gradients,
      jax.lax.stop_gradient(policy_targets.weights), policy_targets.weights)
  actions = jax.lax.select(
      stop_target_gradients,
      jax.lax.stop_gradient(policy_targets.actions), policy_targets.actions)
  # Compute log-probabilities.
  log_probs = distribution.log_prob(actions)
  # Assert shapes are compatible.
  chex.assert_equal_shape([weights, log_probs])
  # We avoid NaNs from `0 * (-inf)` by using `0 * min_logp` in that case.
  min_logp = jnp.finfo(log_probs.dtype).min
  # We average over the samples, over time and batch, and if the actions are
  # a continuous vector also over the actions.
  return -jnp.mean(weights * jnp.maximum(log_probs, min_logp))


def cmpo_policy_targets(
    prior_distribution,
    embeddings,
    rng_key,
    baseline_value,
    q_provider,
    advantage_normalizer,
    *,
    num_actions,
    min_target_advantage=-jnp.inf,
    max_target_advantage=1.0,
    kl_weight=1.0,
) -> PolicyTarget:
  """Policy targets for Clipped MPO.

  The policy targets are in-expectation proportional to:
    `prior(a|s) * exp(clip(norm(Q(s, a))))`

  See "Muesli: Combining Improvements in Policy Optimization" by Hessel et al.
  (https://arxiv.org/pdf/2104.06159.pdf).

  Args:
    prior_distribution: the prior policy distribution.
    embeddings: embeddings for the `q_provider`.
    rng_key: a JAX pseudo random number generator key.
    baseline_value: the baseline for `advantage_normalizer`.
    q_provider: a fn to compute q values.
    advantage_normalizer: a fn to normalise advantages.
    *,
    num_actions: The total number of discrete actions.
    min_target_advantage: The minimum advantage of a policy target.
    max_target_advantage: The max advantage of a policy target.
    kl_weight: The coefficient for the KL regularizer.

  Returns:
    the clipped MPO policy targets.
  """
  # Expecting shape [B].
  chex.assert_rank(baseline_value, 1)
  rng_key, query_rng_key = jax.random.split(rng_key)
  del rng_key

  # Producing all actions with shape [num_actions, B].
  batch_size, = baseline_value.shape
  actions = jnp.broadcast_to(
      jnp.expand_dims(jnp.arange(num_actions, dtype=jnp.int32), axis=-1),
      [num_actions, batch_size])

  # Using vmap over the num_actions in axis=0.
  def _query_q(actions):
    return q_provider(
        # Using the same rng_key for the all actions samples.
        rng_key=query_rng_key,
        action=actions,
        embeddings=embeddings)
  qvalues = jax.vmap(_query_q)(actions)

  # Using the same advantage normalization as for policy gradients.
  raw_advantage = advantage_normalizer(
      returns=qvalues, baseline_value=baseline_value)
  clipped_advantage = jnp.clip(
      raw_advantage, min_target_advantage,
      max_target_advantage)

  # Construct and normalise the weights.
  log_prior = prior_distribution.log_prob(actions)
  weights = softmax_policy_target_normalizer(
      log_prior + clipped_advantage / kl_weight)
  policy_targets = PolicyTarget(actions=actions, weights=weights)
  return policy_targets


def sampled_cmpo_policy_targets(
    prior_distribution,
    embeddings,
    rng_key,
    baseline_value,
    q_provider,
    advantage_normalizer,
    *,
    num_actions=2,
    min_target_advantage=-jnp.inf,
    max_target_advantage=1.0,
    kl_weight=1.0,
) -> PolicyTarget:
  """Policy targets for sampled CMPO.

  As in CMPO the policy targets are in-expectation proportional to:
    `prior(a|s) * exp(clip(norm(Q(s, a))))`
  However we only sample a subset of the actions, this allows to scale to
  large discrete action spaces and to continuous actions.

  See "Muesli: Combining Improvements in Policy Optimization" by Hessel et al.
  (https://arxiv.org/pdf/2104.06159.pdf).

  Args:
    prior_distribution: the prior policy distribution.
    embeddings: embeddings for the `q_provider`.
    rng_key: a JAX pseudo random number generator key.
    baseline_value: the baseline for `advantage_normalizer`.
    q_provider: a fn to compute q values.
    advantage_normalizer: a fn to normalise advantages.
    *,
    num_actions: The number of actions to expand on each step.
    min_target_advantage: The minimum advantage of a policy target.
    max_target_advantage: The max advantage of a policy target.
    kl_weight: The coefficient for the KL regularizer.

  Returns:
    the sampled clipped MPO policy targets.
  """
  # Expecting shape [B].
  chex.assert_rank(baseline_value, 1)
  query_rng_key, action_key = jax.random.split(rng_key)
  del rng_key

  # Sampling the actions from the prior.
  actions = prior_distribution.sample(
      seed=action_key, sample_shape=[num_actions])

  # Using vmap over the num_expanded in axis=0.
  def _query_q(actions):
    return q_provider(
        # Using the same rng_key for the all actions samples.
        rng_key=query_rng_key,
        action=actions,
        embeddings=embeddings)
  qvalues = jax.vmap(_query_q)(actions)

  # Using the same advantage normalization as for policy gradients.
  raw_advantage = advantage_normalizer(
      returns=qvalues, baseline_value=baseline_value)
  clipped_advantage = jnp.clip(
      raw_advantage, min_target_advantage, max_target_advantage)

  # The expected normalized weight would be 1.0. The weights would be
  # normalized, if the baseline_value is the log of the expected weight. I.e.,
  # if the baseline_value is log(sum_a(prior(a|s) * exp(Q(s, a)/c))).
  weights = jnp.exp(clipped_advantage / kl_weight)

  # The weights are tiled, if using multiple continuous actions.
  # It is OK to use multiple continuous actions inside the Q(s, a),
  # because the action is sampled from the joint distribution
  # and weight is not based on non-joint probabilities.
  log_prob = prior_distribution.log_prob(actions)
  weights = jnp.broadcast_to(
      base.lhs_broadcast(weights, log_prob), log_prob.shape)
  return PolicyTarget(actions=actions, weights=weights)


def softmax_policy_target_normalizer(log_weights):
  """Returns self-normalized weights.

  The self-normalizing weights introduce a significant bias,
  if computing the average weight from a small number of samples.

  Args:
    log_weights: log unnormalized weights, shape `[num_targets, ...]`.

  Returns:
    Weights divided by average weight from sample. Weights sum to `num_targets`.
  """
  num_targets = log_weights.shape[0]
  return num_targets * jax.nn.softmax(log_weights, axis=0)


def loo_policy_target_normalizer(log_weights):
  """A leave-one-out normalizer.

  Args:
    log_weights: log unnormalized weights, shape `[num_targets, ...]`.

  Returns:
    Weights divided by a consistent estimate of the average weight. The weights
    are not guaranteed to sum to `num_targets`.
  """
  num_targets = log_weights.shape[0]
  weights = jnp.exp(log_weights)
  # Using a safe consistent estimator of the average weight, independently of
  # the numerator.
  # The unnormalized weight are already approximately normalized by a
  # baseline_value, so we use `1` as the initial estimate of the average weight.
  avg_weight = (
      1 + jnp.sum(weights, axis=0, keepdims=True) - weights) / num_targets
  return weights / avg_weight
