# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Maximum A Posteriori Policy Optimization (MPO/V-MPO) ops.

  Maximum a Posteriori Policy Optimisation.
  https://openreview.net/forum?id=S1ANxQW0b

  Relative Entropy Regularized Policy Iteration.
  https://arxiv.org/abs/1812.02256

  V-MPO: On-Policy Maximum a Posteriori Policy Optimization
  for Discrete and Continuous Control.
  https://openreview.net/forum?id=SylOlp4FvH

Since these functions are calculated per-example (with some aggregation over
all examples), they work with many input shapes as long
as input shapes are consistent across inputs. We use E* to denote the shape of
the examples. For example, E* could be [T, B], [B, T], [T], etc as long as E* is
consistent across all function inputs, and function output shape will also
depend on E*.
"""
import functools
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

from absl import logging
import chex
import jax
import jax.numpy as jnp
from rlax._src import base

# This op is not in the list of officially supported ops because the jax team
# have not fully tested it, but it works nonetheless so we add and use it.
# TODO(b/160450576): Remove when this op is officially supported.
jax.interpreters.pxla.multi_host_supported_collectives.add(jax.lax.pmax_p)

Array = chex.Array
Numeric = chex.Numeric
Scalar = chex.Scalar


class LagrangePenalty(NamedTuple):
  # Dual variable responsible for modulating the penalty for this constraint.
  alpha: Array
  # Bound for this constraint.
  epsilon: Scalar
  # Whether to constrain each dimension separately with its own bound epsilon.
  per_dimension: bool = False


class MpoOutputs(NamedTuple):
  """Additional outputs for mpo loss functions."""
  # In VMPO temperature loss is computed across all data so should be scalar, in
  # MPO this is per example with shape E*
  temperature_loss: Numeric
  # These losses are per example with shape E*.
  policy_loss: Array
  kl_loss: Array
  alpha_loss: Array
  # Other outputs.
  normalized_weights: Array
  num_samples: Numeric

_EPSILON = 1e-10  # For numerical stability.
_INFINITY = 1e6


def mpo_loss(
    sample_log_probs: Array,
    sample_q_values: Array,
    temperature_constraint: LagrangePenalty,
    kl_constraints: Sequence[Tuple[Array, LagrangePenalty]],
    projection_operator: Callable[[Numeric], Numeric] = functools.partial(
        jnp.clip, a_min=_EPSILON),
    policy_loss_weight: float = 1.0,
    temperature_loss_weight: float = 1.0,
    kl_loss_weight: float = 1.0,
    alpha_loss_weight: float = 1.0,
    sample_axis: int = 0,
    use_stop_gradient: bool = True,
) -> Tuple[Array, MpoOutputs]:
  """Implements the MPO loss with a KL bound.

  This loss implements the MPO algorithm for policies with a bound for the KL
  between the current and target policy.

  Note: This is a per-example loss which works on any shape inputs as long as
  they are consistent. We denote this shape E* for ease of reference. Args
  sample_log_probs and sample_q_values are shape E + an extra sample axis that
  contains the sampled actions' log probs and q values respectively. For
  example, if sample_axis = 0, the shapes expected will be [S, E*]. Or if
  E* = [T, B] and sample_axis = 1, the shapes expected will be [T, S, B].

  Args:
    sample_log_probs: An array of shape E* + a sample axis inserted at
      sample_axis containing the log probabilities of the sampled actions under
      the current policy.
    sample_q_values: An array of shape E* + a sample axis inserted at
      sample_axis containing the q function values evaluated on the sampled
      actions.
    temperature_constraint: Lagrange constraint for the E-step temperature
      optimization.
    kl_constraints: KL and variables for applying Lagrangian penalties to bound
      them in the M-step, KLs are  [E*, A?]. Here A is the action dimension
      in the case of per-dimension KL constraints.
    projection_operator: Function to project dual variables (temperature and kl
      constraint alphas) into the positive range.
    policy_loss_weight: Weight for the policy loss.
    temperature_loss_weight: Weight for the temperature loss.
    kl_loss_weight: Weight for the KL loss.
    alpha_loss_weight: Weight for the alpha loss.
    sample_axis: Axis in sample_log_probs and sample_q_values that contains the
      sampled actions' log probs and q values respectively. For example, if
      sample_axis = 0, the shapes expected will be [S, E*]. Or if E* = [T, B]
      and sample_axis = 1, the shapes expected will be [T, S, B].
    use_stop_gradient: bool indicating whether or not to apply stop gradient.

  Returns:
    Per example `loss` with shape E*, and additional data including
    the components of this loss and the normalized weights in the
    AdditionalOutputs.
  """
  chex.assert_equal_shape([sample_log_probs, sample_q_values])
  chex.assert_rank(temperature_constraint.epsilon, 0)
  chex.assert_type([
      sample_log_probs, sample_q_values, temperature_constraint.alpha,
      temperature_constraint.epsilon], float)

  for kl, penalty in kl_constraints:
    chex.assert_rank(penalty.epsilon, 0)
    chex.assert_type([kl, penalty.alpha, penalty.epsilon], float)
    if penalty.per_dimension:
      chex.assert_rank(kl, sample_q_values.ndim)
    else:
      chex.assert_rank(kl, sample_q_values.ndim - 1)

  if sample_axis < 0:
    sample_axis += sample_q_values.ndim
  if not 0 <= sample_axis < sample_q_values.ndim:
    raise ValueError(
        f"`sample_axis` {sample_axis} not in array rank {sample_q_values.ndim}")

  # E-Step. Compute temperature loss, weights, and temperature.
  temperature_loss, norm_weights, num_samples = (
      mpo_compute_weights_and_temperature_loss(
          sample_q_values, temperature_constraint, projection_operator,
          sample_axis=sample_axis))

  norm_weights = jax.lax.select(
      use_stop_gradient, jax.lax.stop_gradient(norm_weights), norm_weights)

  # M-Step. Supervised learning on reweighted probabilities using the weights
  # from the E-Step under an additional KL constraint.
  policy_loss = -jnp.sum(norm_weights * sample_log_probs, axis=sample_axis)
  kl_loss, alpha_loss = compute_parametric_kl_penalty_and_dual_loss(
      kl_constraints, projection_operator, use_stop_gradient)

  chex.assert_equal_shape([policy_loss, kl_loss, alpha_loss])

  # Combine all loss components. The final loss is of shape E*.
  loss = (policy_loss_weight * policy_loss +
          temperature_loss_weight * temperature_loss +
          kl_loss_weight * kl_loss +
          alpha_loss_weight * alpha_loss)
  return loss, MpoOutputs(
      temperature_loss=temperature_loss, policy_loss=policy_loss,
      kl_loss=kl_loss, alpha_loss=alpha_loss, normalized_weights=norm_weights,
      num_samples=num_samples)


def mpo_compute_weights_and_temperature_loss(
    sample_q_values: Array,
    temperature_constraint: LagrangePenalty,
    projection_operator: Callable[[Numeric], Numeric],
    sample_axis: int = 0,
) -> Tuple[Array, Array, Scalar]:
  """Computes the weights and temperature loss for MPO.

  The E-Step computes a non-parameteric sample-based approximation of the
  current policy by reweighting the state-action value function.

  Here, we compute this nonparametric policy and optimize the temperature
  parameter used in the reweighting.

  Args:
    sample_q_values: An array of shape E* + a sample axis inserted at
      sample_axis containing the q function values evaluated on the sampled
      actions.
    temperature_constraint: Lagrange constraint for the E-step temperature
      optimization.
    projection_operator: Function to project temperature into the positive
      range.
    sample_axis: Axis in sample_q_values containing sampled actions.

  Returns:
    The temperature loss, normalized weights and number of actions samples per
    state.
  """
  chex.assert_rank(temperature_constraint.epsilon, 0)
  chex.assert_type([sample_q_values, temperature_constraint.alpha,
                    temperature_constraint.epsilon], float)

  if sample_axis < 0:
    sample_axis += sample_q_values.ndim
  if not 0 <= sample_axis < sample_q_values.ndim:
    raise ValueError(
        f"`sample_axis` {sample_axis} not in array rank {sample_q_values.ndim}")

  n_action_samples = sample_q_values.shape[sample_axis]

  # Clip the temperature value (temperature must be positive).
  temperature = projection_operator(temperature_constraint.alpha)
  epsilon = temperature_constraint.epsilon

  # Scale the Q-values.
  scaled_sample_q_values = sample_q_values / temperature

  # Temperature optimization.
  q_logsumexp = jax.scipy.special.logsumexp(
      scaled_sample_q_values, axis=sample_axis, keepdims=True)

  # The temperature loss encourages the current and previous policy to stay
  # close. This loss optimizes the convex dual of an upper bound on the average
  # KL (epsilon) between the current and previous state-action values.
  temperature_loss = (
      temperature * epsilon +
      (temperature * (jnp.squeeze(q_logsumexp, axis=sample_axis)
                      - jnp.log(n_action_samples))))

  # The weights corresponds to a softmax over state-action values.
  weights = jnp.exp(scaled_sample_q_values - q_logsumexp)

  # Normalize the weights before the M-Step
  norm_weights = weights / jnp.sum(weights, axis=sample_axis, keepdims=True)

  return temperature_loss, norm_weights, n_action_samples


def compute_parametric_kl_penalty_and_dual_loss(
    kl_constraints: Sequence[Tuple[Array, LagrangePenalty]],
    projection_operator: Callable[[Numeric], Numeric],
    use_stop_gradient: bool = True,
) -> Tuple[Array, Array]:
  """Optimize hard KL constraints between the current and previous policies."""
  for kl, penalty in kl_constraints:
    chex.assert_rank(penalty.epsilon, 0)
    chex.assert_type([kl, penalty.alpha, penalty.epsilon], float)

  kl_losses, alpha_losses = [], []
  for kl, penalty in kl_constraints:
    kl_loss, alpha_loss, _ = kl_constraint_loss(
        kl, penalty, projection_operator, use_stop_gradient)
    kl_losses.append(kl_loss)
    alpha_losses.append(alpha_loss)
  kl_loss, alpha_loss = sum(kl_losses), sum(alpha_losses)
  return kl_loss, alpha_loss


def vmpo_loss(
    sample_log_probs: Array,
    advantages: Array,
    temperature_constraint: LagrangePenalty,
    kl_constraints: Sequence[Tuple[Array, LagrangePenalty]],
    projection_operator: Callable[[Numeric], Numeric] = functools.partial(
        jnp.clip, a_min=_EPSILON),
    restarting_weights: Optional[Array] = None,
    importance_weights: Optional[Array] = None,
    top_k_fraction: float = 0.5,
    policy_loss_weight: float = 1.0,
    temperature_loss_weight: float = 1.0,
    kl_loss_weight: float = 1.0,
    alpha_loss_weight: float = 1.0,
    axis_name: Optional[str] = None,
    use_stop_gradient: bool = True,
) -> Tuple[Array, MpoOutputs]:
  """Calculates the V-MPO policy improvement loss.

  Note: This is a per-example loss which works on any shape inputs as long as
  they are consistent. We denote the shape of the examples E* for ease of
  reference.

  Args:
    sample_log_probs: Log probabilities of actions for each example. Shape E*.
    advantages: Advantages for the E-step. Shape E*.
    temperature_constraint: Lagrange constraint for the E-step temperature
      optimization.
    kl_constraints: KL and variables for applying Lagrangian penalties to bound
      them in the M-step, KLs are E* or [E*, A]. Here A is the action dimension
      in the case of per-dimension KL constraints.
    projection_operator: Function to project dual variables (temperature and kl
      constraint alphas) into the positive range.
    restarting_weights: Optional restarting weights, shape E*, 0 means that this
      step is the start of a new episode and we ignore losses at this step
      because the agent cannot influence these.
    importance_weights: Optional importance weights, shape E*.
    top_k_fraction: Fraction of samples to use in the E-step.
    policy_loss_weight: Weight for the policy loss.
    temperature_loss_weight: Weight for the temperature loss.
    kl_loss_weight: Weight for the KL loss.
    alpha_loss_weight: Weight for the alpha loss.
    axis_name: Optional axis name for `pmap`. If `None`, computations
      are performed locally on each device.
    use_stop_gradient: bool indicating whether or not to apply stop gradient.

  Returns:
    Per example `loss` with same shape E* as array inputs, and additional data
    including the components of this loss and the normalized weights in the
    AdditionalOutputs.
  """
  # Define default restarting weights and importance weights.
  if restarting_weights is None:
    restarting_weights = jnp.ones_like(sample_log_probs)
  if importance_weights is None:
    importance_weights = jnp.ones_like(sample_log_probs)

  # Check shapes.
  chex.assert_equal_shape(
      [advantages, sample_log_probs, restarting_weights, importance_weights])

  chex.assert_rank(temperature_constraint.epsilon, 0)
  chex.assert_type([
      sample_log_probs, advantages, restarting_weights, importance_weights,
      temperature_constraint.alpha, temperature_constraint.epsilon], float)

  for kl, penalty in kl_constraints:
    chex.assert_rank(penalty.epsilon, 0)
    chex.assert_type([kl, penalty.alpha, penalty.epsilon], float)
    if penalty.per_dimension:
      chex.assert_rank(kl, advantages.ndim + 1)
      chex.assert_equal_shape_prefix([kl, advantages], advantages.ndim)
    else:
      chex.assert_equal_shape([kl, advantages])

  # E-step: Calculate the reweighting and the temperature loss.
  temperature_loss, norm_weights, num_samples = (
      vmpo_compute_weights_and_temperature_loss(
          advantages, restarting_weights, importance_weights,
          temperature_constraint, projection_operator, top_k_fraction,
          axis_name=axis_name, use_stop_gradient=use_stop_gradient))

  # M-step: Supervised learning of reweighted trajectories using the weights
  # from the E-step, with additional KL constraints.
  # The weights are normalized so that the sum is 1. We multiply by the number
  # of examples so that we can give a policy loss per example and take the mean,
  # and we assume `restarting_weights` are already included.
  if axis_name:
    num_examples = jax.lax.all_gather(
        sample_log_probs, axis_name=axis_name).size
  else:
    num_examples = sample_log_probs.size
  policy_loss = -sample_log_probs * norm_weights * num_examples

  kl_loss, alpha_loss = compute_parametric_kl_penalty_and_dual_loss(
      kl_constraints, projection_operator, use_stop_gradient)

  chex.assert_equal_shape([policy_loss, kl_loss, alpha_loss])

  # Calculate the total policy improvement loss.
  loss = (policy_loss_weight * policy_loss +
          temperature_loss_weight * temperature_loss +
          kl_loss_weight * kl_loss +
          alpha_loss_weight * alpha_loss)

  return loss, MpoOutputs(
      temperature_loss=temperature_loss, policy_loss=policy_loss,
      kl_loss=kl_loss, alpha_loss=alpha_loss, normalized_weights=norm_weights,
      num_samples=num_samples)


def get_top_k_weights(
    top_k_fraction: float,
    restarting_weights: Array,
    scaled_advantages: Array,
    axis_name: Optional[str] = None,
    use_stop_gradient: bool = True,
):
  """Get the weights for the top top_k_fraction of advantages.

  Args:
    top_k_fraction: The fraction of weights to use.
    restarting_weights: Restarting weights, shape E*, 0 means that this step is
      the start of a new episode and we ignore losses at this step because the
      agent cannot influence these.
    scaled_advantages: The advantages for each example (shape E*), scaled by
      temperature.
    axis_name: Optional axis name for `pmap`. If `None`, computations are
      performed locally on each device.
    use_stop_gradient: bool indicating whether or not to apply stop gradient.

  Returns:
    Weights for the top top_k_fraction of advantages
  """
  chex.assert_equal_shape([scaled_advantages, restarting_weights])
  chex.assert_type([scaled_advantages, restarting_weights], float)

  if not 0.0 < top_k_fraction <= 1.0:
    raise ValueError(
        f"`top_k_fraction` must be in (0, 1], got {top_k_fraction}")
  logging.info("[vmpo_e_step] top_k_fraction: %f", top_k_fraction)

  if top_k_fraction < 1.0:
    # Don't include the restarting samples in the determination of top-k.
    valid_scaled_advantages = scaled_advantages - (
        1.0 - restarting_weights) * _INFINITY
    # Determine the minimum top-k value across all devices,
    if axis_name:
      all_valid_scaled_advantages = jax.lax.all_gather(
          valid_scaled_advantages, axis_name=axis_name)
    else:
      all_valid_scaled_advantages = valid_scaled_advantages
    top_k = int(top_k_fraction * jnp.size(all_valid_scaled_advantages))
    if top_k == 0:
      raise ValueError(
          "top_k_fraction too low to get any valid scaled advantages.")
    # TODO(b/160450251): Use jnp.partition(all_valid_scaled_advantages, top_k)
    #   when this is implemented in jax.
    top_k_min = jnp.sort(jnp.reshape(all_valid_scaled_advantages, [-1]))[-top_k]
    # Fold the top-k into the restarting weights.
    top_k_weights = jnp.greater_equal(valid_scaled_advantages,
                                      top_k_min).astype(jnp.float32)
    top_k_weights = jax.lax.select(
        use_stop_gradient, jax.lax.stop_gradient(top_k_weights), top_k_weights)
    top_k_restarting_weights = restarting_weights * top_k_weights
  else:
    top_k_restarting_weights = restarting_weights

  return top_k_restarting_weights


def vmpo_compute_weights_and_temperature_loss(
    advantages: Array,
    restarting_weights: Array,
    importance_weights: Array,
    temperature_constraint: LagrangePenalty,
    projection_operator: Callable[[Numeric], Numeric],
    top_k_fraction: float,
    axis_name: Optional[str] = None,
    use_stop_gradient: bool = True,
) -> Tuple[Scalar, Array, Scalar]:
  """Computes the weights and temperature loss for V-MPO.

  Args:
    advantages: Advantages for the E-step. Shape E*.
    restarting_weights: Restarting weights, 0 means that this
      step is the start of a new episode and we ignore losses at this step
      because the agent cannot influence these. Shape E*.
    importance_weights: Optional importance weights. Shape E*
    temperature_constraint: Lagrange constraint for the E-step temperature
      optimization.
    projection_operator: Function to project dual variables (temperature and kl
      constraint alphas) into the positive range.
    top_k_fraction: Fraction of samples to use in the E-step.
    axis_name: Optional axis name for `pmap` or 'vmap'. If `None`, computations
      are performed locally on each device.
    use_stop_gradient: bool indicating whether or not to apply stop gradient.

  Returns:
    The temperature loss, normalized weights and number of samples used.
  """
  chex.assert_equal_shape([advantages, restarting_weights, importance_weights])
  chex.assert_rank(temperature_constraint.epsilon, 0)
  chex.assert_type([
      advantages, restarting_weights, importance_weights,
      temperature_constraint.alpha, temperature_constraint.epsilon], float)

  importance_weights = jax.lax.select(
      use_stop_gradient, jax.lax.stop_gradient(importance_weights),
      importance_weights)

  # Lagrange constraint.
  temperature = projection_operator(temperature_constraint.alpha)
  epsilon_temperature = temperature_constraint.epsilon

  # Scale the advantages.
  scaled_advantages = restarting_weights * advantages / temperature
  max_scaled_advantage = jnp.max(scaled_advantages)
  # If the axis_name is not None find the maximum across all devices.
  if axis_name:
    assert use_stop_gradient  # Cant differentiate through pmax.
    max_scaled_advantage = jax.lax.stop_gradient(max_scaled_advantage)
    max_scaled_advantage = jax.lax.pmax(
        max_scaled_advantage, axis_name=axis_name)
  else:
    max_scaled_advantage = jax.lax.select(
        use_stop_gradient, jax.lax.stop_gradient(max_scaled_advantage),
        max_scaled_advantage)
  # Maybe don't use all of the advantages.
  top_k_restarting_weights = get_top_k_weights(
      top_k_fraction, restarting_weights, scaled_advantages, axis_name,
      use_stop_gradient)

  all_sum = base.AllSum(axis_name)

  # Reweight the old trajectories.
  unnormalized_weights = (top_k_restarting_weights * importance_weights
                          * jnp.exp(scaled_advantages - max_scaled_advantage))
  # If the axis_name is not None these sums will be taken across all devices.
  sum_weights = all_sum(unnormalized_weights) + _EPSILON
  num_samples = all_sum(top_k_restarting_weights) + _EPSILON

  normalized_weights = unnormalized_weights / sum_weights

  # Calculate the temperature loss.
  log_mean_weights = (jnp.log(sum_weights) + max_scaled_advantage
                      - jnp.log(num_samples))
  temperature_loss = temperature * (epsilon_temperature + log_mean_weights)

  return temperature_loss, normalized_weights, num_samples


def kl_constraint_loss(
    kl: Array,
    penalty: LagrangePenalty,
    projection_operator: Callable[[Numeric], Numeric],
    use_stop_gradient: bool = True,
) -> Tuple[Array, Array, Array]:
  """Implements a hard KL constraint.

  The optimization proceeds in two phases. First, we optimize the weighting
  term `alpha` keeping the KL constant and then we optimize the KL keeping
  `alpha` constant. Each phase is implemented by appropriately using a
  stop_gradient.

  If `bound` - `kl` > 0, then `alpha` is pushed toward `min_alpha`. However
  this also means the policy is free to change more since `kl` < `bound `.
  This eventually leads to `bound` - `kl` < 0 which pressures alpha to get to
  a high positive value. This coordinate ascent results in `kl` staying close
  to `bound`.

  Args:
    kl: The kl per example which is being constrained.
    penalty: The dual variable used to impose a penalty and parameters of the
      constraint.
    projection_operator: Function to project dual variables kl constraint alphas
      into the positive range.
    use_stop_gradient: bool indicating whether or not to apply stop gradient.

  Returns:
    A `tuple` consisting of three arrays: `kl_loss`, `alpha_loss` and
    `clipped_alpha`. The first two terms represent the losses for the two phases
    of computation. `clipped_alpha` is the clipped lagrangian multiplier `alpha`
    that is learnt.
  """
  chex.assert_type([kl, penalty.alpha, penalty.epsilon], float)

  alpha = projection_operator(penalty.alpha)
  alpha_constant = jax.lax.select(
      use_stop_gradient, jax.lax.stop_gradient(penalty.alpha), penalty.alpha)

  # First step: Optimize w.r.t. alphas
  alpha_loss = alpha * (
      penalty.epsilon -
      jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(kl), kl))

  # Second step: KL loss.
  kl_loss = alpha_constant * kl

  # If kl_loss and alpha_loss are per dimension then at this point sum over
  # dimensions.
  if penalty.per_dimension:
    kl_loss = jnp.sum(kl_loss, axis=-1)
    alpha_loss = jnp.sum(alpha_loss, axis=-1)

  return kl_loss, alpha_loss, alpha


def kl_alpha_loss(
    restarting_weights: Array,
    kl_constraints: Sequence[Tuple[Array, LagrangePenalty]] = (),
    axis_name: Optional[str] = None):
  """Calculates the losses for multiple KL constraints.

  Args:
    restarting_weights: Restarting weights, shape E*, 0 means that this step is
      the start of a new episode and we ignore losses at this step because the
      agent cannot influence these.
    kl_constraints: KL and variables for applying Lagrangian penalties to bound
      them in the M-step, KLs are [E*, A?]. Here A is the action dimension
      in the case of per-dimension KL constraints.
    axis_name: Optional axis name for `pmap`. If `None`, computations are
      performed locally on each device.

  Returns:
    The kl loss and dual variable loss both shape E*.
  """
  chex.assert_type(restarting_weights, float)
  if kl_constraints:
    for kl, penalty in kl_constraints:
      chex.assert_rank(penalty.epsilon, 0)
      chex.assert_type([kl, penalty.alpha, penalty.epsilon], float)
      chex.assert_equal_shape_prefix([kl, restarting_weights],
                                     restarting_weights.ndim)

    # Implement decoupled KL constraints.
    kl_alpha_losses = [kl_constraint_loss(kl, penalty, lambda x: x)[:2]
                       for kl, penalty in kl_constraints]
    kl_loss, alpha_loss = [sum(losses) for losses in zip(*kl_alpha_losses)]
    all_sum = base.AllSum(axis_name)
    num_samples = all_sum(restarting_weights) + _EPSILON
    # Reshape in case KL is per dimension.
    kl_loss = all_sum(kl_loss * restarting_weights) / num_samples
    alpha_loss = all_sum(alpha_loss * restarting_weights) / num_samples
  else:
    # No M-step constraint.
    kl_loss = jnp.asarray(0.0)
    alpha_loss = jnp.asarray(0.0)
  return kl_loss, alpha_loss

