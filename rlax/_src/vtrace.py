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

"""JAX functions for V-Trace algorithm.

V-Trace is a form of importance sampling correction that was introduced by
Espeholt et al. in the context of an off-policy actor-critic agent (IMPALA).
This subpackage implements the specific targets used in IMPALA to implement
both the value and the policy. Note however that the V-Trace return estimate is
a special case of the multistep return estimates from `multistep.py`.
"""

import collections

import chex
import jax
import jax.numpy as jnp


Array = chex.Array
Numeric = chex.Numeric
VTraceOutput = collections.namedtuple(
    'vtrace_output', ['errors', 'pg_advantage', 'q_estimate'])


def vtrace(
    v_tm1: Array,
    v_t: Array,
    r_t: Array,
    discount_t: Array,
    rho_tm1: Array,
    lambda_: Numeric = 1.0,
    clip_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> Array:
  """Calculates V-Trace errors from importance weights.

  V-trace computes TD-errors from multistep trajectories by applying
  off-policy corrections based on clipped importance sampling ratios.

  See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
  Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561).

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_tm1: importance sampling ratios at time t-1.
    lambda_: mixing parameter; a scalar or a vector for timesteps t.
    clip_rho_threshold: clip threshold for importance weights.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    V-Trace error.
  """
  chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [1, 1, 1, 1, 1, {0, 1}])
  chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [float, float, float, float, float, float])
  chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

  # Clip importance sampling ratios.
  c_tm1 = jnp.minimum(1.0, rho_tm1) * lambda_
  clipped_rhos_tm1 = jnp.minimum(clip_rho_threshold, rho_tm1)

  # Compute the temporal difference errors.
  td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

  # Work backwards computing the td-errors.
  err = 0.0
  errors = []
  for i in reversed(range(v_t.shape[0])):
    err = td_errors[i] + discount_t[i] * c_tm1[i] * err
    errors.insert(0, err)

  # Return errors, maybe disabling gradient flow through bootstrap targets.
  return jax.lax.select(
      stop_target_gradients,
      jax.lax.stop_gradient(jnp.array(errors) + v_tm1) - v_tm1,
      jnp.array(errors))


def leaky_vtrace(
    v_tm1: Array,
    v_t: Array,
    r_t: Array,
    discount_t: Array,
    rho_tm1: Array,
    alpha_: float = 1.0,
    lambda_: Numeric = 1.0,
    clip_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True):
  """Calculates Leaky V-Trace errors from importance weights.

  Leaky-Vtrace is a combination of Importance sampling and V-trace, where the
  degree of mixing is controlled by a scalar `alpha` (that may be meta-learnt).

  See "Self-Tuning Deep Reinforcement Learning"
  by Zahavy et al. (https://arxiv.org/abs/2002.12928)

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_tm1: importance weights at time t-1.
    alpha_: mixing parameter for Importance Sampling and V-trace.
    lambda_: mixing parameter; a scalar or a vector for timesteps t.
    clip_rho_threshold: clip threshold for importance weights.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    Leaky V-Trace error.
  """
  chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [1, 1, 1, 1, 1, {0, 1}])
  chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [float, float, float, float, float, float])
  chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

  # Mix clipped and unclipped importance sampling ratios.
  c_tm1 = (
      (1 - alpha_) * rho_tm1 + alpha_ * jnp.minimum(1.0, rho_tm1)) * lambda_
  clipped_rhos_tm1 = (
      (1 - alpha_) * rho_tm1 + alpha_ * jnp.minimum(clip_rho_threshold, rho_tm1)
  )

  # Compute the temporal difference errors.
  td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

  # Work backwards computing the td-errors.
  err = 0.0
  errors = []
  for i in reversed(range(v_t.shape[0])):
    err = td_errors[i] + discount_t[i] * c_tm1[i] * err
    errors.insert(0, err)

  # Return errors, maybe disabling gradient flow through bootstrap targets.
  return jax.lax.select(
      stop_target_gradients,
      jax.lax.stop_gradient(jnp.array(errors) + v_tm1) - v_tm1,
      jnp.array(errors))


def vtrace_td_error_and_advantage(
    v_tm1: Array,
    v_t: Array,
    r_t: Array,
    discount_t: Array,
    rho_tm1: Array,
    lambda_: Numeric = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> VTraceOutput:
  """Calculates V-Trace errors and PG advantage from importance weights.

  This functions computes the TD-errors and policy gradient Advantage terms
  as used by the IMPALA distributed actor-critic agent.

  See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
  Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561)

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_tm1: importance weights at time t-1.
    lambda_: mixing parameter; a scalar or a vector for timesteps t.
    clip_rho_threshold: clip threshold for importance ratios.
    clip_pg_rho_threshold: clip threshold for policy gradient importance ratios.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    a tuple of V-Trace error, policy gradient advantage, and estimated Q-values.
  """
  chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [1, 1, 1, 1, 1, {0, 1}])
  chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [float, float, float, float, float, float])
  chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

  # If scalar make into vector.
  lambda_ = jnp.ones_like(discount_t) * lambda_

  errors = vtrace(
      v_tm1, v_t, r_t, discount_t, rho_tm1,
      lambda_, clip_rho_threshold, stop_target_gradients)
  targets_tm1 = errors + v_tm1
  q_bootstrap = jnp.concatenate([
      lambda_[:-1] * targets_tm1[1:] + (1 - lambda_[:-1]) * v_tm1[1:],
      v_t[-1:],
  ], axis=0)
  q_estimate = r_t + discount_t * q_bootstrap
  clipped_pg_rho_tm1 = jnp.minimum(clip_pg_rho_threshold, rho_tm1)
  pg_advantages = clipped_pg_rho_tm1 * (q_estimate - v_tm1)
  return VTraceOutput(
      errors=errors, pg_advantage=pg_advantages, q_estimate=q_estimate)


def leaky_vtrace_td_error_and_advantage(
    v_tm1: chex.Array,
    v_t: chex.Array,
    r_t: chex.Array,
    discount_t: chex.Array,
    rho_tm1: chex.Array,
    alpha: float = 1.0,
    lambda_: Numeric = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> VTraceOutput:
  """Calculates Leaky V-Trace errors and PG advantage from importance weights.

  This functions computes the Leaky V-Trace TD-errors and policy gradient
  Advantage terms as used by the IMPALA distributed actor-critic agent.

  Leaky-Vtrace is a combination of Importance sampling and V-trace, where the
  degree of mixing is controlled by a scalar `alpha` (that may be meta-learnt).

  See "Self-Tuning Deep Reinforcement Learning"
  by Zahavy et al. (https://arxiv.org/abs/2002.12928) and
  "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
  Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561)

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_tm1: importance weights at time t-1.
    alpha: mixing the clipped importance sampling weights with unclipped ones.
    lambda_: mixing parameter; a scalar or a vector for timesteps t.
    clip_rho_threshold: clip threshold for importance ratios.
    clip_pg_rho_threshold: clip threshold for policy gradient importance ratios.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    a tuple of V-Trace error, policy gradient advantage, and estimated Q-values.
  """
  chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [1, 1, 1, 1, 1, {0, 1}])
  chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                   [float, float, float, float, float, float])
  chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

  # If scalar make into vector.
  lambda_ = jnp.ones_like(discount_t) * lambda_

  errors = leaky_vtrace(
      v_tm1, v_t, r_t, discount_t, rho_tm1, alpha,
      lambda_, clip_rho_threshold, stop_target_gradients)
  targets_tm1 = errors + v_tm1
  q_bootstrap = jnp.concatenate([
      lambda_[:-1] * targets_tm1[1:] + (1 - lambda_[:-1]) * v_tm1[1:],
      v_t[-1:],
  ], axis=0)
  q_estimate = r_t + discount_t * q_bootstrap
  clipped_pg_rho_tm1 = ((1 - alpha) * rho_tm1 + alpha *
                        jnp.minimum(clip_pg_rho_threshold, rho_tm1))
  pg_advantages = clipped_pg_rho_tm1 * (q_estimate - v_tm1)
  return VTraceOutput(
      errors=errors, pg_advantage=pg_advantages, q_estimate=q_estimate)
