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
from rlax._src import base


ArrayLike = base.ArrayLike
VTraceOutput = collections.namedtuple(
    'vtrace_output', ['errors', 'pg_advantage', 'q_estimate'])


def vtrace(
    v_tm1: ArrayLike,
    v_t: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    rho_t: ArrayLike,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> ArrayLike:
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
    rho_t: importance sampling ratios.
    lambda_: scalar mixing parameter lambda.
    clip_rho_threshold: clip threshold for importance weights.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    V-Trace error.
  """
  chex.rank_assert([v_tm1, v_t, r_t, discount_t, rho_t], [1, 1, 1, 1, 1])
  chex.type_assert([v_tm1, v_t, r_t, discount_t, rho_t],
                   [float, float, float, float, float])

  # Clip importance sampling ratios.
  c_t = jnp.minimum(1.0, rho_t) * lambda_
  clipped_rhos = jnp.minimum(clip_rho_threshold, rho_t)

  # Compute the temporal difference errors.
  td_errors = clipped_rhos * (r_t + discount_t * v_t - v_tm1)

  # Work backwards computing the td-errors.
  err = 0.0
  errors = []
  for i in jnp.arange(v_t.shape[0] - 1, -1, -1):
    err = td_errors[i] + discount_t[i] * c_t[i] * err
    errors.insert(0, err)

  # Return errors.
  if not stop_target_gradients:
    return jnp.array(errors)
  # In TD-like algorithms, we want gradients to only flow in the predictions,
  # and not in the values used to bootstrap. In this case, add the value of the
  # initial state value to get the implied estimates of the returns, stop
  # gradient around such target and then subtract again the initial state value.
  else:
    target_tm1 = jnp.array(errors) + v_tm1
    target_tm1 = jax.lax.stop_gradient(target_tm1)
  return target_tm1 - v_tm1


def leaky_vtrace(
    v_tm1: ArrayLike,
    v_t: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    rho_t: ArrayLike,
    alpha_: float = 1.0,
    lambda_: float = 1.0,
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
    rho_t: importance weights at time t.
    alpha_: mixing parameter for Importance Sampling and V-trace.
    lambda_: scalar mixing parameter lambda.
    clip_rho_threshold: clip threshold for importance weights.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    Leaky V-Trace error.
  """
  chex.rank_assert([v_tm1, v_t, r_t, discount_t, rho_t], [1, 1, 1, 1, 1])
  chex.type_assert([v_tm1, v_t, r_t, discount_t, rho_t],
                   [float, float, float, float, float])

  # Mix clipped and unclipped importance sampling ratios.
  c_t = (
      (1 - alpha_) * rho_t + alpha_ * jnp.minimum(1.0, rho_t)) * lambda_
  clipped_rhos = (
      (1 - alpha_) * rho_t + alpha_ * jnp.minimum(clip_rho_threshold, rho_t))

  # Compute the temporal difference errors.
  td_errors = clipped_rhos * (r_t + discount_t * v_t - v_tm1)

  # Work backwards computing the td-errors.
  err = 0.0
  errors = []
  for i in jnp.arange(v_t.shape[0] - 1, -1, -1):
    err = td_errors[i] + discount_t[i] * c_t[i] * err
    errors.insert(0, err)

  # Return errors.
  if not stop_target_gradients:
    return jnp.array(errors)
  # In TD-like algorithms, we want gradients to only flow in the predictions,
  # and not in the values used to bootstrap. In this case, add the value of the
  # initial state value to get the implied estimates of the returns, stop
  # gradient around such target and then subtract again the initial state value.
  else:
    target_tm1 = jnp.array(errors) + v_tm1
    return jax.lax.stop_gradient(target_tm1) - v_tm1


def vtrace_td_error_and_advantage(
    v_tm1: ArrayLike,
    v_t: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    rho_t: ArrayLike,
    lambda_: float = 1.0,
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
    rho_t: importance weights at time t.
    lambda_: scalar mixing parameter lambda.
    clip_rho_threshold: clip threshold for importance ratios.
    clip_pg_rho_threshold: clip threshold for policy gradient importance ratios.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    a tuple of V-Trace error, policy gradient advantage, and estimated Q-values.
  """
  chex.rank_assert([v_tm1, v_t, r_t, discount_t, rho_t], 1)
  chex.type_assert([v_tm1, v_t, r_t, discount_t, rho_t], float)

  errors = vtrace(
      v_tm1, v_t, r_t, discount_t, rho_t,
      lambda_, clip_rho_threshold, stop_target_gradients)
  targets_tm1 = errors + v_tm1
  q_bootstrap = jnp.concatenate([
      lambda_ * targets_tm1[1:] + (1 - lambda_) * v_tm1[1:],
      v_t[-1:],
  ], axis=0)
  q_estimate = r_t + discount_t * q_bootstrap
  clipped_pg_rho_tm1 = jnp.minimum(clip_pg_rho_threshold, rho_t)
  pg_advantages = clipped_pg_rho_tm1 * (q_estimate - v_tm1)
  return VTraceOutput(
      errors=errors, pg_advantage=pg_advantages, q_estimate=q_estimate)
