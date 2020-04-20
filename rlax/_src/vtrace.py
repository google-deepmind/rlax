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
import jax.numpy as jnp
from rlax._src import base
from rlax._src import value_learning

ArrayLike = base.ArrayLike
VTraceOutput = collections.namedtuple(
    'vtrace_output', ['errors', 'pg_advantage', 'q_estimate'])


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
  base.rank_assert([v_tm1, v_t, r_t, discount_t, rho_t], 1)
  base.type_assert([v_tm1, v_t, r_t, discount_t, rho_t], float)

  errors = value_learning.vtrace(
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
