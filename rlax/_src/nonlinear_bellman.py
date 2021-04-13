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
"""Transformed value functions.

Canonical value functions map states onto the expected discounted sum of rewards
that may be collected by an agent from any starting state. Value functions may
also be defined as the fixed points of certain linear recursive relations known
as Bellman equations. It is sometimes useful to consider transformed values that
are the solution to non-linear generalization of traditional Bellman equations.
In this subpackage we provide a general utility for wrapping bootstrapped return
calculations to construct regression targets for these transformed values.
We also use this to implement different learning algorithms from the literature.
"""

import collections
import functools

import chex
import jax.numpy as jnp
from rlax._src import base
from rlax._src import multistep
from rlax._src import transforms

Array = chex.Array
TxPair = collections.namedtuple('TxPair', ['apply', 'apply_inv'])


# Example transform pairs; these typically consist of a monotonically increasing
# squashing fn `apply` and its inverse `apply_inv`. Other choices are possible.

IDENTITY_PAIR = TxPair(
    transforms.identity, transforms.identity)
SIGNED_LOGP1_PAIR = TxPair(
    transforms.signed_logp1, transforms.signed_expm1)
SIGNED_HYPERBOLIC_PAIR = TxPair(
    transforms.signed_hyperbolic, transforms.signed_parabolic)
HYPERBOLIC_SIN_PAIR = TxPair(
    transforms.hyperbolic_arcsin, transforms.hyperbolic_sin)


def transform_values(build_targets, *value_argnums):
  """Decorator to convert targets to use transformed value function."""
  @functools.wraps(build_targets)
  def wrapped_build_targets(tx_pair, *args, **kwargs):
    tx_args = list(args)
    for index in value_argnums:
      tx_args[index] = tx_pair.apply_inv(tx_args[index])

    targets = build_targets(*tx_args, **kwargs)
    return tx_pair.apply(targets)

  return wrapped_build_targets


transformed_lambda_returns = transform_values(multistep.lambda_returns, 2)
transformed_general_off_policy_returns_from_action_values = transform_values(
    multistep.general_off_policy_returns_from_action_values, 0)
transformed_n_step_returns = transform_values(
    multistep.n_step_bootstrapped_returns, 2)


def transformed_q_lambda(
    q_tm1: Array,
    a_tm1: Array,
    r_t: Array,
    discount_t: Array,
    q_t: Array,
    lambda_: Array,
    stop_target_gradients: bool = True,
    tx_pair: TxPair = IDENTITY_PAIR,
) -> Array:
  """Calculates Peng's or Watkins' Q(lambda) temporal difference error.

  See "General non-linear Bellman equations" by van Hasselt et al.
  (https://arxiv.org/abs/1907.03687).

  Args:
    q_tm1: sequence of Q-values at time t-1.
    a_tm1: sequence of action indices at time t-1.
    r_t: sequence of rewards at time t.
    discount_t: sequence of discounts at time t.
    q_t: sequence of Q-values at time t.
    lambda_: mixing parameter lambda, either a scalar (e.g. Peng's Q(lambda)) or
      a sequence (e.g. Watkin's Q(lambda)).
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
    tx_pair: TxPair of value function transformation and its inverse.

  Returns:
    Q(lambda) temporal difference error.
  """
  chex.assert_rank([q_tm1, a_tm1, r_t, discount_t, q_t, lambda_],
                   [2, 1, 1, 1, 2, {0, 1}])
  chex.assert_type([q_tm1, a_tm1, r_t, discount_t, q_t, lambda_],
                   [float, int, float, float, float, float])

  qa_tm1 = base.batched_index(q_tm1, a_tm1)
  v_t = jnp.max(q_t, axis=-1)
  target_tm1 = transformed_lambda_returns(
      tx_pair, r_t, discount_t, v_t, lambda_, stop_target_gradients)
  return target_tm1 - qa_tm1


def transformed_retrace(
    q_tm1: Array,
    q_t: Array,
    a_tm1: Array,
    a_t: Array,
    r_t: Array,
    discount_t: Array,
    pi_t: Array,
    mu_t: Array,
    lambda_: float,
    eps: float = 1e-8,
    stop_target_gradients: bool = True,
    tx_pair: TxPair = IDENTITY_PAIR,
) -> Array:
  """Calculates transformed Retrace errors.

  See "Recurrent Experience Replay in Distributed Reinforcement Learning" by
  Kapturowski et al. (https://openreview.net/pdf?id=r1lyTjAqYX).

  Args:
    q_tm1: Q-values at time t-1.
    q_t: Q-values at time t.
    a_tm1: action index at time t-1.
    a_t: action index at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    pi_t: target policy probs at time t.
    mu_t: behavior policy probs at time t.
    lambda_: scalar mixing parameter lambda.
    eps: small value to add to mu_t for numerical stability.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
    tx_pair: TxPair of value function transformation and its inverse.

  Returns:
    Transformed Retrace error.
  """
  chex.assert_rank([q_tm1, q_t, a_tm1, a_t, r_t, discount_t, pi_t, mu_t],
                   [2, 2, 1, 1, 1, 1, 2, 1])
  chex.assert_type([q_tm1, q_t, a_tm1, a_t, r_t, discount_t, pi_t, mu_t],
                   [float, float, int, int, float, float, float, float])

  pi_a_t = base.batched_index(pi_t, a_t)
  c_t = jnp.minimum(1.0, pi_a_t / (mu_t + eps)) * lambda_
  target_tm1 = transformed_general_off_policy_returns_from_action_values(
      tx_pair, q_t, a_t, r_t, discount_t, c_t, pi_t, stop_target_gradients)
  q_a_tm1 = base.batched_index(q_tm1, a_tm1)
  return target_tm1 - q_a_tm1


def transformed_n_step_q_learning(
    q_tm1: Array,
    a_tm1: Array,
    target_q_t: Array,
    a_t: Array,
    r_t: Array,
    discount_t: Array,
    n: int,
    stop_target_gradients: bool = True,
    tx_pair: TxPair = IDENTITY_PAIR,
) -> Array:
  """Calculates transformed n-step TD errors.

  See "Recurrent Experience Replay in Distributed Reinforcement Learning" by
  Kapturowski et al. (https://openreview.net/pdf?id=r1lyTjAqYX).

  Args:
    q_tm1: Q-values at times [0, ..., T - 1].
    a_tm1: action index at times [0, ..., T - 1].
    target_q_t: target Q-values at time [1, ..., T].
    a_t: action index at times [[1, ... , T]] used to select target q-values to
      bootstrap from; max(target_q_t) for normal Q-learning, max(q_t) for double
      Q-learning.
    r_t: reward at times [1, ..., T].
    discount_t: discount at times [1, ..., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
    tx_pair: TxPair of value function transformation and its inverse.

  Returns:
    Transformed N-step TD error.
  """
  chex.assert_rank([q_tm1, target_q_t, a_tm1, a_t, r_t, discount_t],
                   [2, 2, 1, 1, 1, 1])
  chex.assert_type([q_tm1, target_q_t, a_tm1, a_t, r_t, discount_t],
                   [float, float, int, int, float, float])

  v_t = base.batched_index(target_q_t, a_t)
  target_tm1 = transformed_n_step_returns(
      tx_pair, r_t, discount_t, v_t, n,
      stop_target_gradients=stop_target_gradients)
  q_a_tm1 = base.batched_index(q_tm1, a_tm1)
  return target_tm1 - q_a_tm1
