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
"""JAX functions for calculating multistep returns.

In this subpackage we expose a number of functions that may be used to compute
multistep truncated bootstrapped estimates of the return (the discounted sum of
rewards collected by an agent). These estimate compute returns from trajectories
of experience; trajectories are not assumed to align with episode boundaries,
and bootstrapping is used to estimate returns beyond the end of a trajectory.
"""

import chex
import jax.numpy as jnp
from rlax._src import base

Array = chex.Array
Numeric = chex.Numeric


def lambda_returns(
    r_t: Array,
    discount_t: Array,
    v_t: Array,
    lambda_: Numeric = 1.,
) -> Array:
  """Estimates a multistep truncated lambda return from a trajectory.

  Given a a trajectory of length `T+1`, generated under some policy π, for each
  time-step `t` we can estimate a target return `G_t`, by combining rewards,
  discounts, and state values, according to a mixing parameter `lambda`.

  The parameter `lambda_`  mixes the different multi-step bootstrapped returns,
  corresponding to accumulating `k` rewards and then bootstrapping using `v_t`.

    rₜ₊₁ + γₜ₊₁ vₜ₊₁
    rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ vₜ₊₂
    rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ rₜ₊₂ + γₜ₊₁ γₜ₊₂ γₜ₊₃ vₜ₊₃

  The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

    Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

  In the `on-policy` case, we estimate a return target `G_t` for the same
  policy π that was used to generate the trajectory. In this setting the
  parameter `lambda_` is typically a fixed scalar factor. Depending
  on how values `v_t` are computed, this function can be used to construct
  targets for different multistep reinforcement learning updates:

    TD(λ):  `v_t` contains the state value estimates for each state under π.
    Q(λ):  `v_t = max(q_t, axis=-1)`, where `q_t` estimates the action values.
    Sarsa(λ):  `v_t = q_t[..., a_t]`, where `q_t` estimates the action values.

  In the `off-policy` case, the mixing factor is a function of state, and
  different definitions of `lambda` implement different off-policy corrections:

    Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
    V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)

  Note that the second option is equivalent to applying per-decision importance
  sampling, but using an adaptive λ(ρₜ) = min(1/ρₜ, 1), such that the effective
  bootstrap parameter at time t becomes λₜ = λ(ρₜ) * ρₜ = min(1, ρₜ).
  This is the interpretation used in the ABQ(ζ) algorithm (Mahmood 2017).

  Of course this can be augmented to include an additional factor λ.  For
  instance we could use V-trace with a fixed additional parameter λ = 0.9, by
  setting λₜ = 0.9 * min(1, ρₜ) or, alternatively (but not equivalently),
  λₜ = min(0.9, ρₜ).

  Estimated return are then often used to define a td error, e.g.:  ρₜ(Gₜ - vₜ).

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/sutton/book/ebook/node74.html).

  Args:
    r_t: sequence of rewards rₜ for timesteps t in [1, T].
    discount_t: sequence of discounts γₜ for timesteps t in [1, T].
    v_t: sequence of state values estimates under π for timesteps t in [1, T].
    lambda_: mixing parameter; a scalar or a vector for timesteps t in [1, T].

  Returns:
    Multistep lambda returns.
  """
  chex.rank_assert([r_t, discount_t, v_t, lambda_], [1, 1, 1, [0, 1]])
  chex.type_assert([r_t, discount_t, v_t, lambda_], float)

  # If scalar make into vector.
  lambda_ = jnp.ones_like(discount_t) * lambda_

  # Work backwards to compute `G_{T-1}`, ..., `G_0`.
  returns = []
  g = v_t[-1]
  for i in jnp.arange(v_t.shape[0] - 1, -1, -1):
    g = r_t[i] + discount_t[i] * ((1-lambda_[i]) * v_t[i] + lambda_[i] * g)
    returns.insert(0, g)

  return jnp.array(returns)


def n_step_bootstrapped_returns(
    r_t: Array,
    discount_t: Array,
    v_t: Array,
    n: int
) -> Array:
  """Computes strided n-step bootstrapped return targets over a sequence.

  The returns are computed in a backwards fashion according to the equation:

     Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))),

  Args:
    r_t: rewards at times [1, ..., T].
    discount_t: discounts at times [1, ..., T].
    v_t: state or state-action values to bootstrap from at time [1, ...., T]
    n: number of steps over which to accumulate reward before bootstrapping.

  Returns:
    estimated bootstrapped returns at times [1, ...., T]
  """
  chex.rank_assert([r_t, discount_t, v_t], 1)
  chex.type_assert([r_t, discount_t, v_t], float)
  seq_len = r_t.shape[0]

  # Pad end of reward and discount sequences with 0 and 1 respectively.
  r_t = jnp.concatenate([r_t, jnp.zeros(n - 1)])
  discount_t = jnp.concatenate([discount_t, jnp.ones(n - 1)])

  # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
  pad_size = min(n - 1, seq_len)
  targets = jnp.concatenate([v_t[n - 1:], jnp.array([v_t[-1]] * pad_size)])

  # Work backwards to compute discounted, bootstrapped n-step returns.
  for i in reversed(range(n)):
    targets = r_t[i:i + seq_len] + discount_t[i:i + seq_len] * targets
  return targets


def discounted_returns(
    r_t: Array,
    discount_t: Array,
    v_t: Array
) -> Array:
  """Calculates a discounted return from a trajectory.

  The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

    Gₜ = rₜ₊₁ + γₜ₊₁ Gₜ₊₁.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/sutton/book/ebook/node61.html).

  Args:
    r_t: reward sequence at time t.
    discount_t: discount sequence at time t.
    v_t: value sequence or scalar at time t.

  Returns:
    Discounted returns.
  """
  chex.rank_assert([r_t, discount_t, v_t], [1, 1, [0, 1]])
  chex.type_assert([r_t, discount_t, v_t], float)

  # If scalar make into vector.
  bootstrapped_v = jnp.ones_like(discount_t) * v_t
  return lambda_returns(r_t, discount_t, bootstrapped_v, lambda_=1.)


def importance_corrected_td_errors(
    r_t: Array,
    discount_t: Array,
    rho_tm1: Array,
    lambda_: Array,
    values: Array,
) -> Array:
  """Computes the multistep td errors with per decision importance sampling.

  Given a trajectory of length `T+1`, generated under some policy π, for each
  time-step `t` we can estimate a multistep temporal difference error δₜ(ρ,λ),
  by combining rewards, discounts, and state values, according to a mixing
  parameter `λ` and importance sampling ratios ρₜ = π(aₜ|sₜ) / μ(aₜ|sₜ):

    td-errorₜ = ρₜ δₜ(ρ,λ)
    δₜ(ρ,λ) = δₜ + ρₜ₊₁ λₜ₊₁ γₜ₊₁ δₜ₊₁(ρ,λ),

  where δₜ = rₜ₊₁ + γₜ₊₁ vₜ₊₁ - vₜ is the one step, temporal difference error
  for the agent's state value estimates. This is equivalent to computing
  the λ-return with λₜ = ρₜ (e.g. using the `lambda_returns` function from
  above), and then computing errors as  td-errorₜ = ρₜ(Gₜ - vₜ).

  See "A new Q(λ) with interim forward view and Monte Carlo equivalence"
  by Sutton et al. (http://proceedings.mlr.press/v32/sutton14.html).

  Args:
    r_t: sequence of rewards rₜ for timesteps t in [1, T].
    discount_t: sequence of discounts γₜ for timesteps t in [1, T].
    rho_tm1: sequence of importance ratios for all timesteps t in [0, T-1].
    lambda_: mixing parameter; scalar or have per timestep values in [1, T].
    values: sequence of state values under π for all timesteps t in [0, T].

  Returns:
    Off-policy estimates of the multistep lambda returns from each state.
  """
  chex.rank_assert([r_t, discount_t, rho_tm1, values], [1, 1, 1, 1])
  chex.type_assert([r_t, discount_t, rho_tm1, values], float)

  v_tm1 = values[:-1]  # Predictions to compute errors for.
  v_t = values[1:]  # Values for bootstrapping.
  rho_t = jnp.concatenate((rho_tm1[1:], jnp.array([1.])))  # Unused dummy value.
  lambda_ = jnp.ones_like(discount_t) * lambda_  # If scalar, make into vector.

  # Compute the one step temporal difference errors.
  one_step_delta = r_t + discount_t * v_t - v_tm1

  # Work backwards to compute `delta_{T-1}`, ..., `delta_0`.
  delta, errors = 0.0, []
  for i in jnp.arange(one_step_delta.shape[0] - 1, -1, -1):
    delta = one_step_delta[i] + discount_t[i] * rho_t[i] * lambda_[i] * delta
    errors.insert(0, delta)

  return rho_tm1 * jnp.array(errors)


def general_off_policy_returns_from_action_values(
    q_t: Array,
    a_t: Array,
    r_t: Array,
    discount_t: Array,
    c_t: Array,
    pi_t: Array,
) -> Array:
  """Calculates errors for various off-policy correction algorithms.

  Given a window of experience of length `K`, generated by a behaviour policy μ,
  for each time-step `t` we can estimate the return `G_t` from that step
  onwards, under some target policy π, using the rewards in the trajectory, the
  actions selected by μ and the action-values under π, according to equation:

    Gₜ = rₜ₊₁ + γₜ₊₁ * (E[q(aₜ₊₁)] - cₜ * q(aₜ₊₁) + cₜ * Gₜ₊₁),

  where, depending on the choice of `c_t`, the algorithm implements:

    Importance Sampling             c_t = π(x_t, a_t) / μ(x_t, a_t),
    Harutyunyan's et al. Q(lambda)  c_t = λ,
    Precup's et al. Tree-Backup     c_t = π(x_t, a_t),
    Munos' et al. Retrace           c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).

  See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
  (https://arxiv.org/abs/1606.02647).

  Args:
    q_t: Q-values at time t.
    a_t: action index at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    c_t: importance weights at time t.
    pi_t: target policy probs at time t.

  Returns:
    Off-policy estimates of the multistep lambda returns from each state..
  """
  chex.rank_assert([q_t, a_t, r_t, discount_t, c_t, pi_t], [2, 1, 1, 1, 1, 2])
  chex.type_assert([q_t, a_t, r_t, discount_t, c_t, pi_t],
                   [float, int, float, float, float, float])

  # Get the expected values and the values of actually selected actions.
  exp_q_t = (pi_t * q_t).sum(axis=-1)
  # The generalized returns are independent of Q-values and cs at the final
  # state.
  q_a_t = base.batched_index(q_t, a_t)[:-1]
  c_t = c_t[:-1]

  return general_off_policy_returns_from_q_and_v(
      q_a_t, exp_q_t, r_t, discount_t, c_t)


def general_off_policy_returns_from_q_and_v(
    q_t: Array,
    v_t: Array,
    r_t: Array,
    discount_t: Array,
    c_t: Array,
) -> Array:
  """Calculates targets for various off-policy evaluation algorithms.

  Given a window of experience of length `K+1`, generated by a behaviour policy
  μ, for each time-step `t` we can estimate the return `G_t` from that step
  onwards, under some target policy π, using the rewards in the trajectory, the
  values under π of states and actions selected by μ, according to equation:

    Gₜ = rₜ₊₁ + γₜ₊₁ * (vₜ₊₁ - cₜ₊₁ * q(aₜ₊₁) + cₜ₊₁* Gₜ₊₁),

  where, depending on the choice of `c_t`, the algorithm implements:

    Importance Sampling             c_t = π(x_t, a_t) / μ(x_t, a_t),
    Harutyunyan's et al. Q(lambda)  c_t = λ,
    Precup's et al. Tree-Backup     c_t = π(x_t, a_t),
    Munos' et al. Retrace           c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).

  See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
  (https://arxiv.org/abs/1606.02647).

  Args:
    q_t: Q-values under π of actions executed by μ at times [1, ..., K - 1].
    v_t: Values under π at times [1, ..., K].
    r_t: rewards at times [1, ..., K].
    discount_t: discounts at times [1, ..., K].
    c_t: weights at times [1, ..., K - 1].

  Returns:
    Off-policy estimates of the generalized returns from states visited at times
    [0, ..., K - 1].
  """
  chex.rank_assert([q_t, v_t, r_t, discount_t, c_t], 1)
  chex.type_assert([q_t, v_t, r_t, discount_t, c_t], float)

  # Work backwards to compute `G_K-1`, ..., `G_1`, `G_0`.
  g = r_t[-1] + discount_t[-1] * v_t[-1]  # G_K-1.
  returns = [g]
  for i in jnp.arange(q_t.shape[0] - 1, -1, -1):  # [K - 2, ..., 0]
    g = r_t[i] + discount_t[i] * (v_t[i] - c_t[i] * q_t[i] + c_t[i] * g)
    returns.insert(0, g)

  return jnp.array(returns)

