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
"""JAX functions for common discrete-action value and action-value learning.

These functions define learning rules for discrete, scalar, action spaces.
Actions must be represented as indices in the range `[0, A)` where `A` is the
number of distinct actions.
"""

import jax
import jax.numpy as jnp
from rlax._src import base
from rlax._src import clipping
from rlax._src import distributions
from rlax._src import multistep

ArrayLike = base.ArrayLike


def td_learning(
    v_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    v_t: ArrayLike,
) -> ArrayLike:
  """Calculates the TD-learning temporal difference error.

  See "Learning to Predict by the Methods of Temporal Differences" by Sutton.
  (https://link.springer.com/article/10.1023/A:1022633531479).

  Args:
    v_tm1: state values at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    v_t: state values at time t.

  Returns:
    TD-learning temporal difference error.
  """

  base.rank_assert([v_tm1, r_t, discount_t, v_t], 0)
  base.type_assert([v_tm1, r_t, discount_t, v_t], float)

  target_tm1 = r_t + discount_t * v_t
  return jax.lax.stop_gradient(target_tm1) - v_tm1


def td_lambda(
    v_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    v_t: ArrayLike,
    lambda_: ArrayLike,
) -> ArrayLike:
  """Calculates the TD(lambda) temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node74.html).

  Args:
    v_tm1: sequence of state values at time t-1.
    r_t: sequence of rewards at time t.
    discount_t: sequence of discounts at time t.
    v_t: sequence of state values at time t.
    lambda_: mixing parameter lambda, either a scalar or a sequence.

  Returns:
    TD(lambda) temporal difference error.
  """
  base.rank_assert([v_tm1, r_t, discount_t, v_t, lambda_], [1, 1, 1, 1, [0, 1]])
  base.type_assert([v_tm1, r_t, discount_t, v_t, lambda_], float)

  target_tm1 = multistep.lambda_returns(r_t, discount_t, v_t, lambda_)
  return jax.lax.stop_gradient(target_tm1) - v_tm1


def sarsa(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t: ArrayLike,
    a_t: ArrayLike,
) -> ArrayLike:
  """Calculates the SARSA temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node64.html.)

  Args:
    q_tm1: Q-values at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_t: Q-values at time t.
    a_t: action index at time t.

  Returns:
    SARSA temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, q_t, a_t],
                   [1, 0, 0, 0, 1, 0])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, q_t, a_t],
                   [float, int, float, float, float, int])

  target_tm1 = r_t + discount_t * q_t[a_t]
  return jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]


def expected_sarsa(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t: ArrayLike,
    probs_a_t: ArrayLike,
) -> ArrayLike:
  """Calculates the expected SARSA (SARSE) temporal difference error.

  See "A Theoretical and Empirical Analysis of Expected Sarsa" by Seijen,
  van Hasselt, Whiteson et al.
  (http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf).

  Args:
    q_tm1: Q-values at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_t: Q-values at time t.
    probs_a_t: action probabilities at time t.

  Returns:
    Expected SARSA temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, q_t, probs_a_t],
                   [1, 0, 0, 0, 1, 1])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, q_t, probs_a_t],
                   [float, int, float, float, float, float])

  target_tm1 = r_t + discount_t * jnp.dot(q_t, probs_a_t)
  return jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]


def sarsa_lambda(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t: ArrayLike,
    a_t: ArrayLike,
    lambda_: ArrayLike,
    stop_target_gradients: bool = True,
) -> ArrayLike:
  """Calculates the SARSA(lambda) temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node77.html).

  Args:
    q_tm1: sequence of Q-values at time t-1.
    a_tm1: sequence of action indices at time t-1.
    r_t: sequence of rewards at time t.
    discount_t: sequence of discounts at time t.
    q_t: sequence of Q-values at time t.
    a_t: sequence of action indices at time t.
    lambda_: mixing parameter lambda, either a scalar or a sequence.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    SARSA(lambda) temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, q_t, a_t, lambda_],
                   [2, 1, 1, 1, 2, 1, [0, 1]])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, q_t, a_t, lambda_],
                   [float, int, float, float, float, int, float])

  qa_tm1 = base.batched_index(q_tm1, a_tm1)
  qa_t = base.batched_index(q_t, a_t)
  target_tm1 = multistep.lambda_returns(r_t, discount_t, qa_t, lambda_)

  if stop_target_gradients:
    target_tm1 = jax.lax.stop_gradient(target_tm1)
  return target_tm1 - qa_tm1


def q_learning(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t: ArrayLike,
) -> ArrayLike:
  """Calculates the Q-learning temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node65.html).

  Args:
    q_tm1: Q-values at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_t: Q-values at time t.

  Returns:
    Q-learning temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, q_t], [1, 0, 0, 0, 1])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, q_t],
                   [float, int, float, float, float])

  target_tm1 = r_t + discount_t * jnp.max(q_t)
  return jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]


def double_q_learning(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t_value: ArrayLike,
    q_t_selector: ArrayLike,
) -> ArrayLike:
  """Calculates the double Q-learning temporal difference error.

  See "Double Q-learning" by van Hasselt.
  (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

  Args:
    q_tm1: Q-values at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_t_value: Q-values at time t.
    q_t_selector: selector Q-values at time t.

  Returns:
    Double Q-learning temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, q_t_value, q_t_selector],
                   [1, 0, 0, 0, 1, 1])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, q_t_value, q_t_selector],
                   [float, int, float, float, float, float])

  target_tm1 = r_t + discount_t * q_t_value[q_t_selector.argmax()]
  return jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]


def persistent_q_learning(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t: ArrayLike,
    action_gap_scale: float,
) -> ArrayLike:
  """Calculates the persistent Q-learning temporal difference error.

  See "Increasing the Action Gap: New Operators for Reinforcement Learning"
  by Bellemare, Ostrovski, Guez et al. (https://arxiv.org/abs/1512.04860).

  Args:
    q_tm1: Q-values at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_t: Q-values at time t.
    action_gap_scale: coefficient in [0, 1] for scaling the action gap term.

  Returns:
    Persistent Q-learning temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, q_t], [1, 0, 0, 0, 1])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, q_t],
                   [float, int, float, float, float])

  corrected_q_t = (
      (1. - action_gap_scale) * jnp.max(q_t)
      + action_gap_scale * q_t[a_tm1]
  )
  target_tm1 = r_t + discount_t * corrected_q_t
  return jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]


def qv_learning(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    v_t: ArrayLike,
) -> ArrayLike:
  """Calculates the QV-learning temporal difference error.

  See "Two Novel On-policy Reinforcement Learning Algorithms based on
  TD(lambda)-methods" by Wiering and van Hasselt
  (https://ieeexplore.ieee.org/abstract/document/4220845.)

  Args:
    q_tm1: Q-values at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    v_t: state values at time t.

  Returns:
    QV-learning temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, v_t], [1, 0, 0, 0, 0])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, v_t],
                   [float, int, float, float, float])

  target_tm1 = r_t + discount_t * v_t
  return jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]


def qv_max(
    v_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t: ArrayLike,
) -> ArrayLike:
  """Calculates the QVMAX temporal difference error.

  See "The QV Family Compared to Other Reinforcement Learning Algorithms" by
  Wiering and van Hasselt (2009).
  (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.713.1931)

  Args:
    v_tm1: state values at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_t: Q-values at time t.

  Returns:
    QVMAX temporal difference error.
  """
  base.rank_assert([v_tm1, r_t, discount_t, q_t], [0, 0, 0, 1])
  base.type_assert([v_tm1, r_t, discount_t, q_t], float)

  target_tm1 = r_t + discount_t * jnp.max(q_t)
  return jax.lax.stop_gradient(target_tm1) - v_tm1


def q_lambda(
    q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_t: ArrayLike,
    lambda_: ArrayLike,
    stop_target_gradients: bool = True,
) -> ArrayLike:
  """Calculates Peng's or Watkins' Q(lambda) temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node78.html).

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

  Returns:
    Q(lambda) temporal difference error.
  """
  base.rank_assert([q_tm1, a_tm1, r_t, discount_t, q_t, lambda_],
                   [2, 1, 1, 1, 2, [0, 1]])
  base.type_assert([q_tm1, a_tm1, r_t, discount_t, q_t, lambda_],
                   [float, int, float, float, float, float])

  qa_tm1 = base.batched_index(q_tm1, a_tm1)
  v_t = jnp.max(q_t, axis=-1)
  target_tm1 = multistep.lambda_returns(r_t, discount_t, v_t, lambda_)

  if stop_target_gradients:
    target_tm1 = jax.lax.stop_gradient(target_tm1)
  return target_tm1 - qa_tm1


def retrace(
    q_tm1: ArrayLike,
    q_t: ArrayLike,
    a_tm1: ArrayLike,
    a_t: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    pi_t: ArrayLike,
    mu_t: ArrayLike,
    lambda_: float,
    eps: float = 1e-8,
    stop_target_gradients: bool = True,
) -> ArrayLike:
  """Calculates Retrace errors.

  See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
  (https://arxiv.org/abs/1606.02647).

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

  Returns:
    Retrace error.
  """
  base.rank_assert([q_tm1, q_t, a_tm1, a_t, r_t, discount_t, pi_t, mu_t],
                   [2, 2, 1, 1, 1, 1, 2, 1])
  base.type_assert([q_tm1, q_t, a_tm1, a_t, r_t, discount_t, pi_t, mu_t],
                   [float, float, int, int, float, float, float, float])

  pi_a_t = base.batched_index(pi_t, a_t)
  c_t = jnp.minimum(1.0, pi_a_t / (mu_t + eps)) * lambda_
  target_tm1 = multistep.general_off_policy_returns_from_action_values(
      q_t, a_t, r_t, discount_t, c_t, pi_t)

  q_a_tm1 = base.batched_index(q_tm1, a_tm1)

  if stop_target_gradients:
    target_tm1 = jax.lax.stop_gradient(target_tm1)
  return target_tm1 - q_a_tm1


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
  """Calculates V-Trace errors from policy logits.

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
  base.rank_assert(
      [v_tm1, v_t, r_t, discount_t, rho_t], [1, 1, 1, 1, 1])
  base.type_assert(
      [v_tm1, v_t, r_t, discount_t, rho_t], [float, float, float, float, float])

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

  # Add the value of the initial state to get the estimates of the returns.
  target_tm1 = jnp.array(errors) + v_tm1

  # Stop gradients and return temporal difference error.
  if stop_target_gradients:
    target_tm1 = jax.lax.stop_gradient(target_tm1)
  return target_tm1 - v_tm1


def _categorical_l2_project(
    z_p: ArrayLike,
    probs: ArrayLike,
    z_q: ArrayLike
) -> ArrayLike:
  """Projects a categorical distribution (z_p, p) onto a different support z_q.

  The projection step minimizes an L2-metric over the cumulative distribution
  functions (CDFs) of the source and target distributions.

  Let kq be len(z_q) and kp be len(z_p). This projection works for any
  support z_q, in particular kq need not be equal to kp.

  See "A Distributional Perspective on RL" by Bellemare et al.
  (https://arxiv.org/abs/1707.06887).

  Args:
    z_p: support of distribution p.
    probs: probability values.
    z_q: support to project distribution (z_p, probs) onto.

  Returns:
    Projection of (z_p, p) onto support z_q under Cramer distance.
  """
  base.rank_assert([z_p, probs, z_q], 1)
  base.type_assert([z_p, probs, z_q], float)

  kp = z_p.shape[0]
  kq = z_q.shape[0]

  # Construct helper arrays from z_q.
  d_pos = jnp.roll(z_q, shift=-1)
  d_neg = jnp.roll(z_q, shift=1)

  # Clip z_p to be in new support range (vmin, vmax).
  z_p = jnp.clip(z_p, z_q[0], z_q[-1])[None, :]
  assert z_p.shape == (1, kp)

  # Get the distance between atom values in support.
  d_pos = (d_pos - z_q)[:, None]  # z_q[i+1] - z_q[i]
  d_neg = (z_q - d_neg)[:, None]  # z_q[i] - z_q[i-1]
  z_q = z_q[:, None]
  assert z_q.shape == (kq, 1)

  # Ensure that we do not divide by zero, in case of atoms of identical value.
  d_neg = jnp.where(d_neg > 0, 1. / d_neg, jnp.zeros_like(d_neg))
  d_pos = jnp.where(d_pos > 0, 1. / d_pos, jnp.zeros_like(d_pos))

  delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]
  d_sign = (delta_qp >= 0.).astype(probs.dtype)
  assert delta_qp.shape == (kq, kp)
  assert d_sign.shape == (kq, kp)

  # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
  delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
  probs = probs[None, :]
  assert delta_hat.shape == (kq, kp)
  assert probs.shape == (1, kp)

  return jnp.sum(jnp.clip(1. - delta_hat, 0., 1.) * probs, axis=-1)


def categorical_td_learning(
    v_atoms_tm1: ArrayLike,
    v_logits_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    v_atoms_t: ArrayLike,
    v_logits_t: ArrayLike
) -> ArrayLike:
  """Implements TD-learning for categorical value distributions.

  See "A Distributional Perspective on Reinforcement Learning", by
    Bellemere, Dabney and Munos (https://arxiv.org/pdf/1707.06887.pdf).

  Args:
    v_atoms_tm1: atoms of V distribution at time t-1.
    v_logits_tm1: logits of V distribution at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    v_atoms_t: atoms of V distribution at time t.
    v_logits_t: logits of V distribution at time t.

  Returns:
    Categorical Q learning loss (i.e. temporal difference error).
  """
  base.rank_assert(
      [v_atoms_tm1, v_logits_tm1, r_t, discount_t, v_atoms_t, v_logits_t],
      [1, 1, 0, 0, 1, 1])
  base.type_assert(
      [v_atoms_tm1, v_logits_tm1, r_t, discount_t, v_atoms_t, v_logits_t],
      [float, float, float, float, float, float])

  # Scale and shift time-t distribution atoms by discount and reward.
  target_z = r_t + discount_t * v_atoms_t

  # Convert logits to distribution.
  v_t_probs = jax.nn.softmax(v_logits_t)

  # Project using the Cramer distance.
  target = jax.lax.stop_gradient(
      _categorical_l2_project(target_z, v_t_probs, v_atoms_tm1))

  # Compute loss (i.e. temporal difference error).
  return distributions.categorical_cross_entropy(
      labels=target, logits=v_logits_tm1)


def categorical_q_learning(
    q_atoms_tm1: ArrayLike,
    q_logits_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_atoms_t: ArrayLike,
    q_logits_t: ArrayLike,
) -> ArrayLike:
  """Implements Q-learning for categorical Q distributions.

  See "A Distributional Perspective on Reinforcement Learning", by
    Bellemere, Dabney and Munos (https://arxiv.org/pdf/1707.06887.pdf).

  Args:
    q_atoms_tm1: atoms of Q distribution at time t-1.
    q_logits_tm1: logits of Q distribution at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_atoms_t: atoms of Q distribution at time t.
    q_logits_t: logits of Q distribution at time t.

  Returns:
    Categorical Q-learning loss (i.e. temporal difference error).
  """
  base.rank_assert([
      q_atoms_tm1, q_logits_tm1, a_tm1, r_t, discount_t, q_atoms_t, q_logits_t
  ], [1, 2, 0, 0, 0, 1, 2])
  base.type_assert([
      q_atoms_tm1, q_logits_tm1, a_tm1, r_t, discount_t, q_atoms_t, q_logits_t
  ], [float, float, int, float, float, float, float])

  # Scale and shift time-t distribution atoms by discount and reward.
  target_z = r_t + discount_t * q_atoms_t

  # Convert logits to distribution, then find greedy action in state s_t.
  q_t_probs = jax.nn.softmax(q_logits_t)
  q_t_mean = jnp.sum(q_t_probs * q_atoms_t, axis=1)
  pi_t = jnp.argmax(q_t_mean)

  # Compute distribution for greedy action.
  p_target_z = q_t_probs[pi_t]

  # Project using the Cramer distance.
  target = jax.lax.stop_gradient(
      _categorical_l2_project(target_z, p_target_z, q_atoms_tm1))

  # Compute loss (i.e. temporal difference error).
  logit_qa_tm1 = q_logits_tm1[a_tm1]
  return distributions.categorical_cross_entropy(
      labels=target, logits=logit_qa_tm1)


def categorical_double_q_learning(
    q_atoms_tm1: ArrayLike,
    q_logits_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    q_atoms_t: ArrayLike,
    q_logits_t: ArrayLike,
    q_t_selector: ArrayLike,
) -> ArrayLike:
  """Implements double Q-learning for categorical Q distributions.

  See "A Distributional Perspective on Reinforcement Learning", by
    Bellemere, Dabney and Munos (https://arxiv.org/pdf/1707.06887.pdf)
  and "Double Q-learning" by van Hasselt.
  (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

  Args:
    q_atoms_tm1: atoms of Q distribution at time t-1.
    q_logits_tm1: logits of Q distribution at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_atoms_t: atoms of Q distribution at time t.
    q_logits_t: logits of Q distribution at time t.
    q_t_selector: selector Q-values at time t.

  Returns:
    Categorical double Q-learning loss (i.e. temporal difference error).
  """
  base.rank_assert([
      q_atoms_tm1, q_logits_tm1, a_tm1, r_t, discount_t, q_atoms_t, q_logits_t,
      q_t_selector
  ], [1, 2, 0, 0, 0, 1, 2, 1])
  base.type_assert([
      q_atoms_tm1, q_logits_tm1, a_tm1, r_t, discount_t, q_atoms_t, q_logits_t,
      q_t_selector
  ], [float, float, int, float, float, float, float, float])

  # Scale and shift time-t distribution atoms by discount and reward.
  target_z = r_t + discount_t * q_atoms_t

  # Select logits for greedy action in state s_t and convert to distribution.
  p_target_z = jax.nn.softmax(q_logits_t[q_t_selector.argmax()])

  # Project using the Cramer distance.
  target = jax.lax.stop_gradient(
      _categorical_l2_project(target_z, p_target_z, q_atoms_tm1))

  # Compute loss (i.e. temporal difference error).
  logit_qa_tm1 = q_logits_tm1[a_tm1]
  return distributions.categorical_cross_entropy(
      labels=target, logits=logit_qa_tm1)


def _quantile_regression_loss(
    dist_src: ArrayLike,
    tau_src: ArrayLike,
    dist_target: ArrayLike,
    huber_param: float = 0.
) -> ArrayLike:
  """Compute (Huber) QR loss between two discrete quantile-valued distributions.

  See "Distributional Reinforcement Learning with Quantile Regression" by
  Dabney et al. (https://arxiv.org/abs/1710.10044).

  Args:
    dist_src: source probability distribution.
    tau_src: source distribution probability thresholds.
    dist_target: target probability distribution.
    huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

  Returns:
    Quantile regression loss.
  """
  base.rank_assert([dist_src, tau_src, dist_target], 1)
  base.type_assert([dist_src, tau_src, dist_target], float)

  # Calculate quantile error.
  delta = dist_target[None, :] - dist_src[:, None]
  delta_neg = (delta < 0.).astype(jnp.float32)
  delta_neg = jax.lax.stop_gradient(delta_neg)
  weight = jnp.abs(tau_src[:, None] - delta_neg)

  # Calculate Huber loss.
  if huber_param > 0.:
    loss = clipping.huber_loss(delta, huber_param)
  else:
    loss = jnp.abs(delta)
  loss *= weight

  # Average over target-samples dimension, sum over src-samples dimension.
  return jnp.sum(jnp.mean(loss, axis=-1))


def quantile_q_learning(
    dist_q_tm1: ArrayLike,
    tau_q_tm1: ArrayLike,
    a_tm1: ArrayLike,
    r_t: ArrayLike,
    discount_t: ArrayLike,
    dist_q_t_selector: ArrayLike,
    dist_q_t: ArrayLike,
    huber_param: float = 0.
) -> ArrayLike:
  """Implements Q-learning for quantile-valued Q distributions.

  See "Distributional Reinforcement Learning with Quantile Regression" by
  Dabney et al. (https://arxiv.org/abs/1710.10044).

  Args:
    dist_q_tm1: Q distribution at time t-1.
    tau_q_tm1: Q distribution probability thresholds.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    dist_q_t_selector: Q distribution at time t for selecting greedy action in
      target policy. This is separate from dist_q_t as in Double Q-Learning, but
      can be computed with the target network and a separate set of samples.
    dist_q_t: target Q distribution at time t.
    huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

  Returns:
    Quantile regression Q learning loss.
  """
  base.rank_assert([
      dist_q_tm1, tau_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t
  ], [2, 1, 0, 0, 0, 2, 2])
  base.type_assert([
      dist_q_tm1, tau_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t
  ], [float, float, int, float, float, float, float])

  # Only update the taken actions.
  dist_qa_tm1 = dist_q_tm1[:, a_tm1]

  # Select target action according to greedy policy w.r.t. dist_q_t_selector.
  q_t_selector = jnp.mean(dist_q_t_selector, axis=0)
  a_t = jnp.argmax(q_t_selector)
  dist_qa_t = dist_q_t[:, a_t]

  # Compute target, do not backpropagate into it.
  dist_target = r_t + discount_t * dist_qa_t
  dist_target = jax.lax.stop_gradient(dist_target)

  return _quantile_regression_loss(
      dist_qa_tm1, tau_q_tm1, dist_target, huber_param)
