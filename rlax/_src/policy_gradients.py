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
"""JAX functions implementing policy gradient losses.

Policy gradient algorithms directly update the policy of an agent based on
a stochatic estimate of the direction of steepest ascent in a score function
representing the expected return of that policy. This subpackage provides a
number of utility functions for implementing policy gradient algorithms for
discrete and continuous policies.
"""

from typing import Optional, Tuple
import chex
import jax
import jax.numpy as jnp
from rlax._src import distributions
from rlax._src import losses

Array = chex.Array
Scalar = chex.Scalar


def _clip_by_l2_norm(x: Array, max_norm: float) -> Array:
  """Clip gradients to maximum l2 norm `max_norm`."""
  # Compute the sum of squares and find out where things are zero.
  sum_sq = jnp.sum(jnp.vdot(x, x))
  nonzero = sum_sq > 0

  # Compute the norm wherever sum_sq > 0 and leave it <= 0 otherwise. This makes
  # use of the the "double where" trick; see
  # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
  # for more info. In short this is necessary because although norm ends up
  # computed correctly where nonzero is true if we ignored this we'd end up with
  # nans on the off-branches which would leak through when computed gradients in
  # the backward pass.
  sum_sq_ones = jnp.where(nonzero, sum_sq, jnp.ones_like(sum_sq))
  norm = jnp.where(nonzero, jnp.sqrt(sum_sq_ones), sum_sq)

  # Normalize by max_norm. Whenever norm < max_norm we're left with x (this
  # happens trivially for indices where nonzero is false). Otherwise we're left
  # with the desired x * max_norm / norm.
  return (x * max_norm) / jnp.maximum(norm, max_norm)


def dpg_loss(
    a_t: Array,
    dqda_t: Array,
    dqda_clipping: Optional[Scalar] = None,
    use_stop_gradient: bool = True,
) -> Array:
  """Calculates the deterministic policy gradient (DPG) loss.

  See "Deterministic Policy Gradient Algorithms" by Silver, Lever, Heess,
  Degris, Wierstra, Riedmiller (http://proceedings.mlr.press/v32/silver14.pdf).

  Args:
    a_t: continuous-valued action at time t.
    dqda_t: gradient of Q(s,a) wrt. a, evaluated at time t.
    dqda_clipping: clips the gradient to have norm <= `dqda_clipping`.
    use_stop_gradient: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    DPG loss.
  """
  chex.assert_rank([a_t, dqda_t], 1)
  chex.assert_type([a_t, dqda_t], float)

  if dqda_clipping is not None:
    dqda_t = _clip_by_l2_norm(dqda_t, dqda_clipping)
  target_tm1 = dqda_t + a_t
  target_tm1 = jax.lax.select(use_stop_gradient,
                              jax.lax.stop_gradient(target_tm1), target_tm1)
  return losses.l2_loss(target_tm1 - a_t)


def policy_gradient_loss(
    logits_t: Array,
    a_t: Array,
    adv_t: Array,
    w_t: Array,
    use_stop_gradient: bool = True,
) -> Array:
  """Calculates the policy gradient loss.

  See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
  (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

  Args:
    logits_t: a sequence of unnormalized action preferences.
    a_t: a sequence of actions sampled from the preferences `logits_t`.
    adv_t: the observed or estimated advantages from executing actions `a_t`.
    w_t: a per timestep weighting for the loss.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.

  Returns:
    Loss whose gradient corresponds to a policy gradient update.
  """
  chex.assert_rank([logits_t, a_t, adv_t, w_t], [2, 1, 1, 1])
  chex.assert_type([logits_t, a_t, adv_t, w_t], [float, int, float, float])

  log_pi_a_t = distributions.softmax().logprob(a_t, logits_t)
  adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
  loss_per_timestep = -log_pi_a_t * adv_t
  return jnp.mean(loss_per_timestep * w_t)


def entropy_loss(
    logits_t: Array,
    w_t: Array,
) -> Array:
  """Calculates the entropy regularization loss.

  See "Function Optimization using Connectionist RL Algorithms" by Williams.
  (https://www.tandfonline.com/doi/abs/10.1080/09540099108946587)

  Args:
    logits_t: a sequence of unnormalized action preferences.
    w_t: a per timestep weighting for the loss.

  Returns:
    Entropy loss.
  """
  chex.assert_rank([logits_t, w_t], [2, 1])
  chex.assert_type([logits_t, w_t], float)

  entropy_per_timestep = distributions.softmax().entropy(logits_t)
  return -jnp.mean(entropy_per_timestep * w_t)


def _compute_advantages(logits_t: Array,
                        q_t: Array,
                        use_stop_gradient=True) -> Tuple[Array, Array]:
  """Computes summed advantage using logits and action values."""
  policy_t = jax.nn.softmax(logits_t, axis=1)

  # Avoid computing gradients for action_values.
  q_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(q_t), q_t)
  baseline_t = jnp.sum(policy_t * q_t, axis=1)

  adv_t = q_t - jnp.expand_dims(baseline_t, 1)
  return policy_t, adv_t


def qpg_loss(
    logits_t: Array,
    q_t: Array,
    use_stop_gradient: bool = True,
) -> Array:
  """Computes the QPG (Q-based Policy Gradient) loss.

  See "Actor-Critic Policy Optimization in Partially Observable Multiagent
  Environments" by Srinivasan, Lanctot (https://arxiv.org/abs/1810.09026).

  Args:
    logits_t: a sequence of unnormalized action preferences.
    q_t: the observed or estimated action value from executing actions `a_t` at
      time t.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.

  Returns:
    QPG Loss.
  """
  chex.assert_rank([logits_t, q_t], 2)
  chex.assert_type([logits_t, q_t], float)

  policy_t, advantage_t = _compute_advantages(logits_t, q_t)
  advantage_t = jax.lax.select(use_stop_gradient,
                               jax.lax.stop_gradient(advantage_t), advantage_t)
  policy_advantages = -policy_t * advantage_t
  loss = jnp.mean(jnp.sum(policy_advantages, axis=1), axis=0)
  return loss


def rm_loss(
    logits_t: Array,
    q_t: Array,
    use_stop_gradient: bool = True,
) -> Array:
  """Computes the RMPG (Regret Matching Policy Gradient) loss.

  The gradient of this loss adapts the Regret Matching rule by weighting the
  standard PG update with thresholded regret.

  See "Actor-Critic Policy Optimization in Partially Observable Multiagent
  Environments" by Srinivasan, Lanctot (https://arxiv.org/abs/1810.09026).

  Args:
    logits_t: a sequence of unnormalized action preferences.
    q_t: the observed or estimated action value from executing actions `a_t` at
      time t.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.

  Returns:
    RM Loss.
  """
  chex.assert_rank([logits_t, q_t], 2)
  chex.assert_type([logits_t, q_t], float)

  policy_t, advantage_t = _compute_advantages(logits_t, q_t)
  action_regret_t = jax.nn.relu(advantage_t)
  action_regret_t = jax.lax.select(use_stop_gradient,
                                   jax.lax.stop_gradient(action_regret_t),
                                   action_regret_t)
  policy_regret = -policy_t * action_regret_t
  loss = jnp.mean(jnp.sum(policy_regret, axis=1), axis=0)
  return loss


def rpg_loss(
    logits_t: Array,
    q_t: Array,
    use_stop_gradient: bool = True,
) -> Array:
  """Computes the RPG (Regret Policy Gradient) loss.

  The gradient of this loss adapts the Regret Matching rule by weighting the
  standard PG update with regret.

  See "Actor-Critic Policy Optimization in Partially Observable Multiagent
  Environments" by Srinivasan, Lanctot (https://arxiv.org/abs/1810.09026).

  Args:
    logits_t: a sequence of unnormalized action preferences.
    q_t: the observed or estimated action value from executing actions `a_t` at
      time t.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.

  Returns:
    RPG Loss.
  """
  chex.assert_rank([logits_t, q_t], 2)
  chex.assert_type([logits_t, q_t], float)

  _, adv_t = _compute_advantages(logits_t, q_t, use_stop_gradient)
  regrets_t = jnp.sum(jax.nn.relu(adv_t), axis=1)

  total_regret_t = jnp.mean(regrets_t, axis=0)
  return total_regret_t


def clipped_surrogate_pg_loss(
    prob_ratios_t: Array,
    adv_t: Array,
    epsilon: Scalar,
    use_stop_gradient=True) -> Array:
  """Computes the clipped surrogate policy gradient loss.

  L_clipₜ(θ) = - min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)

  Where rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ) and Âₜ are the advantages.

  See Proximal Policy Optimization Algorithms, Schulman et al.:
  https://arxiv.org/abs/1707.06347

  Args:
    prob_ratios_t: Ratio of action probabilities for actions a_t:
        rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ)
    adv_t: the observed or estimated advantages from executing actions a_t.
    epsilon: Scalar value corresponding to how much to clip the objecctive.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.

  Returns:
    Loss whose gradient corresponds to a clipped surrogate policy gradient
        update.
  """
  chex.assert_rank([prob_ratios_t, adv_t], [1, 1])
  chex.assert_type([prob_ratios_t, adv_t], [float, float])

  adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
  clipped_ratios_t = jnp.clip(prob_ratios_t, 1. - epsilon, 1. + epsilon)
  clipped_objective = jnp.fmin(prob_ratios_t * adv_t, clipped_ratios_t * adv_t)
  return -jnp.mean(clipped_objective)
