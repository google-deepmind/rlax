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
"""JAX functions implementing policy gradient losses.

Policy gradient algorithms directly update the policy of an agent based on
a stochatic estimate of the direction of steepest ascent in a score function
representing the expected return of that policy. This subpackage provides a
number of utility functions for implementing policy gradient algorithms for
discrete and continuous policies.
"""

from typing import Optional
import jax
import jax.numpy as jnp
from rlax._src import base
from rlax._src import distributions
from rlax._src import losses

ArrayLike = base.ArrayLike


def _clip_by_l2_norm(x: ArrayLike, max_norm: float) -> ArrayLike:
  """Clip gradients to maximum l2 norm `max_norm`."""
  norm = jnp.sqrt(jnp.sum(jnp.vdot(x, x)))
  return jnp.where(norm > max_norm, x * (max_norm / norm), x)


def dpg_loss(
    a_t: ArrayLike,
    dqda_t: ArrayLike,
    dqda_clipping: Optional[base.Scalar] = None
) -> ArrayLike:
  """Calculates the deterministic policy gradient (DPG) loss.

  See "Deterministic Policy Gradient Algorithms" by Silver, Lever, Heess,
  Degris, Wierstra, Riedmiller (http://proceedings.mlr.press/v32/silver14.pdf).

  Args:
    a_t: continuous-valued action at time t.
    dqda_t: gradient of Q(s,a) wrt. a, evaluated at time t.
    dqda_clipping: clips the gradient to have norm <= `dqda_clipping`.

  Returns:
    DPG loss.
  """
  base.rank_assert([a_t, dqda_t], 1)
  base.type_assert([a_t, dqda_t], float)

  if dqda_clipping is not None:
    dqda_t = _clip_by_l2_norm(dqda_t, dqda_clipping)
  target_tm1 = dqda_t + a_t
  return losses.l2_loss(jax.lax.stop_gradient(target_tm1) - a_t)


def policy_gradient_loss(
    logits_t: ArrayLike,
    a_t: ArrayLike,
    adv_t: ArrayLike,
    w_t: ArrayLike,
) -> ArrayLike:
  """Calculates the policy gradient loss.

  See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
  (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

  Args:
    logits_t: a sequence of unnormalized action preferences.
    a_t: a sequence of actions sampled from the preferences `logits_t`.
    adv_t: the observed or estimated advantages from executing actions `a_t`.
    w_t: a per timestep weighting for the loss.

  Returns:
    Loss whose gradient corresponds to a policy gradient update.
  """
  base.rank_assert([logits_t, a_t, adv_t, w_t], [2, 1, 1, 1])
  base.type_assert([logits_t, a_t, adv_t, w_t], [float, int, float, float])

  log_pi_a = distributions.softmax().logprob(a_t, logits_t)
  adv_t = jax.lax.stop_gradient(adv_t)
  loss_per_timestep = - log_pi_a * adv_t
  return jnp.mean(loss_per_timestep * w_t)


def entropy_loss(
    logits_t: ArrayLike,
    w_t: ArrayLike,
) -> ArrayLike:
  """Calculates the entropy regularization loss.

  See "Function Optimization using Connectionist RL Algorithms" by Williams.
  (https://www.tandfonline.com/doi/abs/10.1080/09540099108946587)

  Args:
    logits_t: a sequence of unnormalized action preferences.
    w_t: a per timestep weighting for the loss.

  Returns:
    Entropy loss.
  """
  base.rank_assert([logits_t, w_t], [2, 1])
  base.type_assert([logits_t, w_t], float)

  entropy_per_timestep = distributions.softmax().entropy(logits_t)
  return -jnp.mean(entropy_per_timestep * w_t)
