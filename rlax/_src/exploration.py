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
"""JAX functions implementing different forms of perturbations.

It is common in reinforcement learning to introduce various forms of
stochasticity in the training process in order to aid exploration. This
subpackage exposes popular forms of perturbations used by RL agents.
"""

import chex
import jax
from rlax._src import base

ArrayLike = base.ArrayLike


def add_gaussian_noise(
    key: ArrayLike,
    action: ArrayLike,
    stddev: float
) -> ArrayLike:
  """Returns continuous action with noise drawn from a Gaussian distribution.

  Args:
    key: a key from `jax.random`.
    action: continuous action scalar or vector.
    stddev: standard deviation of noise distribution.

  Returns:
    noisy action, of the same shape as input action.
  """
  chex.type_assert(action, float)

  noise = jax.random.normal(key, shape=action.shape) * stddev
  return action + noise


def add_ornstein_uhlenbeck_noise(
    key: ArrayLike,
    action: ArrayLike,
    noise_tm1: ArrayLike,
    damping: float,
    stddev: float
) -> ArrayLike:
  """Returns continuous action with noise from Ornstein-Uhlenbeck process.

  See "On the theory of Brownian Motion" by Uhlenbeck and Ornstein.
  (https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823).

  Args:
    key: a key from `jax.random`.
    action: continuous action scalar or vector.
    noise_tm1: noise sampled from OU process in previous timestep.
    damping: parameter for controlling autocorrelation of OU process.
    stddev: standard deviation of noise distribution.

  Returns:
    noisy action, of the same shape as input action.
  """
  chex.rank_assert([action, noise_tm1], 1)
  chex.type_assert([action, noise_tm1], float)

  noise_t = (1. - damping) * noise_tm1 + jax.random.normal(
      key, shape=action.shape) * stddev

  return action + noise_t
