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
"""Utilities for creating and managing moving averages."""

from typing import Union

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class EmaMoments:
  """data-class holding the latest mean and variance estimates."""
  # The tree of means.
  mean: chex.ArrayTree
  # The tree of variances.
  variance: chex.ArrayTree


@chex.dataclass(frozen=True)
class EmaState:
  """data-class holding the exponential moving average state."""
  # The tree of exponential moving averages of the values
  mu: chex.ArrayTree
  # The tree of exponential moving averages of the squared values
  nu: chex.ArrayTree
  # The product of the all decays from the start of accumulating.
  decay_product: Union[float, jax.Array]

  def debiased_moments(self):
    """Returns debiased moments as in Adam."""
    tiny = jnp.finfo(self.decay_product).tiny
    debias = 1.0 / jnp.maximum(1 - self.decay_product, tiny)
    mean = jax.tree.map(lambda m1: m1 * debias, self.mu)
    # This computation of the variance may lose some numerical precision, if
    # the mean is not approximately zero.
    variance = jax.tree.map(
        lambda m2, m: jnp.maximum(0.0, m2 * debias - jnp.square(m)),
        self.nu, mean)
    return EmaMoments(mean=mean, variance=variance)


def create_ema(decay=0.999, pmean_axis_name=None):
  """An updater of moments.

  Given a `tree` it will track first and second moments of the leaves.

  Args:
    decay: The decay of the moments. I.e., the learning rate is `1 - decay`.
    pmean_axis_name: If not None, use lax.pmean to average the moment updates.

  Returns:
    Two functions: `(init_state, update_moments)`.
  """

  def init_state(template_tree):
    zeros = jax.tree.map(lambda x: jnp.zeros_like(jnp.mean(x)), template_tree)
    scalar_zero = jnp.ones([], dtype=jnp.float32)
    return EmaState(mu=zeros, nu=zeros, decay_product=scalar_zero)

  def _update(moment, value):
    mean = jnp.mean(value)
    # Compute the mean across all learner devices involved in the `pmap`.
    if pmean_axis_name is not None:
      mean = jax.lax.pmean(mean, axis_name=pmean_axis_name)
    return decay * moment + (1 - decay) * mean

  def update_moments(tree, state):
    squared_tree = jax.tree.map(jnp.square, tree)
    mu = jax.tree.map(_update, state.mu, tree)
    nu = jax.tree.map(_update, state.nu, squared_tree)
    state = EmaState(
        mu=mu, nu=nu, decay_product=state.decay_product * decay)
    return state.debiased_moments(), state

  return init_state, update_moments
