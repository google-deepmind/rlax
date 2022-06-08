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

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class EmaState:
  # The tree of first moments.
  first_moment: chex.ArrayTree
  # The tree of second moments.
  second_moment: chex.ArrayTree
  # The product of the all decays from the start of accumulating.
  decay_product: float


@chex.dataclass(frozen=True)
class EmaMoments:
  # The tree of means.
  mean: chex.ArrayTree
  # The tree of variances.
  variance: chex.ArrayTree


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
    zeros = jax.tree_map(lambda x: jnp.zeros_like(jnp.mean(x)), template_tree)
    scalar_zero = jnp.ones([], dtype=jnp.float32)
    return EmaState(
        first_moment=zeros, second_moment=zeros, decay_product=scalar_zero)

  def _update(moment, value):
    mean = jnp.mean(value)
    # Compute the mean across all learner devices involved in the `pmap`.
    if pmean_axis_name is not None:
      mean = jax.lax.pmean(mean, axis_name=pmean_axis_name)
    return decay * moment + (1 - decay) * mean

  def update_moments(tree, state):
    squared_tree = jax.tree_map(jnp.square, tree)
    first_moment = jax.tree_map(_update, state.first_moment, tree)
    second_moment = jax.tree_map(_update, state.second_moment, squared_tree)
    state = EmaState(
        first_moment=first_moment, second_moment=second_moment,
        decay_product=state.decay_product * decay)
    moments = compute_moments(state)
    return moments, state

  return init_state, update_moments


def compute_moments(state):
  """Returns debiased moments as in Adam."""
  tiny = jnp.finfo(state.decay_product).tiny
  debias = 1.0 / jnp.maximum(1 - state.decay_product, tiny)
  mean = jax.tree_map(lambda m1: m1 * debias, state.first_moment)
  # This computation of the variance may lose some numerical precision, if
  # the mean is not approximately zero.
  variance = jax.tree_map(
      lambda m2, m: jnp.maximum(0.0, m2 * debias - jnp.square(m)),
      state.second_moment, mean)
  return EmaMoments(mean=mean, variance=variance)
