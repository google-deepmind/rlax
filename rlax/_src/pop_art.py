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
"""JAX functions implementing PopArt."""

import collections

from typing import Mapping, Tuple

import chex
import jax
import jax.numpy as jnp
from rlax._src import base

Array = chex.Array

LinearParams = Mapping[str, Array]
PopArtState = collections.namedtuple(
    "PopArtState", ["shift", "scale", "second_moment"])
PopArtOutput = collections.namedtuple(
    "PopArtOutput", ["normalized", "unnormalized"])


def _cross_replica_scatter_add(source: Array, indices: Array, updates: Array,
                               axis_name):
  """tf.scatter_add, but with JAX, cross replica, and without state.

  Args:
    source: An array of shape [O].
    indices: An array indicating which index each update is for.
    updates: The updates to apply to `source`. Of same shape as indices.
    axis_name: What axis to aggregate over, if str. If passed an iterable,
      aggregates over multiple axes. Defaults to no aggregation, i.e. None.

  Returns:
    An array of shape [O], which is source + the scattered updates from all
    replicas.
  """
  assert updates.shape == indices.shape
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  assert source.ndim == 1
  # Flatten indices, updates.
  num_classes = source.shape[0]
  indices = jnp.reshape(indices, [-1])
  updates = jnp.reshape(updates, [-1])
  # Scatter updates according to value of indices.
  updates_at_idxs = updates[..., None] * base.one_hot(indices, num_classes)
  # Aggregate locally first, then across replicas.
  total_updates = jnp.sum(updates_at_idxs, axis=0)
  if axis_name is not None:
    axis_names = (axis_name,) if isinstance(axis_name, str) else axis_name
    for axis_name in axis_names:
      total_updates = jax.lax.psum(total_updates, axis_name=axis_name)
  return source + total_updates


def normalize(state: PopArtState, unnormalized: Array, indices: Array) -> Array:
  """Returns normalized values.

  Args:
    state: The PopArt summary stats.
    unnormalized: unnormalized values that we applied PopArt to.
    indices: Which scale and shifts to use

  Returns:
    Normalized PopArt values.
  """
  scale = state.scale[indices]
  shift = state.shift[indices]
  normalized = (unnormalized - shift) / scale
  return normalized


def unnormalize(state: PopArtState, normalized: Array, indices: Array) -> Array:
  """Returns unnormalized values.

  Args:
    state: The PopArt summary stats.
    normalized: normalized values that we apply PopArt to.
    indices: Which scale and shifts to use

  Returns:
    Unnormalized PopArt values.
  """
  scale = state.scale[indices]
  shift = state.shift[indices]
  unnormalized = scale * normalized + shift
  return unnormalized


def unnormalize_linear(state: PopArtState, inputs: Array,
                       indices: Array) -> PopArtOutput:
  """Selects and unnormalizes output of a Linear.

  Args:
    state: The PopArt summary stats.
    inputs: The (normalized) output of the Linear that we apply PopArt to.
    indices: Which indices of `inputs` to use.

  Returns:
    PopArtOutput, a tuple of the normalized and unnormalized PopArt values.
  """
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  assert indices.shape == inputs.shape[:-1]
  normalized = jnp.take_along_axis(inputs, indices[..., None], axis=-1)
  normalized = jnp.squeeze(normalized, axis=-1)
  return PopArtOutput(normalized, unnormalize(state, normalized, indices))


def art(state: PopArtState,
        targets: Array,
        indices: Array,
        step_size: float,
        scale_lb: float,
        scale_ub: float,
        axis_name=None) -> PopArtState:
  """Adaptively rescale targets.

  Args:
    state: The PopArt summary stats.
    targets: targets which are rescaled.
    indices: Which indices of the state to use.
    step_size: The step size for learning the scale & shift parameters.
    scale_lb: Lower bound for the scale.
    scale_ub: Upper bound for the scale.
    axis_name: What axis to aggregate over, if str. If passed an iterable,
      aggregates over multiple axes. Defaults to no aggregation, i.e. None.

  Returns:
    New popart state which can be used to rescale targets.
  """
  assert targets.shape == indices.shape
  assert jnp.issubdtype(indices.dtype, jnp.integer)

  # Update shift.
  shift_gather = state.shift[indices]
  shift_update = step_size * (targets - shift_gather)
  shift_new = _cross_replica_scatter_add(state.shift, indices, shift_update,
                                         axis_name)

  # Update second moment.
  second_moment_gather = state.second_moment[indices]
  second_moment_update = step_size * (
      jnp.square(targets) - second_moment_gather)
  second_moment_new = _cross_replica_scatter_add(state.second_moment, indices,
                                                 second_moment_update,
                                                 axis_name)

  # Derive scale (stdev) from second moment and mean.
  scale_sq = second_moment_new - jnp.square(shift_new)
  scale_sq = jnp.clip(scale_sq, scale_lb**2, scale_ub**2)
  scale_new = jnp.sqrt(scale_sq)

  state_new = PopArtState(shift_new, scale_new, second_moment_new)
  return state_new


def pop(params: LinearParams, old: PopArtState, new: PopArtState):
  """Preserves outputs precisely.

  Args:
    params: The parameters of the linear to preserve.
    old: The old PopArt state.
    new: The new PopArt state.

  Returns:
    new parameters.
  """
  w_new = params["w"] * jnp.broadcast_to(old.scale / new.scale,
                                         params["w"].shape)
  b_new = (old.scale * params["b"] + old.shift - new.shift) / new.scale
  params_new = dict(w=w_new, b=b_new)
  return params_new


def popart(num_outputs: int,
           step_size: float,
           scale_lb: float,
           scale_ub: float,
           axis_name=None):
  """Generates functions giving initial PopArt state and update rule.

  Args:
    num_outputs: The number of outputs generated by the linear we're preserving.
    step_size: The step size for learning the scale & shift parameters.
    scale_lb: Lower bound for the scale.
    scale_ub: Upper bound for the scale.
    axis_name: What axis to aggregate over, if str. If passed an iterable,
      aggregates over multiple axes. Defaults to no aggregation, i.e. None.

  Returns:
    A tuple of:
      initial_state: A function returning the initial PopArt state.
      popart_update: A function updating the PopArt state and parameters
        of the preceding linear.
  """

  def initial_state():
    return PopArtState(
        jnp.zeros([num_outputs]), jnp.ones([num_outputs]),
        jnp.ones([num_outputs]))

  def popart_update(params: LinearParams, state: PopArtState, targets: Array,
                    indices: Array) -> Tuple[LinearParams, PopArtState]:
    """Computes the PopArt update.

    Args:
      params: The parameters of the linear to preserve.
      state: The current PopArt state.
      targets: Values whose distribution to learn.
      indices: For each target, which shift and scale element to adjust.

    Returns:
      A tuple of:
        new_params: The new parameters of the linear, preserving outputs.
        new_state: The new PopArt state.
    """
    # Disables Popart if step_size is None
    if step_size is None:
      return params, state

    # Adaptively rescale targets.
    state_new = art(state, targets, indices, step_size, scale_lb, scale_ub,
                    axis_name)
    # Preserve outputs precisely.
    params_new = pop(params, state, state_new)
    return params_new, state_new

  return initial_state, popart_update
