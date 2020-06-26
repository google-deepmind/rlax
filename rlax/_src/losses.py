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
"""JAX implementation of common losses.

Deep reinforcement learning algorithms are often expressed as gradients of
suitable pseudo-loss functions constructed from the observations and rewards
collected in the environment. In this subpackage we collate common mathematical
transformations used to construct such losses.
"""

import functools

from typing import Optional, Union
import chex
import jax
import jax.numpy as jnp
from rlax._src import general_value_functions
from rlax._src import value_learning

Scalar = chex.Scalar
Array = chex.Array


def l2_loss(predictions: Array,
            targets: Optional[Array] = None) -> Array:
  """Caculates the L2 loss of predictions wrt targets.

  If targets are not provided this function acts as an L2-regularizer for preds.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  if targets is None:
    targets = jnp.zeros_like(predictions)
  chex.type_assert([predictions, targets], float)
  return 0.5 * (predictions - targets)**2


def likelihood(predictions: Array, targets: Array) -> Array:
  """Calculates the likelihood of predictions wrt targets.

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  chex.type_assert([predictions, targets], float)
  likelihood_vals = predictions**targets * (1. - predictions)**(1. - targets)
  # Note: 0**0 evaluates to NaN on TPUs, manually set these cases to 1.
  filter_indices = jnp.logical_or(
      jnp.logical_and(targets == 1, predictions == 1),
      jnp.logical_and(targets == 0, predictions == 0))
  return jnp.where(filter_indices, 1, likelihood_vals)


def log_loss(
    predictions: Array,
    targets: Array,
) -> Array:
  """Calculates the log loss of predictions wrt targets.

  Args:
    predictions: a vector of probabilities of arbitrary shape.
    targets: a vector of probabilities of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  chex.type_assert([predictions, targets], float)
  return -jnp.log(likelihood(predictions, targets))


def pixel_control_loss(
    observations: Array,
    actions: Array,
    action_values: Array,
    discount_factor: Union[Array, Scalar],
    cell_size: int):
  """Calculate n-step Q-learning loss for pixel control auxiliary task.

  For each pixel-based pseudo reward signal, the corresponding action-value
  function is trained off-policy, using Q(lambda). A discount of 0.9 is
  commonly used for learning the value functions.

  Note that, since pseudo rewards have a spatial structure, with neighbouring
  cells exhibiting strong correlations, it is convenient to predict the action
  values for all the cells through a deconvolutional head.

  See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
  Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).

  Args:
    observations: A tensor of shape `[T+1, ...]`; `...` is the observation
      shape, `T` the sequence length.
    actions: A tensor, shape `[T,]`, of the actions across each sequence.
    action_values: A tensor, shape `[T+1, H, W, N]` of pixel control action
      values, where `H`, `W` are the number of pixel control cells/tasks, and
      `N` is the number of actions.
    discount_factor: discount used for learning the value function associated
      to the pseudo rewards; must be a scalar or a Tensor of shape [T].
    cell_size: size of the cells used to derive the pixel based pseudo-rewards.

  Returns:
    a tensor containing the spatial loss, shape [T, H, W].

  Raises:
    ValueError: if the shape of `action_values` is not compatible with that of
      the pseudo-rewards derived from the observations.
  """
  # Check shapes
  assert len(actions.shape) == 1
  assert len(action_values.shape) == 4
  # Check types
  chex.type_assert([observations], float)
  chex.type_assert([actions], int)
  # Useful shapes.
  sequence_length = actions.shape[0]
  num_actions = action_values.shape[-1]
  height_width_q = action_values.shape[1:-1]
  # Calculate rewards using the observations.
  # Compute pseudo-rewards and get their shape.
  pseudo_rewards = general_value_functions.pixel_control_rewards(
      observations, cell_size)
  height_width = pseudo_rewards.shape[1:]
  # Check that pseudo-rewards and Q-values are compatible in shape.
  if height_width != height_width_q:
    raise ValueError(
        "Pixel Control values are not compatible with the shape of the"
        "pseudo-rewards derived from the observation. Pseudo-rewards have shape"
        "{}, while Pixel Control values have shape {}".format(
            height_width, height_width_q))
  # We now have Q(s,a) and rewards, so can calculate the n-step loss. The
  # QLambda loss op expects inputs of shape [T,N] and [T], but our tensors
  # are in a variety of incompatible shapes. The state-action values have
  # shape [T,H,W,N] and rewards have shape [T,H,W]. We can think of the
  # [H,W] dimensions as extra batch dimensions for the purposes of the loss
  # calculation, so we first collapse [H,W] into a single dimension.
  q_tm1 = jnp.reshape(action_values[:-1], (sequence_length, -1, num_actions))
  r_t = jnp.reshape(pseudo_rewards, (sequence_length, -1))
  q_t = jnp.reshape(action_values[1:], (sequence_length, -1, num_actions))
  # The actions tensor is of shape [T], and is the same for each H and W.
  # We thus expand it to be same shape as the reward tensor, [T,HW].
  expanded_actions = actions[..., None, None]
  a_tm1 = jnp.tile(expanded_actions, (1,) + height_width)
  a_tm1 = jnp.reshape(a_tm1, (sequence_length, -1))
  # We similarly expand-and-tile the discount to [T,HW].
  discount_factor = jnp.asarray(discount_factor)
  if not discount_factor.shape:
    pcont_t = jnp.reshape(discount_factor, (1,))
    pcont_t = jnp.tile(pcont_t, a_tm1.shape)
  elif len(discount_factor.shape) == 1:
    tiled_pcont = jnp.tile(
        discount_factor[:, None, None], (1,) + height_width)
    pcont_t = jnp.reshape(tiled_pcont, (sequence_length, -1))
  else:
    raise ValueError(
        "The discount_factor must be a scalar or a tensor of rank 1. "
        "instead is a tensor of shape {}".format(
            discount_factor.shape))
  # Compute a QLambda loss of shape [T,HW]
  batched_q_lambda = jax.vmap(
      functools.partial(
          value_learning.q_lambda, lambda_=1.0),
      in_axes=1, out_axes=1)
  td_error = batched_q_lambda(q_tm1, a_tm1, r_t, pcont_t, q_t)
  loss = 0.5 * td_error**2
  expanded_shape = (sequence_length,) + height_width
  spatial_loss = jnp.reshape(loss, expanded_shape)  # [T,H,W].
  return spatial_loss
