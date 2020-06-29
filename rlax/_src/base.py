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
"""Common utilities for RLax functions."""

import chex
import jax.numpy as jnp

Array = chex.Array


def batched_index(
    values: Array, indices: Array, keepdims: bool = False
) -> Array:
  """Index into the last dimension of a tensor, preserving all others dims.

  Args:
    values: a tensor of shape [..., D],
    indices: indices of shape [...].
    keepdims: whether to keep the final dimension.

  Returns:
    a tensor of shape [...] or [..., 1].
  """
  indexed = jnp.take_along_axis(values, indices[..., None], axis=-1)
  if not keepdims:
    indexed = jnp.squeeze(indexed, axis=-1)
  return indexed


def one_hot(indices, num_classes, dtype=jnp.float32):
  """Returns a one-hot version of indices.

  Args:
    indices: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: The dtype.

  Returns:
    The one-hot tensor. If indices' shape is [A, B, ...], shape is
      [A, B, ..., num_classes].
  """
  labels = jnp.arange(num_classes)
  for _ in range(indices.ndim):
    labels = jnp.expand_dims(labels, axis=0)
  return jnp.array(
      indices[..., jnp.newaxis] == labels, dtype=dtype)
