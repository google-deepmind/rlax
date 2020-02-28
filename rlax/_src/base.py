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

from typing import List, Type, Union
import jax.numpy as jnp
import numpy as np

Scalar = Union[float, int]
ArrayLike = Union[Scalar, np.ndarray, jnp.DeviceArray]


def batched_index(
    values: ArrayLike, indices: ArrayLike, keepdims: bool = False) -> ArrayLike:
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
  return jnp.array(
      indices[..., jnp.newaxis] == jnp.arange(num_classes), dtype=dtype)


def rank_assert(
    inputs: Union[ArrayLike, List[ArrayLike]],
    expected_ranks: Union[int, List[Union[int, List[int]]]]):
  """Checks that the rank of all inputs matches specified expected_ranks.

  Args:
    inputs: list of inputs.
    expected_ranks: list of expected ranks associated with each input, where the
      expected rank is either an integer or list of integer options; if all
      inputs have same rank, a single scalar may be passed as `expected_ranks`.

  Raises:
    ValueError: if the length of inputs and expected_ranks do not match.
  """
  if not isinstance(inputs, list):
    inputs = [inputs]
  if not isinstance(expected_ranks, list):
    expected_ranks = [expected_ranks] * len(inputs)
  if len(inputs) != len(expected_ranks):
    raise ValueError("Length of inputs and expected_ranks must match.")
  for idx, (x, expected) in enumerate(zip(inputs, expected_ranks)):
    if hasattr(x, "shape"):
      shape = x.shape
    else:
      shape = ()  # scalars have shape () by definition.
    rank = len(shape)

    expected_as_list = expected if isinstance(expected, list) else [expected]

    if rank not in expected_as_list:
      raise ValueError(
          "Error in rank compatibility check: input {} has rank {} "
          "(shape {}) but expected {}.".format(idx, rank, shape, expected))


def type_assert(
    inputs: Union[ArrayLike, List[ArrayLike]],
    expected_types: Union[Type[Scalar], List[Type[Scalar]]]):
  """Checks that the type of all inputs matches specified expected_types.

  Args:
    inputs: list of inputs.
    expected_types: list of expected types associated with each input; if all
      inputs have same type, a single type may be passed as `expected_types`.

  Raises:
    ValueError: if the length of inputs and expected_types do not match.
  """
  if not isinstance(inputs, list):
    inputs = [inputs]
  if not isinstance(expected_types, list):
    expected_types = [expected_types] * len(inputs)
  if len(inputs) != len(expected_types):
    raise ValueError("Length of inputs and expected_types must match.")
  for idx, (x, expected) in enumerate(zip(inputs, expected_types)):
    if jnp.issubdtype(expected, jnp.floating):
      parent = jnp.floating
    elif jnp.issubdtype(expected, jnp.integer):
      parent = jnp.integer
    else:
      raise ValueError("Error in type compatibility check, unsupported dtype"
                       " {}".format(expected))

    if not jnp.issubdtype(jnp.result_type(x), parent):
      raise ValueError(
          "Error in type compatibility check: input {} has type {} but "
          "expected {}.".format(idx, jnp.result_type(x), expected))
