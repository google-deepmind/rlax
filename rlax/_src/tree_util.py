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
"""Tree utilities."""

import functools
from typing import Any, Callable, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import base

Array = chex.Array
tree_structure = jax.tree_util.tree_structure


def tree_select(pred: Array, on_true: Any, on_false: Any):
  """Select either one of two identical nested structs based on condition.

  Args:
    pred: a boolean condition.
    on_true: an arbitrary nested structure.
    on_false: a nested structure identical to `on_true`.

  Returns:
    the selected nested structure.
  """
  if tree_structure(on_true) != tree_structure(on_false):
    raise ValueError('The two branches must have the same structure.')
  return jax.tree_util.tree_map(
      lambda x, y: jax.lax.select(pred, x, y), on_true, on_false
  )


def tree_map_zipped(fn: Callable[..., Any], nests: Sequence[Any]):
  """Map a function over a list of identical nested structures.

  Args:
    fn: the function to map; must have arity equal to `len(list_of_nests)`.
    nests: a list of identical nested structures.

  Returns:
    a nested structure whose leaves are outputs of applying `fn`.
  """
  if not nests:
    return nests
  tree_def = tree_structure(nests[0])
  if any(tree_structure(x) != tree_def for x in nests[1:]):
    raise ValueError('All elements must share the same tree structure.')
  return jax.tree_util.tree_unflatten(
      tree_def, [fn(*d) for d in zip(*[jax.tree.leaves(x) for x in nests])])


def tree_split_key(rng_key: Array, tree_like: Any):
  """Generate random keys for each leaf in a tree.

  Args:
    rng_key: a JAX pseudo random number generator key.
    tree_like: a nested structure.

  Returns:
    a new key, and a tree of keys with same shape as `tree_like`.
  """
  leaves, treedef = jax.tree_util.tree_flatten(tree_like)
  rng_key, *keys = jax.random.split(rng_key, num=len(leaves) + 1)
  return rng_key, jax.tree_util.tree_unflatten(treedef, keys)


def tree_split_leaves(tree_like: Any,
                      axis: int = 0,
                      keepdim: bool = False):
  """Splits a tree of arrays into an array of trees avoiding data copying.

  Note: `jax.numpy.DeviceArray`'s data gets copied.

  Args:
    tree_like: a nested object with leaves to split.
    axis: an axis for splitting.
    keepdim: a bool indicating whether to keep `axis` dimension.

  Returns:
   A tuple of `size(axis)` trees containing results of splitting.
  """

  # Disable pylint to correctly process `np.ndarray`s.
  if len(tree_like) == 0:  # pylint: disable=g-explicit-length-test
    return tree_like
  leaves, treedef = jax.tree_util.tree_flatten(tree_like)
  axis_size = leaves[0].shape[axis]
  split_leaves = [np.split(l, axis_size, axis=axis) for l in leaves]
  ind_ = lambda x, i: x[i] if keepdim else np.squeeze(x[i], axis)
  split_trees = ((ind_(l, i) for l in split_leaves) for i in range(axis_size))
  return tuple(jax.tree_util.tree_unflatten(treedef, t) for t in split_trees)


def tree_replace_masked(tree_data, tree_replacement, mask):
  """Replace slices of the leaves when mask is 1.

  Args:
    tree_data: a nested object with leaves to mask.
    tree_replacement: nested object with the same structure of `tree_data`,
      that cointains the data to insert according to `mask`. If `None`,
      then the masked elements in `tree_data` will be replaced with zeros.
    mask: a mask of 0/1s, whose shape is a prefix of the shape of the leaves
      in `tree_data` and in `tree_replacement`.

  Returns:
    the updated tensor.
  """
  if tree_replacement is None:
    tree_replacement = jax.tree.map(jnp.zeros_like, tree_data)
  return jax.tree.map(
      lambda data, replacement: base.replace_masked(data, replacement, mask),
      tree_data, tree_replacement)


def tree_fn(fn, **unmapped_kwargs):
  """Wrap a function to jax.tree.map over its arguments.

  You may set some named arguments via a partial to skip the `tree_map` on those
  arguments. Usual caveats of `partial` apply (e.g. set named args must be a
  suffix of the argument list).

  Args:
    fn: the function to be wrapped.
    **unmapped_kwargs: the named arguments to be set via a partial.

  Returns:
    a function
  """
  pfn = functools.partial(fn, **unmapped_kwargs)
  def _wrapped(*args):
    return jax.tree.map(pfn, *args)
  return _wrapped


def transpose_last_axis_to_first(tree: chex.ArrayTree) -> chex.ArrayTree:
  """Function to transpose the last axis to be first for all leaves in a pytree.

  This function will transpose the last axis to the front for all leaves in a
  pytree; each leaf with shape [D_1, ..., D_{n-1}, D_n] will have shape
  [D_n, D_1, ..., D_{n-1}].

  Args:
    tree: the pytree of Arrays to be transposed.

  Returns:
    tree: the transposed output tree.
  """
  def _transpose(tree):
    return jnp.transpose(tree, [tree.ndim - 1] + list(range(tree.ndim - 1)))
  return tree_fn(_transpose)(tree)


def transpose_first_axis_to_last(tree: chex.ArrayTree) -> chex.ArrayTree:
  """Function to transpose the first axis to be last for all leaves in a pytree.

  This function will transpose the first axis to the last dim of all leaves in a
  pytree; each leaf with shape [D_1, D_2, ..., D_n] will have shape
  [D_2, ..., D_n, D_1].

  Args:
    tree: the pytree of Arrays to be transposed.

  Returns:
    tree: the transposed output tree.
  """
  def _transpose(tree):
    return jnp.transpose(tree, list(range(1, tree.ndim)) + [0])
  return tree_fn(_transpose)(tree)
