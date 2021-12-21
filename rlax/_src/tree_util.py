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

from typing import Any, Callable, Optional, Sequence
import chex
import jax
import jax.numpy as jnp


def tree_select(pred: chex.Array, on_true: Any, on_false: Any):
  """Select either one of two identical nested structs based on condition.

  Args:
    pred: a boolean condition.
    on_true: an arbitrary nested structure.
    on_false: a nested structure identical to `on_true`.

  Returns:
    the selected nested structure.
  """
  if jax.tree_structure(on_true) != jax.tree_structure(on_false):
    raise ValueError('The two branches must have the same structure.')
  return jax.tree_multimap(
      lambda x, y: jax.lax.select(pred, x, y), on_true, on_false)


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
  tree_def = jax.tree_structure(nests[0])
  if any(jax.tree_structure(x) != tree_def for x in nests[1:]):
    raise ValueError('All elements must share the same tree structure.')
  return jax.tree_unflatten(
      tree_def, [fn(*d) for d in zip(*[jax.tree_leaves(x) for x in nests])])


def tree_split_key(rng_key: chex.Array, tree_like: Any):
  """Generate random keys for each leaf in a tree.

  Args:
    rng_key: a JAX pseudo random number generator key.
    tree_like: a nested structure.

  Returns:
    a new key, and a tree of keys with same shape as `tree_like`.
  """
  leaves, treedef = jax.tree_flatten(tree_like)
  rng_key, *keys = jax.random.split(rng_key, num=len(leaves) + 1)
  return rng_key, jax.tree_unflatten(treedef, keys)


def tree_split_leaves(tree_like: Any,
                      axis: int = 0,
                      keepdim: bool = False,
                      num_splits: Optional[int] = None):
  """Splits a tree of arrays into a sequence of trees avoiding data copying.

  Note: `jax.numpy.DeviceArray`'s data gets copied.

  Args:
    tree_like: a nested object with leaves to split.
    axis: an axis for splitting.
    keepdim: a bool indicating whether to keep `axis` dimension.
    num_splits: number of splits to partition data into along axis `axis`.

  Returns:
   A tuple of `num_splits` trees containing results of splitting.
  """

  # Disable pylint to correctly process `np.ndarray`s.
  if len(tree_like) == 0:  # pylint: disable=g-explicit-length-test
    return tree_like
  leaves, treedef = jax.tree_flatten(tree_like)
  if num_splits is None:
    num_splits = leaves[0].shape[axis]
  if num_splits != leaves[0].shape[axis] and not keepdim:
    raise ValueError(
        '`keepdim` must be True when not splitting into singleton chuncks.')
  split_leaves = [jnp.split(l, num_splits, axis=axis) for l in leaves]
  ind_ = lambda x, i: x[i] if keepdim else x[i][0]
  split_trees = ((ind_(l, i) for l in split_leaves) for i in range(num_splits))
  return tuple(jax.tree_unflatten(treedef, t) for t in split_trees)
