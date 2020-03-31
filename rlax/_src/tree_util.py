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
"""Tree utilities."""

import jax

from rlax._src import base


def tree_split_key(rng_key: base.ArrayLike, tree_like):
  """Generate random keys for each leaf in a tree.

  Args:
    rng_key: a JAX pseudo random number generator key.
    tree_like: a nested structure.

  Returns:
    a new key, and a tree of keys with same shape as `tree_like`.
  """
  leaves, treedef = jax.tree_util.tree_flatten(tree_like)
  rng_key, *keys = jax.random.split(rng_key, num=len(leaves)+1)
  return rng_key, jax.tree_util.tree_unflatten(treedef, keys)
