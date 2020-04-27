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
"""Tests for `tree_util.py`."""

from absl.testing import absltest
import jax
from rlax._src import tree_util


class TreeUtilTest(absltest.TestCase):

  def test_tree_split_key(self):
    rng_key = jax.random.PRNGKey(42)
    tree_like = (1, (2, 3), {'a': 4})
    _, tree_keys = tree_util.tree_split_key(rng_key, tree_like)
    assert len(jax.tree_util.tree_leaves(tree_keys)) == 4


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
