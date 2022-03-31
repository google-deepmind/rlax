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
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import tree_util

NUM_NESTS = 5


class TreeUtilTest(parameterized.TestCase):

  def test_tree_split_key(self):
    rng_key = jax.random.PRNGKey(42)
    tree_like = (1, (2, 3), {'a': 4})
    _, tree_keys = tree_util.tree_split_key(rng_key, tree_like)
    self.assertLen(jax.tree_leaves(tree_keys), 4)

  def test_tree_map_zipped(self):
    nests = [
        dict(a=jnp.zeros((1, 3)), b=jnp.zeros((1, 5)))] * NUM_NESTS
    nest_output = tree_util.tree_map_zipped(
        lambda *args: jnp.concatenate(args), nests)
    self.assertEqual(nest_output['a'].shape, (NUM_NESTS, 3))
    self.assertEqual(nest_output['b'].shape, (NUM_NESTS, 5))

  def test_tree_map_zipped_wrong_structure(self):
    nests = [
        dict(a=jnp.zeros((1, 3)), b=jnp.zeros((1, 5)))] * (NUM_NESTS - 1)
    nests.append(dict(c=jnp.zeros((1, 3))))  # add a non-matching nest
    with self.assertRaisesRegex(ValueError, 'must share the same tree'):
      tree_util.tree_map_zipped(
          lambda *args: jnp.concatenate(args), nests)

  def test_tree_map_zipped_empty(self):
    outputs = tree_util.tree_map_zipped(lambda *args: jnp.concatenate(args), [])
    self.assertEmpty(outputs)

  def test_select_true(self):
    on_true = ((jnp.zeros(3,),), jnp.zeros(4,))
    on_false = ((jnp.ones(3,),), jnp.ones(4,))
    output = tree_util.tree_select(True, on_true, on_false)
    chex.assert_tree_all_close(output, on_true)

  def test_select_false(self):
    on_true = ((jnp.zeros(3,),), jnp.zeros(4,))
    on_false = ((jnp.ones(3,),), jnp.ones(4,))
    output = tree_util.tree_select(False, on_true, on_false)
    chex.assert_tree_all_close(output, on_false)

  @parameterized.parameters(False, True)
  def test_tree_split_leaves(self, use_jnp: bool):
    if use_jnp:
      np_ver = jnp
      assert_type = lambda x: self.assertIsInstance(x, jax.xla.DeviceArray)
    else:
      np_ver = np
      assert_type = lambda x: self.assertNotIsInstance(x, jax.xla.DeviceArray)

    t = {
        'a0': np_ver.zeros(3),
        'd': {
            'a1': np_ver.arange(3),
            'a2': np_ver.zeros([3, 3]) + 2,
        },
        't4': (np_ver.zeros([3, 2]) + 4, np_ver.arange(3)),
    }

    for keepdim in (False, True):
      expd_shapes = jax.tree_map(lambda x: np.zeros(x.shape[1:]), t)
      if keepdim:
        expd_shapes = jax.tree_map(lambda x: np.expand_dims(x, 0), expd_shapes)

      res_trees = tree_util.tree_split_leaves(t, axis=0, keepdim=keepdim)
      self.assertLen(res_trees, 3)
      chex.assert_tree_all_equal_shapes(expd_shapes, *res_trees)
      for i, res_t in enumerate(res_trees):
        np.testing.assert_allclose(res_t['a0'], 0)
        np.testing.assert_allclose(res_t['d']['a1'], i)
        np.testing.assert_allclose(res_t['d']['a2'], 2)
        np.testing.assert_allclose(res_t['t4'][0], 4)
        np.testing.assert_allclose(res_t['t4'][1], i)
        jax.tree_map(assert_type, res_t)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
