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
"""Unit tests for `base.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import base


class OneHotTest(parameterized.TestCase):

  def test_one_hot(self):
    num_classes = 3
    indices = jnp.array(
        [[[1., 2., 3.], [1., 2., 2.]]])
    expected_result = jnp.array([
        [[[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]],
         [[0., 1., 0.], [0., 0., 1.], [0., 0., 1.]]]])
    result = base.one_hot(indices, num_classes)
    np.testing.assert_array_almost_equal(result, expected_result)


class BroadcastTest(parameterized.TestCase):

  @parameterized.parameters(
      ([1], [1, 2, 3], [1, 1, 1]),
      ([1, 2, 1], [1, 2, 3], [1, 2, 1]),
      ([2, 1, 2], [2, 2, 2, 3], [2, 1, 2, 1]),
      ([1, 2, 4], [1, 2, 4], [1, 2, 4]),
  )
  def test_lhs_broadcasting(
      self, source_shape, target_shape, expected_result_shape):
    source = jnp.ones(shape=source_shape, dtype=jnp.float32)
    target = jnp.ones(shape=target_shape, dtype=jnp.float32)
    expected_result = jnp.ones(shape=expected_result_shape, dtype=jnp.float32)
    result = base.lhs_broadcast(source, target)
    np.testing.assert_array_almost_equal(result, expected_result)

  def test_lhs_broadcast_raises(self):
    source = jnp.ones(shape=(1, 2), dtype=jnp.float32)
    target = jnp.ones(shape=(1, 3, 1, 1), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, 'source shape'):
      base.lhs_broadcast(source, target)

if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
