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
"""Unit tests for `base.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import base


class RankAssertTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('rank_1', [1, 2], 2),
      ('rank_2', [[1, 2], [3, 4]], 1),
      ('rank_3', [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[1, 2, 4]]),
  )
  def test_rank_assert_should_raise_single_input(self, array, wrong_rank):
    with self.assertRaises(ValueError):
      base.rank_assert(jnp.array(array), wrong_rank)

  @parameterized.named_parameters(
      ('rank_1', [1, 2], 2),
      ('rank_2', [[1, 2], [3, 4]], 1),
      ('rank_3', [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 4),
  )
  def test_rank_assert_should_raise_list_input_single_rank(
      self, array, wrong_rank):
    with self.assertRaises(ValueError):
      base.rank_assert([jnp.array(array)] * 3, wrong_rank)

  def test_rank_assert_should_raise_list_input_list_rank(self):
    with self.assertRaises(ValueError):
      base.rank_assert(
          [jnp.array([1, 2]), jnp.array([[1, 2], [3, 4]])], [2, 2])
      base.rank_assert(
          [jnp.array([1, 2]), jnp.array([[1, 2], [3, 4]])], [[1, 3], 2])
      base.rank_assert(
          [jnp.array([1, 2]), jnp.array([[1, 2], [3, 4]])], [[1, 2], 3])

  @parameterized.named_parameters(
      ('rank_1', [1, 2], 1),
      ('rank_2', [[1, 2], [3, 4]], 2),
      ('rank_3', [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[1, 2, 3]]),
  )
  def test_rank_assert_should_not_raise_single_input(self, array, correct_rank):
    base.rank_assert(jnp.array(array), correct_rank)

  @parameterized.named_parameters(
      ('rank_1', [1, 2], 1),
      ('rank_2', [[1, 2], [3, 4]], 2),
      ('rank_3', [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 3),
  )
  def test_rank_assert_should_not_raise_list_input_single_rank(
      self, array, correct_rank):
    base.rank_assert([jnp.array(array)] * 3, correct_rank)

  def test_rank_assert_should_not_raise_list_input_list_rank(self):
    base.rank_assert(
        [jnp.array([1, 2]), jnp.array([[1, 2], [3, 4]])], [1, 2])
    base.rank_assert(
        [jnp.array([1, 2]), jnp.array([[1, 2], [3, 4]])], [[1, 2], 2])

  def test_rank_assert_should_raise_different_length(self):
    with self.assertRaises(ValueError):
      base.rank_assert(
          [jnp.array([1, 2])], [2, 2])


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


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
