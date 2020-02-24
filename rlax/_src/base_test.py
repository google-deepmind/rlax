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


class TypeAssertTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('one_float', 3., int),
      ('one_int', 3, float),
      ('many_floats', [1., 2., 3.], int),
      ('many_floats_verbose', [1., 2., 3.], [float, float, int]),
  )
  def test_type_assert_should_raise_single_input(self, array, wrong_type):
    with self.assertRaises(ValueError):
      base.type_assert(array, wrong_type)

  @parameterized.named_parameters(
      ('one_float_array_np', [1., 2.], int, False),
      ('one_int_array_np', [1, 2], float, False),
      ('one_float_array_jax', [1., 2.], int, True),
      ('one_int_array_jax', [1, 2], float, True),
  )
  def test_type_assert_should_raise_array(self, array, wrong_type, is_jax):
    array = jnp.asarray(array) if is_jax else np.asarray(array)
    with self.assertRaises(ValueError):
      base.type_assert(array, wrong_type)

  @parameterized.named_parameters(
      ('one_float', 3., float),
      ('one_int', 3, int),
      ('many_floats', [1., 2., 3.], float),
      ('many_floats_verbose', [1., 2., 3.], [float, float, float]),
  )
  def test_type_assert_should_not_raise_single_input(self, array, wrong_type):
    base.type_assert(array, wrong_type)

  @parameterized.named_parameters(
      ('one_float_array_np', [1., 2.], float, False),
      ('one_int_array_np', [1, 2], int, False),
      ('one_float_array_jax', [1., 2.], float, True),
      ('one_int_array_jax', [1, 2], int, True),
  )
  def test_type_assert_should_not_raise_array(self, array, wrong_type, is_jax):
    array = jnp.asarray(array) if is_jax else np.asarray(array)
    base.type_assert(array, wrong_type)

  def test_mixed_inputs_should_raise(self):
    a_float = 1.
    an_int = 2
    a_np_float = np.asarray([3., 4.])
    a_jax_int = jnp.asarray([5, 6])
    with self.assertRaises(ValueError):
      base.type_assert(
          [a_float, an_int, a_np_float, a_jax_int], [float, int, float, float])

  def test_mixed_inputs_should_not_raise(self):
    a_float = 1.
    an_int = 2
    a_np_float = np.asarray([3., 4.])
    a_jax_int = jnp.asarray([5, 6])
    base.type_assert(
        [a_float, an_int, a_np_float, a_jax_int], [float, int, float, int])

  def test_different_length_should_raise(self):
    a_float = 1.
    an_int = 2
    a_np_float = np.asarray([3., 4.])
    a_jax_int = jnp.asarray([5, 6])
    with self.assertRaises(ValueError):
      base.type_assert(
          [a_float, an_int, a_np_float, a_jax_int], [int, float, int])

  def test_unsupported_type_should_raise(self):
    a_float = 1.
    an_int = 2
    a_np_float = np.asarray([3., 4.])
    a_jax_int = jnp.asarray([5, 6])
    with self.assertRaises(ValueError):
      base.type_assert(
          [a_float, an_int, a_np_float, a_jax_int],
          [np.complex, np.complex, float, int])


class OneHotTest(parameterized.TestCase):
  """Testing base.one_hot"""

  def test_one_hot(self):
    indices = jnp.array([[[1.,2.,3.],
                          [1.,2.,2.]]])
    num_classes = 3
    expected_result = jnp.array([[[[0., 1., 0.],
                                   [0., 0., 1.],
                                   [0., 0., 0.]],
                                  [[0., 1., 0.],
                                   [0., 0., 1.],
                                   [0., 0., 1.]]]])
    result = base.one_hot(indices, num_classes)
    np.testing.assert_array_almost_equal(result, expected_result)

if __name__ == '__main__':
  absltest.main()
