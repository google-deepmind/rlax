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
"""Tests for model_learning.py."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from rlax._src import model_learning


class ModelLearningTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.trajectories = jnp.array([  # [T, B]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    ]).transpose()
    self.start_indices = jnp.array([  # [B, num_starts]
        [0, 1, 4],
        [1, 2, 5]
    ])
    self.invalid_start_indices = jnp.array([  # [B, num_starts]
        [0, 1, 25],  # 25 is out of bound
        [1, 2, 5]])

  def test_extract_subsequences(self):
    output = model_learning.extract_subsequences(
        self.trajectories, self.start_indices, 3)
    expected_output = jnp.array([
        [[0, 1, 4],
         [10, 20, 50]],
        [[1, 2, 5],
         [20, 30, 60]],
        [[2, 3, 6],
         [30, 40, 70]]])
    # Test output.
    np.testing.assert_allclose(output, expected_output)

  def test_extract_subsequences_with_validation_bounds(self):
    with self.assertRaisesRegex(AssertionError, 'Expected len >='):
      model_learning.extract_subsequences(
          self.trajectories, self.invalid_start_indices, 1,
          max_valid_start_idx=24)


if __name__ == '__main__':
  absltest.main()
