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
"""Tests for episodic_memory.py."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from rlax._src import episodic_memory


class KNNQueryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.data = np.array([[0., 0.], [7.5, 1.], [40., 40.]])
    self.query_points = np.array([[2.0, 1.3], [7.5, 0.0]])

  @chex.all_variants()
  def test_small_k_query(self):
    num_neighbors = 2
    expected_neighbors = np.array([[[0., 0.], [7.5, 1.]],
                                   [[7.5, 1.], [0., 0.]]])
    expected_neg_distances = np.array([[-5.69, -30.34], [-1., -56.25]])
    expected_neighbor_indices = np.array([[0., 1.], [1., 0.]])

    @self.variant
    def query_variant(data, points):
      return episodic_memory.knn_query(data, points, num_neighbors)
    actual = query_variant(self.data, self.query_points)

    np.testing.assert_allclose(actual.neighbors,
                               expected_neighbors,
                               atol=1e-6)
    np.testing.assert_allclose(actual.neighbor_indices,
                               expected_neighbor_indices,
                               atol=1e-6)
    np.testing.assert_allclose(actual.neighbor_neg_distances,
                               expected_neg_distances,
                               atol=1e-6)

  @chex.all_variants()
  @parameterized.named_parameters(('3neighbors', 3),
                                  ('5neighbors', 5))
  def test_big_k_query(self, num_neighbors):
    expected_neighbors = np.array([[[0., 0.], [7.5, 1.], [40., 40.]],
                                   [[7.5, 1.], [0., 0.], [40., 40.]]])
    expected_neg_distances = np.array([[-5.69, -30.34, -2941.69],
                                       [-1., -56.25, -2656.25]])
    expected_neighbor_indices = np.array([[0, 1, 2], [1, 0, 2],])

    @self.variant
    def query_variant(data, points):
      return episodic_memory.knn_query(data, points, num_neighbors)
    actual = query_variant(self.data, self.query_points)

    np.testing.assert_allclose(actual.neighbors,
                               expected_neighbors,
                               atol=1e-6)
    np.testing.assert_allclose(actual.neighbor_indices,
                               expected_neighbor_indices,
                               atol=1e-6)
    np.testing.assert_allclose(actual.neighbor_neg_distances,
                               expected_neg_distances,
                               atol=1e-6)

  def test_empty(self):
    data = np.array([])
    self.query_points = np.array([])
    with self.assertRaises(AssertionError):
      episodic_memory.knn_query(data, self.query_points, num_neighbors=2)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
