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
"""Tests for embedding.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from rlax._src import embedding


class EmbeddingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._features = np.array([[1., 2.], [3., 2.]])
    self._num_actions = 3
    self._actions = np.array([1, 2])
    self._rewards = np.array([1., 1])

  def test_embed_zeros(self):
    # Embedding zero feature array.
    zero_features = np.zeros_like(self._features)
    emb = embedding.embed_oar(zero_features, self._actions, self._rewards,
                              self._num_actions)
    np.testing.assert_array_equal(emb[:, :self._features.shape[-1]],
                                  zero_features)

  def test_embed_shape(self):
    # Test output shape [T?, B, D+A+1].
    emb = embedding.embed_oar(self._features, self._actions, self._rewards,
                              self._num_actions)
    np.testing.assert_array_equal(
        emb.shape[-1], self._features.shape[-1] + self._num_actions + 1)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
