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
"""Tests for exploration.py."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from rlax._src import exploration


class GaussianTest(parameterized.TestCase):

  def setUp(self):
    super(GaussianTest, self).setUp()
    self._num_actions = 3
    self._rng_key = jax.random.PRNGKey(42)

  @chex.all_variants()
  def test_deterministic(self):
    """Check that noisy and noisless actions match for zero stddev."""
    add_noise = self.variant(exploration.add_gaussian_noise)
    # Test that noisy and noisless actions match for zero stddev
    for _ in range(10):
      action = np.random.normal(0., 1., self._num_actions)
      # Test output.
      self._rng_key, key = jax.random.split(self._rng_key)
      noisy_action = add_noise(key, action, 0.)
      np.testing.assert_allclose(action, noisy_action)


class OrnsteinUhlenbeckTest(parameterized.TestCase):

  def setUp(self):
    super(OrnsteinUhlenbeckTest, self).setUp()
    self._num_actions = 3
    self._rng_key = jax.random.PRNGKey(42)

  @chex.all_variants()
  def test_deterministic(self):
    """Check that noisy and noisless actions match for zero stddev."""
    add_noise = self.variant(exploration.add_ornstein_uhlenbeck_noise)
    # Test that noisy and noisless actions match for zero stddev
    noise_tm1 = np.zeros((self._num_actions,))
    for _ in range(10):
      action = np.random.normal(0., 1., self._num_actions)
      # Test output.
      self._rng_key, key = jax.random.split(self._rng_key)
      noisy_action = add_noise(key, action, noise_tm1, 1., 0.)
      noise_tm1 = action - noisy_action
      np.testing.assert_allclose(action, noisy_action)


class DirichletNoiseTest(parameterized.TestCase):

  def setUp(self):
    super(DirichletNoiseTest, self).setUp()
    self._batch_size = 5
    self._num_actions = 10
    self._rng_key = jax.random.PRNGKey(42)

  @chex.all_variants()
  def test_deterministic(self):
    """Check that noisy and noisless actions match for zero stddev."""
    add_noise = self.variant(exploration.add_dirichlet_noise)

    # Test that noisy and noisless actions match for zero Dirichlet noise
    for _ in range(10):
      prior = np.random.normal(0., 1., (self._batch_size, self._num_actions))

      # Test output.
      self._rng_key, key = jax.random.split(self._rng_key)
      noisy_prior = add_noise(
          key, prior, dirichlet_alpha=0.3, dirichlet_fraction=0.)
      np.testing.assert_allclose(prior, noisy_prior)


class EMIntrinsicRewardTest(parameterized.TestCase):

  def setUp(self):
    super(EMIntrinsicRewardTest, self).setUp()
    self.num_neighbors = 2
    self.reward_scale = 1.

  @chex.all_variants()
  def test_zeros(self):
    """Check that we get reward or 1 for identical state and embeddings."""

    @self.variant
    def episodic_memory_intrinsic_rewards(embeddings, reward_scale):
      return exploration.episodic_memory_intrinsic_rewards(
          embeddings, self.num_neighbors, reward_scale)

    embeddings = np.array([[0., 0.], [0., 0.]])
    reward, intrinsic_reward_state = episodic_memory_intrinsic_rewards(
        embeddings, self.reward_scale)

    np.testing.assert_array_equal(intrinsic_reward_state.memory, embeddings)
    np.testing.assert_allclose(reward, np.ones_like(reward), atol=1e-3)

  @chex.all_variants()
  def test_custom_memory(self):
    """Check that embeddings are appended to non-empty memory."""

    @self.variant
    def episodic_memory_intrinsic_rewards(embeddings, memory, reward_scale):
      return exploration.episodic_memory_intrinsic_rewards(
          embeddings, self.num_neighbors, reward_scale,
          exploration.IntrinsicRewardState(memory=memory))

    embeddings = np.array([[2., 2.]])
    memory = np.array([[
        0.,
        0.,
    ], [1., 1.]])
    _, intrinsic_reward_state = episodic_memory_intrinsic_rewards(
        embeddings, memory, self.reward_scale)

    np.testing.assert_array_equal(intrinsic_reward_state.memory,
                                  np.array([[
                                      0.,
                                      0.,
                                  ], [1., 1.], [2., 2.]]))


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
