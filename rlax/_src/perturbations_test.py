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
"""Tests for perturbations.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from rlax._src import perturbations
from rlax._src import test_util


class GaussianTest(parameterized.TestCase):

  def setUp(self):
    super(GaussianTest, self).setUp()
    self._num_actions = 3
    self._rng_key = jax.random.PRNGKey(42)

  @test_util.parameterize_variant()
  def test_deterministic(self, variant):
    """Check that noisy and noisless actions match for zero stddev."""
    add_noise = variant(perturbations.add_gaussian_noise)
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

  @test_util.parameterize_variant()
  def test_deterministic(self, variant):
    """Check that noisy and noisless actions match for zero stddev."""
    add_noise = variant(perturbations.add_ornstein_uhlenbeck_noise)
    # Test that noisy and noisless actions match for zero stddev
    noise_tm1 = np.zeros((self._num_actions,))
    for _ in range(10):
      action = np.random.normal(0., 1., self._num_actions)
      # Test output.
      self._rng_key, key = jax.random.split(self._rng_key)
      noisy_action = add_noise(key, action, noise_tm1, 1., 0.)
      noise_tm1 = action - noisy_action
      np.testing.assert_allclose(action, noisy_action)


if __name__ == '__main__':
  absltest.main()
