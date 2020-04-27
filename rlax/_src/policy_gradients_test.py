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
"""Unit tests for `policy_gradients.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import policy_gradients
from rlax._src import test_util


class DpgLossTest(parameterized.TestCase):

  def setUp(self):
    super(DpgLossTest, self).setUp()

    self.s_t = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.float32)  # [B, T]
    self.w_s = np.ones([3, 2], dtype=np.float32)
    b_s = np.zeros([2], dtype=np.float32)
    # Add batch dimension to satisfy shape assertions.
    self.b_s = jnp.expand_dims(b_s, 0)
    self.w = np.ones([2, 1], dtype=np.float32)
    self.b = np.zeros([1], dtype=np.float32)

    self.expected = np.array([0.5, 0.5], dtype=np.float32)

  @test_util.parameterize_vmap_variant()
  def test_dpg_loss_batch(self, variant):
    """Tests for a full batch."""
    dpg = variant(policy_gradients.dpg_loss, dqda_clipping=1.)
    # Actor and critic function approximators.
    actor = lambda s_t: jnp.matmul(s_t, self.w_s) + self.b_s
    critic = lambda a_t: jnp.squeeze(jnp.matmul(a_t, self.w) + self.b)
    # Compute loss.
    a_t = actor(self.s_t)
    dqda = jax.vmap(jax.grad(critic))(a_t)
    # Test outputs.
    actual = np.sum(dpg(a_t, dqda), axis=1)
    np.testing.assert_allclose(actual, self.expected, atol=1e-4)


class PolicyGradientLossTest(parameterized.TestCase):

  def setUp(self):
    super(PolicyGradientLossTest, self).setUp()

    logits = np.array(
        [[1., 1., 1.], [2., 0., 0.], [-1., -2., -3.]], dtype=np.float32)
    self.logits = np.stack([logits, logits + 1.])

    weights = np.array([-2., 2., 0], dtype=np.float32)
    self.weights = np.stack([weights, weights - 1.])
    advantages = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    self.advantages = np.stack([advantages, -advantages])
    self.actions = np.array([[0, 1, 2], [0, 0, 0]], dtype=np.int32)

    self.expected = np.array([0.0788835088, 0.327200909], dtype=np.float32)

  @test_util.parameterize_vmap_variant()
  def test_policy_gradient_loss_batch(self, variant):
    """Tests for a full batch."""
    policy_gradient_loss = variant(policy_gradients.policy_gradient_loss)
    # Test outputs.
    actual = policy_gradient_loss(self.logits, self.actions, self.advantages,
                                  self.weights)
    np.testing.assert_allclose(self.expected, actual, atol=1e-4)


class EntropyLossTest(parameterized.TestCase):

  def setUp(self):
    super(EntropyLossTest, self).setUp()

    logits = np.array(
        [[1., 1., 1.], [2., 0., 0.], [-1., -2., -3.]], dtype=np.float32)
    self.logits = np.stack([logits, logits + 1.])
    weights = np.array([-2., 2., 0], dtype=np.float32)
    self.weights = np.stack([weights, weights - 1.])

    self.expected = np.array([0.288693, 1.15422], dtype=np.float32)

  @test_util.parameterize_vmap_variant()
  def test_entropy_loss_batch(self, variant):
    """Tests for a full batch."""
    entropy_loss = variant(policy_gradients.entropy_loss)
    # Test outputs.
    actual = entropy_loss(self.logits, self.weights)
    np.testing.assert_allclose(self.expected, actual, atol=1e-4)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
