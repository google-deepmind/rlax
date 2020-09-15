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

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import distributions
from rlax._src import policy_gradients


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

  @chex.all_variants()
  def test_dpg_loss_batch(self):
    """Tests for a full batch."""
    dpg = self.variant(jax.vmap(functools.partial(
        policy_gradients.dpg_loss, dqda_clipping=1.)))
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

  @chex.all_variants()
  def test_policy_gradient_loss_batch(self):
    """Tests for a full batch."""
    policy_gradient_loss = self.variant(jax.vmap(
        policy_gradients.policy_gradient_loss))
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

  @chex.all_variants()
  def test_entropy_loss_batch(self):
    """Tests for a full batch."""
    entropy_loss = self.variant(jax.vmap(policy_gradients.entropy_loss))
    # Test outputs.
    actual = entropy_loss(self.logits, self.weights)
    np.testing.assert_allclose(self.expected, actual, atol=1e-4)


class QPGLossTest(parameterized.TestCase):

  def setUp(self):
    super(QPGLossTest, self).setUp()

    self.q_values = jnp.array([[0., -1., 1.], [1., -1., 0]])
    self.policy_logits = jnp.array([[1., 1., 1.], [1., 1., 4.]])

    # baseline = \sum_a pi_a * Q_a = 0.
    # -\sum_a pi_a * relu(Q_a - baseline)
    # negative sign as it's a loss term and loss needs to be minimized.
    self.expected_policy_loss = (0.0 + 0.0) / 2

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_qpg_loss_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    qpg_loss = compile_fn(policy_gradients.qpg_loss)

    # Optionally convert to device array.
    policy_logits, q_values = jax.tree_map(place_fn,
                                           (self.policy_logits, self.q_values))
    # Test outputs.
    actual = qpg_loss(policy_logits, q_values)
    np.testing.assert_allclose(self.expected_policy_loss, actual, atol=1e-4)


class RMLossTest(parameterized.TestCase):

  def setUp(self):
    super(RMLossTest, self).setUp()
    self.q_values = jnp.array([[0., -1., 1.], [1., -1., 0]])
    self.policy_logits = jnp.array([[1., 1., 1.], [1., 1., 4.]])

    # baseline = \sum_a pi_a * Q_a = 0.
    # -\sum_a pi_a * relu(Q_a - baseline)
    # negative sign as it's a loss term and loss needs to be minimized.
    self.expected_policy_loss = -(.3333 + .0452) / 2

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_rm_loss_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    rm_loss = compile_fn(policy_gradients.rm_loss)

    # Optionally convert to device array.
    policy_logits, q_values = jax.tree_map(place_fn,
                                           (self.policy_logits, self.q_values))
    # Test outputs.
    actual = rm_loss(policy_logits, q_values)
    np.testing.assert_allclose(self.expected_policy_loss, actual, atol=1e-4)


class RPGLossTest(parameterized.TestCase):

  def setUp(self):
    super(RPGLossTest, self).setUp()

    self.q_values = jnp.array([[0., -1., 1.], [1., -1., 0]])
    self.policy_logits = jnp.array([[1., 1., 1.], [1., 1., 4.]])

    # baseline = \sum_a pi_a * Q_a = 0.
    # -\sum_a pi_a * relu(Q_a - baseline)
    # negative sign as it's a loss term and loss needs to be minimized.
    self.expected_policy_loss = (1.0 + 1.0) / 2

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_rpg_loss(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    rpg_loss = compile_fn(policy_gradients.rpg_loss)

    # Optionally convert to device array.
    policy_logits, q_values = jax.tree_map(place_fn,
                                           (self.policy_logits, self.q_values))
    # Test outputs.
    actual = rpg_loss(policy_logits, q_values)
    np.testing.assert_allclose(self.expected_policy_loss, actual, atol=1e-4)


class ClippedSurrogatePGLossTest(parameterized.TestCase):

  def setUp(self):
    super(ClippedSurrogatePGLossTest, self).setUp()

    logits = np.array(
        [[1., 1., 1.], [2., 0., 0.], [-1., -2., -3.]], dtype=np.float32)
    old_logits = np.array(
        [[1., 1., 1.], [2., 0., 0.], [-3., -2., -1.]], dtype=np.float32)
    self.logits = np.stack([logits, logits])
    self.old_logits = np.stack([old_logits, old_logits])

    advantages = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    self.advantages = np.stack([advantages, -advantages])
    self.actions = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    self.epsilon = 0.2
    self.expected = np.array([-0.17117467, 0.19333333])

  @chex.all_variants()
  def test_clipped_surrogate_pg_loss_batch(self):
    """Tests for a full batch."""
    get_ratios = jax.vmap(distributions.categorical_importance_sampling_ratios)
    prob_ratios = get_ratios(self.logits, self.old_logits, self.actions)
    batched_fn_variant = self.variant(jax.vmap(functools.partial(
        policy_gradients.clipped_surrogate_pg_loss, epsilon=self.epsilon)))
    actual = batched_fn_variant(prob_ratios, self.advantages)
    np.testing.assert_allclose(actual, self.expected, atol=1e-4)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
