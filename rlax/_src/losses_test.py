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
"""Tests for losses.py."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import losses


class L2LossTest(parameterized.TestCase):

  def setUp(self):
    super(L2LossTest, self).setUp()
    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    self.ys = jnp.array([2., 0.5, 0.125, 0, 0.125, 0.5, 2.])
    self.dys = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])

  @chex.all_variants()
  def test_l2_loss_scalar(self):
    l2_loss = self.variant(losses.l2_loss)
    x = jnp.array(0.5)
    # Test output.
    np.testing.assert_allclose(l2_loss(x), 0.125)

  @chex.all_variants()
  def test_l2_loss_vector(self):
    l2_loss = self.variant(losses.l2_loss)
    # Test output.
    np.testing.assert_allclose(l2_loss(self.xs), self.ys)

  @chex.all_variants()
  def test_l2_regularizer(self):
    l2_loss = self.variant(losses.l2_loss)
    # Test output.
    np.testing.assert_allclose(
        l2_loss(self.xs), l2_loss(self.xs, jnp.zeros_like(self.xs)))

  @chex.all_variants()
  def test_gradients(self):
    l2_loss = self.variant(losses.l2_loss)
    # Compute gradient in batch
    batch_grad_func = jax.vmap(jax.grad(l2_loss), (0))
    actual = batch_grad_func(self.xs)
    np.testing.assert_allclose(actual, self.dys)


class LogLossTest(parameterized.TestCase):

  def setUp(self):
    super(LogLossTest, self).setUp()
    self.preds = jnp.array([1., 1., 0., 0., 0.5, 0.5])
    self.targets = jnp.array([1., 0., 0., 1., 1., 0])
    self.expected = jnp.array([0., np.inf, 0., np.inf, 0.6931472, 0.6931472])

  @chex.all_variants()
  def test_log_loss_scalar(self):
    log_loss = self.variant(losses.log_loss)
    preds = self.preds[2]
    targets = self.targets[2]
    # Test output.
    np.testing.assert_allclose(
        log_loss(preds, targets), self.expected[2], atol=1e-4)

  @chex.all_variants()
  def test_log_loss_vector(self):
    log_loss = self.variant(losses.log_loss)
    # Test output.
    np.testing.assert_allclose(
        log_loss(self.preds, self.targets), self.expected, atol=1e-4)


class PixelControlLossTest(parameterized.TestCase):
  """Test the `pixel_control_loss` op."""

  def setUp(self):
    """Defines example data and expected result for the op."""
    super(PixelControlLossTest, self).setUp()

    # Observation shape is (2,2,3) (i.e., height 2, width 2, and 3 channels).
    # We will use no cropping, and a cell size of 1. We have num_actions = 3,
    # meaning our Q values should be (2,2,3). We will set the Q value equal to
    # the observation.
    self.seq_length = 3
    self.discount = 0.9
    self.cell_size = 1

    # Observations.
    obs1 = np.array([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]])
    obs2 = np.array([[[7, 8, 9], [1, 2, 3]], [[3, 4, 5], [5, 6, 7]]])
    obs3 = np.array([[[5, 6, 7], [7, 8, 9]], [[1, 2, 3], [3, 4, 5]]])
    obs4 = np.array([[[3, 4, 5], [5, 6, 7]], [[7, 8, 9], [1, 2, 3]]])

    # Actions.
    action1 = 0
    action2 = 1
    action3 = 2

    # Compute loss for constant discount.
    qa_tm1 = obs3[:, :, action3]
    reward3 = np.mean(np.abs(obs4 - obs3), axis=2)
    qmax_t = np.amax(obs4, axis=2)
    target = reward3 + self.discount * qmax_t
    error3 = target - qa_tm1

    qa_tm1 = obs2[:, :, action2]
    reward2 = np.mean(np.abs(obs3 - obs2), axis=2)
    target = reward2 + self.discount * target
    error2 = target - qa_tm1

    qa_tm1 = obs1[:, :, action1]
    reward1 = np.mean(np.abs(obs2 - obs1), axis=2)
    target = reward1 + self.discount * target
    error1 = target - qa_tm1

    # Compute loss for episode termination with discount 0.
    qa_tm1 = obs1[:, :, action1]
    reward1 = np.mean(np.abs(obs2 - obs1), axis=2)
    target = reward1 + 0. * target
    error1_term = target - qa_tm1

    self.error = np.sum(
        np.square(error1) + np.square(error2) + np.square(error3)) * 0.5
    self.error_term = np.sum(
        np.square(error1_term) + np.square(error2) + np.square(error3)) * 0.5

    self.observations = np.stack([obs1, obs2, obs3, obs4], axis=0).astype(
        np.float32)
    self.action_values = self.observations
    self.actions = np.array([action1, action2, action3])

  @chex.all_variants()
  def testPixelControlLossScalarDiscount(self):
    """Compute loss for given observations, actions, values, scalar discount."""
    loss_fn = self.variant(functools.partial(
        losses.pixel_control_loss, cell_size=self.cell_size))
    loss = loss_fn(
        self.observations,
        self.actions,
        self.action_values,
        self.discount)
    loss = jnp.sum(loss)
    np.testing.assert_allclose(loss, self.error, rtol=1e-3)

  @chex.all_variants()
  def testPixelControlLossTensorDiscount(self):
    """Compute loss for given observations, actions, values, tensor discount."""
    zero_discount = np.zeros((1,))
    non_zero_discount = self.discount * np.ones(self.seq_length - 1)
    discount = np.concatenate([zero_discount, non_zero_discount], axis=0)
    loss_fn = self.variant(functools.partial(
        losses.pixel_control_loss, cell_size=self.cell_size))
    loss = loss_fn(
        self.observations,
        self.actions,
        self.action_values,
        discount)
    loss = jnp.sum(loss)
    np.testing.assert_allclose(loss, self.error_term, rtol=1e-3)

  @chex.all_variants()
  def testPixelControlLossShapes(self):
    with self.assertRaisesRegex(
        ValueError, 'Pixel Control values are not compatible'):
      loss_fn = self.variant(functools.partial(
          losses.pixel_control_loss, cell_size=self.cell_size))
      loss_fn(
          self.observations, self.actions,
          self.action_values[:, :-1], self.discount)

  @chex.all_variants()
  def testTensorDiscountShape(self):
    with self.assertRaisesRegex(
        ValueError, 'discount_factor must be a scalar or a tensor of rank 1'):
      discount = np.tile(
          np.reshape(self.discount, (1, 1)), (self.seq_length, 1))
      loss_fn = self.variant(functools.partial(
          losses.pixel_control_loss, cell_size=self.cell_size))
      loss_fn(
          self.observations, self.actions, self.action_values, discount)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
