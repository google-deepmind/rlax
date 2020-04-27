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
"""Unit tests for `clipping.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import clipping
from rlax._src import test_util


class HuberLossTest(parameterized.TestCase):

  def setUp(self):
    super(HuberLossTest, self).setUp()
    self.delta = 1.

    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    self.ys = jnp.array([1.5, 0.5, 0.125, 0, 0.125, 0.5, 1.5])
    self.dys = jnp.array([-1, -1, -0.5, 0, 0.5, 1, 1])

  @test_util.parameterize_variant()
  def test_huber_loss_scalar(self, variant):
    huber_loss = variant(clipping.huber_loss, delta=self.delta)
    x = jnp.array(0.5)
    # Test output.
    np.testing.assert_allclose(huber_loss(x), 0.125)

  @test_util.parameterize_variant()
  def test_huber_loss_vector(self, variant):
    huber_loss = variant(clipping.huber_loss, delta=self.delta)
    xs = self.xs
    # Compute transformation.
    actual = huber_loss(xs)
    # test output.
    np.testing.assert_allclose(actual, self.ys)

  @test_util.parameterize_variant()
  def test_gradients(self, variant):
    huber_loss = variant(clipping.huber_loss, delta=self.delta)
    xs = self.xs
    # Compute gradient in batch
    batch_grad_func = jax.vmap(jax.grad(huber_loss), (0))
    actual = batch_grad_func(xs)
    np.testing.assert_allclose(actual, self.dys)


class ClipGradientsTest(parameterized.TestCase):

  def setUp(self):
    super(ClipGradientsTest, self).setUp()
    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])

  @test_util.parameterize_variant()
  def test_clip_gradient(self, variant):
    clip_gradient = variant(clipping.clip_gradient)
    x = jnp.array(0.5)
    # Test output.
    actual = clip_gradient(x, -1., 1.)
    np.testing.assert_allclose(actual, 0.5)

  @test_util.parameterize_variant()
  def test_clip_gradient_vector(self, variant):
    clip_gradient = variant(clipping.clip_gradient)
    xs = self.xs
    # Test output.
    actual = clip_gradient(xs, -1., 1.)
    np.testing.assert_allclose(actual, self.xs)


class EquivalenceTest(parameterized.TestCase):

  def setUp(self):
    super(EquivalenceTest, self).setUp()
    self.large_delta = 5.
    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])

  @test_util.parameterize_variant(
      ('10', 10.),
      ('0.5', 0.5))
  def test_clip_huber_equivalence(self, td_error, variant):
    def td_error_with_clip(x):
      return 0.5 * jnp.square(
          clipping.clip_gradient(x, -self.large_delta, self.large_delta))
    def td_error_with_huber(x):
      return clipping.huber_loss(x, self.large_delta)
    td_error_with_clip = variant(td_error_with_clip)
    td_error_with_huber = variant(td_error_with_huber)
    td_error = jnp.array(td_error)
    # Compute gradient in batch
    clip_grad = jax.grad(td_error_with_clip)(td_error)
    huber_grad = jax.grad(td_error_with_huber)(td_error)
    np.testing.assert_allclose(clip_grad, huber_grad)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
