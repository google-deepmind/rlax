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
"""Tests for losses.py."""

from absl.testing import absltest
from absl.testing import parameterized
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

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_l2_loss_scalar(self, compile_fn, place_fn):
    # Optionally compile.
    l2_loss = compile_fn(losses.l2_loss)
    # Optionally convert to device array.
    x = place_fn(jnp.array(0.5))
    # Test output.
    np.testing.assert_allclose(l2_loss(x), 0.125)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_l2_loss_vector(self, compile_fn, place_fn):
    # Optionally compile.
    l2_loss = compile_fn(losses.l2_loss)
    # Optionally convert to device array.
    xs = place_fn(self.xs)
    # Test output.
    np.testing.assert_allclose(l2_loss(xs), self.ys)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_l2_regularizer(self, compile_fn, place_fn):
    # Optionally compile.
    l2_loss = compile_fn(losses.l2_loss)
    # Optionally convert to device array.
    xs = place_fn(self.xs)
    # Test output.
    np.testing.assert_allclose(l2_loss(xs), l2_loss(xs, jnp.zeros_like(xs)))

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_gradients(self, compile_fn, place_fn):
    # Optionally compile.
    l2_loss = compile_fn(losses.l2_loss)
    # Optionally convert to device array.
    xs = place_fn(self.xs)
    # Compute gradient in batch
    batch_grad_func = jax.vmap(jax.grad(l2_loss), (0))
    actual = batch_grad_func(xs)
    np.testing.assert_allclose(actual, self.dys)


class LogLossTest(parameterized.TestCase):

  def setUp(self):
    super(LogLossTest, self).setUp()
    self.preds = jnp.array([1., 1., 0., 0., 0.5, 0.5])
    self.targets = jnp.array([1., 0., 0., 1., 1., 0])
    self.expected = jnp.array([0., np.inf, 0., np.inf, 0.6931472, 0.6931472])

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_log_loss_scalar(self, compile_fn, place_fn):
    # Optionally compile.
    log_loss = compile_fn(losses.log_loss)
    # Optionally convert to device array.
    preds = place_fn(self.preds[2])
    targets = place_fn(self.targets[2])
    # Test output.
    np.testing.assert_allclose(
        log_loss(preds, targets), self.expected[2], atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_log_loss_vector(self, compile_fn, place_fn):
    # Optionally compile.
    log_loss = compile_fn(losses.log_loss)
    # Optionally convert to device array.
    preds = place_fn(self.preds)
    targets = place_fn(self.targets)
    # Test output.
    np.testing.assert_allclose(
        log_loss(preds, targets), self.expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
