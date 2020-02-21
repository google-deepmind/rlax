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
"""Unit tests for `transforms.py`."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import transforms


class TransformsTest(parameterized.TestCase):

  def setUp(self):
    super(TransformsTest, self).setUp()
    self.x = 0.5
    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_identity_scalar(self, compile_fn, place_fn):
    # Optionally compile.
    identity = compile_fn(transforms.identity)
    # Optionally convert to device array.
    x = place_fn(jnp.array(self.x))
    # Test output.
    np.testing.assert_allclose(identity(x), self.x)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_identity_vector(self, compile_fn, place_fn):
    # Optionally compile.
    identity = compile_fn(transforms.identity)
    # Optionally convert to device array.
    xs = place_fn(self.xs)
    # Test output.
    np.testing.assert_allclose(identity(xs), self.xs)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_sigmoid_scalar(self, compile_fn, place_fn):
    # Optionally compile.
    sigmoid = compile_fn(transforms.sigmoid)
    logit = compile_fn(transforms.logit)
    # Optionally convert to device array.
    x = place_fn(jnp.array(self.x))
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(x)), self.x, atol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_sigmoid_vector(self, compile_fn, place_fn):
    # Optionally compile.
    sigmoid = compile_fn(transforms.sigmoid)
    logit = compile_fn(transforms.logit)
    # Optionally convert to device array.
    xs = place_fn(jnp.array(self.xs))
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(xs)), self.xs, atol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_signed_log_exp_transform_scalar(self, compile_fn, place_fn):
    # Optionally compile.
    signed_logp1 = compile_fn(transforms.signed_logp1)
    signed_expm1 = compile_fn(transforms.signed_expm1)
    # Optionally convert to device array.
    x = place_fn(jnp.array(self.x))
    # Test inverse.
    np.testing.assert_allclose(signed_expm1(signed_logp1(x)), self.x, atol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_signed_log_exp_transform_vector(self, compile_fn, place_fn):
    # Optionally compile.
    signed_logp1 = compile_fn(transforms.signed_logp1)
    signed_expm1 = compile_fn(transforms.signed_expm1)
    # Optionally convert to device array.
    xs = place_fn(self.xs)
    # Test inverse.
    np.testing.assert_allclose(
        signed_expm1(signed_logp1(xs)), self.xs, atol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_signed_hyper_parabolic_transform_scalar(self, compile_fn, place_fn):
    # Optionally compile.
    signed_hyperbolic = compile_fn(transforms.signed_hyperbolic)
    signed_parabolic = compile_fn(transforms.signed_parabolic)
    # Optionally convert to device array.
    x = place_fn(jnp.array(self.x))
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(x)), self.x, atol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_signed_hyper_parabolic_transform_vector(self, compile_fn, place_fn):
    # Optionally compile.
    signed_hyperbolic = compile_fn(transforms.signed_hyperbolic)
    signed_parabolic = compile_fn(transforms.signed_parabolic)
    # Optionally convert to device array.
    xs = place_fn(self.xs)
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(xs)), self.xs, atol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_signed_power_transform_scalar(self, compile_fn, place_fn):
    # Optionally compile.
    square = compile_fn(functools.partial(transforms.power, p=2.))
    sqrt = compile_fn(functools.partial(transforms.power, p=1/2.))
    # Optionally convert to device array.
    x = place_fn(jnp.array(self.x))
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(x)), self.x, atol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_signed_power_transform_vector(self, compile_fn, place_fn):
    # Optionally compile.
    square = compile_fn(functools.partial(transforms.power, p=2.))
    sqrt = compile_fn(functools.partial(transforms.power, p=1/2.))
    # Optionally convert to device array.
    xs = place_fn(self.xs)
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(xs)), self.xs, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
