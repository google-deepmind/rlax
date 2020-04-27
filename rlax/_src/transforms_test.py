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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import test_util
from rlax._src import transforms


class TransformsTest(parameterized.TestCase):

  def setUp(self):
    super(TransformsTest, self).setUp()
    self.x = 0.5
    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])

  @test_util.parameterize_variant()
  def test_identity_scalar(self, variant):
    identity = variant(transforms.identity)
    x = jnp.array(self.x)
    # Test output.
    np.testing.assert_allclose(identity(x), self.x)

  @test_util.parameterize_variant()
  def test_identity_vector(self, variant):
    identity = variant(transforms.identity)
    # Test output.
    np.testing.assert_allclose(identity(self.xs), self.xs)

  @test_util.parameterize_variant()
  def test_sigmoid_scalar(self, variant):
    sigmoid = variant(transforms.sigmoid)
    logit = variant(transforms.logit)
    x = jnp.array(self.x)
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(x)), self.x, atol=1e-3)

  @test_util.parameterize_variant()
  def test_sigmoid_vector(self, variant):
    sigmoid = variant(transforms.sigmoid)
    logit = variant(transforms.logit)
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(self.xs)), self.xs, atol=1e-3)

  @test_util.parameterize_variant()
  def test_signed_log_exp_transform_scalar(self, variant):
    signed_logp1 = variant(transforms.signed_logp1)
    signed_expm1 = variant(transforms.signed_expm1)
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(signed_expm1(signed_logp1(x)), self.x, atol=1e-3)

  @test_util.parameterize_variant()
  def test_signed_log_exp_transform_vector(self, variant):
    signed_logp1 = variant(transforms.signed_logp1)
    signed_expm1 = variant(transforms.signed_expm1)
    # Test inverse.
    np.testing.assert_allclose(
        signed_expm1(signed_logp1(self.xs)), self.xs, atol=1e-3)

  @test_util.parameterize_variant()
  def test_signed_hyper_parabolic_transform_scalar(self, variant):
    signed_hyperbolic = variant(transforms.signed_hyperbolic)
    signed_parabolic = variant(transforms.signed_parabolic)
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(x)), self.x, atol=1e-3)

  @test_util.parameterize_variant()
  def test_signed_hyper_parabolic_transform_vector(self, variant):
    signed_hyperbolic = variant(transforms.signed_hyperbolic)
    signed_parabolic = variant(transforms.signed_parabolic)
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(self.xs)), self.xs, atol=1e-3)

  @test_util.parameterize_variant()
  def test_signed_power_transform_scalar(self, variant):
    square = variant(transforms.power, p=2.)
    sqrt = variant(transforms.power, p=1/2.)
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(x)), self.x, atol=1e-3)

  @test_util.parameterize_variant()
  def test_signed_power_transform_vector(self, variant):
    square = variant(transforms.power, p=2.)
    sqrt = variant(transforms.power, p=1/2.)
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(self.xs)), self.xs, atol=1e-3)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
