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
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import transforms


TWO_HOT_BINS = 5
TWO_HOT_SCALARS = [-5.0, -3.0, -1.0, -0.4, 0.0, 0.3, 1.0, 4.5, 10.0]
TWO_HOT_PROBABILITIES = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.8, 0.2, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.6, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
]


class TransformsTest(parameterized.TestCase):

  def setUp(self):
    super(TransformsTest, self).setUp()
    self.x = 0.5
    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])

  @chex.all_variants()
  def test_identity_scalar(self):
    identity = self.variant(transforms.identity)
    x = jnp.array(self.x)
    # Test output.
    np.testing.assert_allclose(identity(x), self.x)

  @chex.all_variants()
  def test_identity_vector(self):
    identity = self.variant(transforms.identity)
    # Test output.
    np.testing.assert_allclose(identity(self.xs), self.xs)

  @chex.all_variants()
  def test_sigmoid_scalar(self):
    sigmoid = self.variant(transforms.sigmoid)
    logit = self.variant(transforms.logit)
    x = jnp.array(self.x)
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_sigmoid_vector(self):
    sigmoid = self.variant(transforms.sigmoid)
    logit = self.variant(transforms.logit)
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(self.xs)), self.xs, atol=1e-3)

  @chex.all_variants()
  def test_signed_log_exp_transform_scalar(self):
    signed_logp1 = self.variant(transforms.signed_logp1)
    signed_expm1 = self.variant(transforms.signed_expm1)
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(signed_expm1(signed_logp1(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_signed_log_exp_transform_vector(self):
    signed_logp1 = self.variant(transforms.signed_logp1)
    signed_expm1 = self.variant(transforms.signed_expm1)
    # Test inverse.
    np.testing.assert_allclose(
        signed_expm1(signed_logp1(self.xs)), self.xs, atol=1e-3)

  @chex.all_variants()
  def test_signed_hyper_parabolic_transform_scalar(self):
    signed_hyperbolic = self.variant(transforms.signed_hyperbolic)
    signed_parabolic = self.variant(transforms.signed_parabolic)
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_signed_hyper_parabolic_transform_vector(self):
    signed_hyperbolic = self.variant(transforms.signed_hyperbolic)
    signed_parabolic = self.variant(transforms.signed_parabolic)
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(self.xs)), self.xs, atol=1e-3)

  @chex.all_variants()
  def test_signed_power_transform_scalar(self):
    square = self.variant(functools.partial(transforms.power, p=2.))
    sqrt = self.variant(functools.partial(transforms.power, p=1/2.))
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_signed_power_transform_vector(self):
    square = self.variant(functools.partial(transforms.power, p=2.))
    sqrt = self.variant(functools.partial(transforms.power, p=1/2.))
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(self.xs)), self.xs, atol=1e-3)

  def test_transform_to_2hot(self):
    y = transforms.transform_to_2hot(
        scalar=jnp.array(TWO_HOT_SCALARS),
        min_value=-1.0,
        max_value=1.0,
        num_bins=TWO_HOT_BINS)

    np.testing.assert_allclose(y, np.array(TWO_HOT_PROBABILITIES), atol=1e-4)

  def test_transform_from_2hot(self):
    y = transforms.transform_from_2hot(
        probs=jnp.array(TWO_HOT_PROBABILITIES),
        min_value=-1.0,
        max_value=1.0,
        num_bins=TWO_HOT_BINS)

    np.testing.assert_allclose(
        y, np.clip(np.array(TWO_HOT_SCALARS), -1, 1), atol=1e-4)

  def test_2hot_roundtrip(self):
    min_value = -1.0
    max_value = 1.0
    num_bins = 11

    value = np.arange(min_value, max_value, 0.01)

    transformed = transforms.transform_to_2hot(
        value, min_value, max_value, num_bins)
    restored = transforms.transform_from_2hot(
        transformed, min_value, max_value, num_bins)

    np.testing.assert_almost_equal(value, restored, decimal=5)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
