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
"""Unit tests for `multistep.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.tree_util import tree_map
import numpy as np
from rlax._src import multistep


class LambdaReturnsTest(parameterized.TestCase):

  def setUp(self):
    super(LambdaReturnsTest, self).setUp()
    self.lambda_ = 0.75

    self.r_t = np.array(
        [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
    self.discount_t = np.array(
        [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
    self.v_t = np.array(
        [[3.0, 1.0, 5.0, -5.0, 3.0], [-1.7, 1.2, 2.3, 2.2, 2.7]])

    self.expected = np.array(
        [[1.6460547, 0.72281253, 0.7375001, 0.6500001, 3.4],
         [0.7866317, 0.9913063, 0.1101501, 2.834, 3.99]],
        dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_lambda_returns(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Optionally compile.
    lambda_returns = compile_fn(multistep.lambda_returns)
    # For each element in the batch.
    for r_t, discount_t, v_t, expected in zip(
        self.r_t, self.discount_t, self.v_t, self.expected):
      # Optionally convert to device array.
      r_t, discount_t, v_t = tree_map(place_fn, (r_t, discount_t, v_t))
      # Test outputs.
      actual = lambda_returns(r_t, discount_t, v_t, self.lambda_)
      np.testing.assert_allclose(expected, actual, rtol=1e-5)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_lambda_returns_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    lambda_returns = compile_fn(jax.vmap(
        multistep.lambda_returns, in_axes=(0, 0, 0, None)))
    # Optionally convert to device array.
    r_t, discount_t, v_t = tree_map(
        place_fn, (self.r_t, self.discount_t, self.v_t))
    # Compute lambda return in batch.
    actual = lambda_returns(r_t, discount_t, v_t, self.lambda_)
    # Test return estimate.
    np.testing.assert_allclose(self.expected, actual, rtol=1e-5)


class DiscountedReturnsTest(parameterized.TestCase):

  def setUp(self):
    super(DiscountedReturnsTest, self).setUp()

    self.r_t = np.array(
        [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
    self.discount_t = np.array(
        [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
    self.v_t = np.array(
        [[3.0, 1.0, 5.0, -5.0, 3.0], [-1.7, 1.2, 2.3, 2.2, 2.7]])
    self.bootstrap_v = np.array([v[-1] for v in self.v_t])

    self.expected = np.array(
        [[1.315, 0.63000005, 0.70000005, 1.7, 3.4],
         [1.33592, 0.9288, 0.2576, 3.192, 3.9899998]],
        dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_discounted_returns(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Optionally compile.
    discounted_returns = compile_fn(multistep.discounted_returns)
    # For each element in the batch.
    for r_t, discount_t, v_t, bootstrap_v, expected in zip(
        self.r_t, self.discount_t, self.v_t, self.bootstrap_v, self.expected):
      # Optionally convert to device array.
      r_t, discount_t, bootstrap_v, v_t = tree_map(
          place_fn, (r_t, discount_t, bootstrap_v, v_t))
      # Compute discounted return.
      actual_scalar = discounted_returns(r_t, discount_t, bootstrap_v)
      actual_vector = discounted_returns(r_t, discount_t, v_t)
      # Test output.
      np.testing.assert_allclose(expected, actual_scalar, rtol=1e-5)
      np.testing.assert_allclose(expected, actual_vector, rtol=1e-5)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_discounted_returns_batch(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Vmap and optionally compile.
    discounted_returns = compile_fn(jax.vmap(multistep.discounted_returns))
    # Optionally convert to device array.
    r_t, discount_t, bootstrap_v, v_t = tree_map(
        place_fn, (self.r_t, self.discount_t, self.bootstrap_v, self.v_t))
    # Compute discounted return.
    actual_scalar = discounted_returns(r_t, discount_t, bootstrap_v)
    actual_vector = discounted_returns(r_t, discount_t, v_t)
    # Test output.
    np.testing.assert_allclose(self.expected, actual_scalar, rtol=1e-5)
    np.testing.assert_allclose(self.expected, actual_vector, rtol=1e-5)


class TDErrorTest(parameterized.TestCase):

  def setUp(self):
    super(TDErrorTest, self).setUp()

    self.r_t = np.array(
        [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
    self.discount_t = np.array(
        [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
    self.rho_tm1 = np.array(
        [[0.5, 0.9, 1.3, 0.2, 0.8], [2., 0.1, 1., 0.4, 1.7]])
    self.values = np.array(
        [[3.0, 1.0, 5.0, -5.0, 3.0, 1.], [-1.7, 1.2, 2.3, 2.2, 2.7, 2.]])

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_importance_corrected_td_errors(self, compile_fn, place_fn):
    """Tests equivalence to computing the error from a the lambda-return."""
    # Optionally compile.
    lambda_returns = compile_fn(multistep.lambda_returns)
    importance_corrected_td_errors = compile_fn(
        multistep.importance_corrected_td_errors)
    # For each element in the batch.
    for r_t, discount_t, rho_tm1, values in zip(
        self.r_t, self.discount_t, self.rho_tm1, self.values):
      # Optionally convert to device array.
      r_t, discount_t, rho_tm1, values = tree_map(
          place_fn, (r_t, discount_t, rho_tm1, values))
      # Compute multistep td-error with recursion on deltas.
      td_direct = importance_corrected_td_errors(
          r_t, discount_t, rho_tm1, np.ones_like(discount_t), values)
      # Compute off-policy corrected return, and derive td-error from it.
      lambdas = np.concatenate((rho_tm1[1:], [1.]))
      td_from_returns = rho_tm1 * (
          lambda_returns(r_t, discount_t, values[1:], lambdas) - values[:-1])
      # Check equivalence.
      np.testing.assert_allclose(td_direct, td_from_returns, rtol=1e-5)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_importance_corrected_td_errors_batch(self, compile_fn, place_fn):
    """Tests equivalence to computing the error from a the lambda-return."""
    # Vmap and optionally compile.
    lambda_returns = compile_fn(jax.vmap(multistep.lambda_returns))
    td_errors = compile_fn(jax.vmap(multistep.importance_corrected_td_errors))
    # Optionally convert to device array.
    r_t, discount_t, rho_tm1, values = tree_map(
        place_fn, (self.r_t, self.discount_t, self.rho_tm1, self.values))
    # Compute multistep td-error with recursion on deltas.
    td_direct = td_errors(
        r_t, discount_t, rho_tm1, np.ones_like(self.discount_t), values)
    # Compute off-policy corrected return, and derive td-error from it.
    ls_ = np.concatenate((self.rho_tm1[:, 1:], [[1.], [1.]]), axis=1)
    td_from_returns = self.rho_tm1 * (
        lambda_returns(r_t, discount_t, values[:, 1:], ls_) - values[:, :-1])
    # Check equivalence.
    np.testing.assert_allclose(td_direct, td_from_returns, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
