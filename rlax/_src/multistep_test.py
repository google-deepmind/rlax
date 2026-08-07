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

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import multistep


class LambdaReturnsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
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

  @chex.all_variants()
  def test_lambda_returns_batch(self):
    """Tests for a full batch."""
    lambda_returns = self.variant(jax.vmap(functools.partial(  # pyrefly: ignore[missing-attribute]
        multistep.lambda_returns, lambda_=self.lambda_)))
    # Compute lambda return in batch.
    actual = lambda_returns(self.r_t, self.discount_t, self.v_t)
    # Test return estimate.
    np.testing.assert_allclose(self.expected, actual, rtol=1e-5)

  @staticmethod
  def _sequential_reference(r_t, discount_t, v_t, lambda_):
    lambda_ = jnp.ones_like(discount_t) * lambda_

    def _body(acc, xs):
      reward, discount, value, lambda_ = xs
      acc = reward + discount * ((1. - lambda_) * value + lambda_ * acc)
      return acc, acc

    return jax.lax.scan(
        _body, v_t[-1], (r_t, discount_t, v_t, lambda_), reverse=True)[1]

  @chex.all_variants()
  @parameterized.named_parameters(
      ('length_one_scalar_lambda', 1, False),
      ('non_power_of_two_vector_lambda', 17, True),
      ('long_vector_lambda', 4097, True),
  )
  def test_matches_sequential_reference(self, sequence_length, vector_lambda):
    keys = jax.random.split(jax.random.key(sequence_length), 4)
    r_t = jax.random.normal(keys[0], (sequence_length,))
    discount_t = 0.99 * jax.random.uniform(keys[1], (sequence_length,))
    discount_t = discount_t.at[::max(1, sequence_length // 4)].set(0.)
    v_t = jax.random.normal(keys[2], (sequence_length,))
    if vector_lambda:
      lambda_ = jax.random.uniform(keys[3], (sequence_length,))
    else:
      lambda_ = 0.75

    expected = self._sequential_reference(r_t, discount_t, v_t, lambda_)
    actual = self.variant(multistep.lambda_returns)(
        r_t, discount_t, v_t, lambda_)

    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-6)

  def test_gradients_match_sequential_reference(self):
    keys = jax.random.split(jax.random.key(7), 4)
    inputs = (
        jax.random.normal(keys[0], (17,)),
        0.99 * jax.random.uniform(keys[1], (17,)),
        jax.random.normal(keys[2], (17,)),
        jax.random.uniform(keys[3], (17,)),
    )

    def summed_returns(fn, *args):
      return jnp.sum(fn(*args))

    expected = jax.grad(
        functools.partial(summed_returns, self._sequential_reference),
        argnums=(0, 1, 2, 3))(*inputs)
    actual = jax.grad(
        functools.partial(summed_returns, multistep.lambda_returns),
        argnums=(0, 1, 2, 3))(*inputs)

    chex.assert_trees_all_close(expected, actual, rtol=1e-5, atol=1e-6)

  def test_stop_target_gradients(self):
    r_t = jnp.array([1., 2., 3.])
    discount_t = jnp.array([0.9, 0.9, 0.9])
    v_t = jnp.array([0.5, 1., 1.5])
    lambda_ = jnp.array([0.7, 0.8, 0.9])

    gradients = jax.grad(
        lambda *args: jnp.sum(multistep.lambda_returns(
            *args, stop_target_gradients=True)),
        argnums=(0, 1, 2, 3))(r_t, discount_t, v_t, lambda_)

    chex.assert_trees_all_close(
        gradients, jax.tree.map(jnp.zeros_like, gradients))


class DiscountedReturnsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

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

  @chex.all_variants()
  def test_discounted_returns_batch(self):
    """Tests for a single element."""
    discounted_returns = self.variant(jax.vmap(multistep.discounted_returns))  # pyrefly: ignore[missing-attribute]
    # Compute discounted return.
    actual_scalar = discounted_returns(self.r_t, self.discount_t,
                                       self.bootstrap_v)
    actual_vector = discounted_returns(self.r_t, self.discount_t, self.v_t)
    # Test output.
    np.testing.assert_allclose(self.expected, actual_scalar, rtol=1e-5)
    np.testing.assert_allclose(self.expected, actual_vector, rtol=1e-5)


class NStepBootstrappedReturnsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.r_t = np.array(
        [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
    self.discount_t = np.array(
        [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
    self.v_t = np.array(
        [[3.0, 1.0, 5.0, -5.0, 3.0], [-1.7, 1.2, 2.3, 2.2, 2.7]])

    # Different expected results for different values of n.
    self.expected = {}
    self.expected[3] = np.array(
        [[2.8, -3.15, 0.7, 1.7, 3.4], [1.2155, 0.714, 0.2576, 3.192, 3.99]],
        dtype=np.float32)
    self.expected[5] = np.array(
        [[1.315, 0.63, 0.7, 1.7, 3.4], [1.33592, 0.9288, 0.2576, 3.192, 3.99]],
        dtype=np.float32)
    self.expected[7] = np.array(
        [[1.315, 0.63, 0.7, 1.7, 3.4], [1.33592, 0.9288, 0.2576, 3.192, 3.99]],
        dtype=np.float32)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('smaller_n', 3,), ('equal_n', 5,), ('bigger_n', 7,))
  def test_n_step_sequence_returns_batch(self, n):
    """Tests for a full batch."""
    n_step_returns = self.variant(jax.vmap(functools.partial(  # pyrefly: ignore[missing-attribute]
        multistep.n_step_bootstrapped_returns, n=n)))
    # Compute n-step return in batch.
    actual = n_step_returns(self.r_t, self.discount_t, self.v_t)
    # Test return estimate.
    np.testing.assert_allclose(self.expected[n], actual, rtol=1e-5)

  def test_reduces_to_lambda_returns(self):
    """Test function is the same as lambda_returns when n is sequence length."""
    lambda_t = 0.75
    n = len(self.r_t[0])
    expected = multistep.lambda_returns(self.r_t[0], self.discount_t[0],
                                        self.v_t[0], lambda_t)
    actual = multistep.n_step_bootstrapped_returns(self.r_t[0],
                                                   self.discount_t[0],
                                                   self.v_t[0], n, lambda_t)
    np.testing.assert_allclose(expected, actual, rtol=1e-5)


class TDErrorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.r_t = np.array(
        [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
    self.discount_t = np.array(
        [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
    self.rho_tm1 = np.array(
        [[0.5, 0.9, 1.3, 0.2, 0.8], [2., 0.1, 1., 0.4, 1.7]])
    self.values = np.array(
        [[3.0, 1.0, 5.0, -5.0, 3.0, 1.], [-1.7, 1.2, 2.3, 2.2, 2.7, 2.]])

  @chex.all_variants()
  def test_importance_corrected_td_errors_batch(self):
    """Tests equivalence to computing the error from a the lambda-return."""
    # Vmap and optionally compile.
    lambda_returns = self.variant(jax.vmap(multistep.lambda_returns))  # pyrefly: ignore[missing-attribute]
    td_errors = self.variant(jax.vmap(multistep.importance_corrected_td_errors))  # pyrefly: ignore[missing-attribute]
    # Compute multistep td-error with recursion on deltas.
    td_direct = td_errors(self.r_t, self.discount_t, self.rho_tm1,
                          np.ones_like(self.discount_t), self.values)
    # Compute off-policy corrected return, and derive td-error from it.
    ls_ = np.concatenate((self.rho_tm1[:, 1:], [[1.], [1.]]), axis=1)
    td_from_returns = self.rho_tm1 * (
        lambda_returns(self.r_t, self.discount_t, self.values[:, 1:], ls_) -
        self.values[:, :-1])
    # Check equivalence.
    np.testing.assert_allclose(td_direct, td_from_returns, rtol=1e-5)


class TruncatedGeneralizedAdvantageEstimationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.r_t = jnp.array([[0., 0., 1., 0., -0.5],
                          [0., 0., 0., 0., 1.]])
    self.v_t = jnp.array([[1., 4., -3., -2., -1., -1.],
                          [-3., -2., -1., 0.0, 5., -1.]])
    self.discount_t = jnp.array([[0.99, 0.99, 0.99, 0.99, 0.99],
                                 [0.9, 0.9, 0.9, 0.0, 0.9]])
    self.dummy_rho_tm1 = jnp.array([[1., 1., 1., 1., 1],
                                    [1., 1., 1., 1., 1.]])
    self.array_lambda = jnp.array([[0.9, 0.9, 0.9, 0.9, 0.9],
                                   [0.9, 0.9, 0.9, 0.9, 0.9]])

    # Different expected results for different values of lambda.
    self.expected = {}
    self.expected[1.] = np.array(
        [[-1.45118, -4.4557, 2.5396, 0.5249, -0.49],
         [3., 2., 1., 0., -4.9]],
        dtype=np.float32)
    self.expected[0.7] = np.array(
        [[-0.676979, -5.248167, 2.4846, 0.6704, -0.49],
         [2.2899, 1.73, 1., 0., -4.9]],
        dtype=np.float32)
    self.expected[0.4] = np.array(
        [[0.56731, -6.042, 2.3431, 0.815, -0.49],
         [1.725, 1.46, 1., 0., -4.9]],
        dtype=np.float32)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('lambda1', 1.0),
      ('lambda0.7', 0.7),
      ('lambda0.4', 0.4))
  def test_truncated_gae(self, lambda_):
    """Tests truncated GAE for a full batch."""
    batched_advantage_fn_variant = self.variant(jax.vmap(  # pyrefly: ignore[missing-attribute]
        multistep.truncated_generalized_advantage_estimation,
        in_axes=(0, 0, None, 0), out_axes=0))
    actual = batched_advantage_fn_variant(
        self.r_t, self.discount_t, lambda_, self.v_t)
    np.testing.assert_allclose(self.expected[lambda_], actual, atol=1e-3)

  @chex.all_variants()
  def test_array_lambda(self):
    """Tests that truncated GAE is consistent with scalar or array lambda_."""
    scalar_lambda_fn = self.variant(jax.vmap(  # pyrefly: ignore[missing-attribute]
        multistep.truncated_generalized_advantage_estimation,
        in_axes=(0, 0, None, 0), out_axes=0))
    array_lambda_fn = self.variant(jax.vmap(  # pyrefly: ignore[missing-attribute]
        multistep.truncated_generalized_advantage_estimation))
    scalar_lambda_result = scalar_lambda_fn(
        self.r_t, self.discount_t, 0.9, self.v_t)
    array_lambda_result = array_lambda_fn(
        self.r_t, self.discount_t, self.array_lambda, self.v_t)
    np.testing.assert_allclose(scalar_lambda_result, array_lambda_result,
                               atol=1e-3)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('lambda_1', 1.0),
      ('lambda_0.7', 0.7),
      ('lambda_0.4', 0.4))
  def test_gae_as_special_case_of_importance_corrected_td_errors(self, lambda_):
    """Tests truncated GAE.

    Tests that truncated GAE yields same output as importance corrected
    td errors with dummy ratios.

    Args:
      lambda_: a lambda to use in GAE.
    """
    batched_gae_fn_variant = self.variant(jax.vmap(  # pyrefly: ignore[missing-attribute]
        multistep.truncated_generalized_advantage_estimation,
        in_axes=(0, 0, None, 0), out_axes=0))
    gae_result = batched_gae_fn_variant(
        self.r_t, self.discount_t, lambda_, self.v_t)

    batched_ictd_errors_fn_variant = self.variant(jax.vmap(  # pyrefly: ignore[missing-attribute]
        multistep.importance_corrected_td_errors))
    ictd_errors_result = batched_ictd_errors_fn_variant(
        self.r_t,
        self.discount_t,
        self.dummy_rho_tm1,
        jnp.ones_like(self.discount_t) * lambda_,
        self.v_t)
    np.testing.assert_allclose(gae_result, ictd_errors_result, atol=1e-3)

if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
