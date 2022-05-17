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
"""Unit tests for `nonlinear_bellman.py`."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from rlax._src import nonlinear_bellman


class IdentityTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.q_t = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2]]],
        dtype=np.float64)

  @parameterized.parameters(
      nonlinear_bellman.IDENTITY_PAIR,
      nonlinear_bellman.SIGNED_LOGP1_PAIR,
      nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR,
      nonlinear_bellman.HYPERBOLIC_SIN_PAIR)
  def test_identity(self, tx, inv_tx):
    """Tests that tx(inv_tx(x)) == inv_tx(tx(x)) == x."""
    np.testing.assert_allclose(inv_tx(tx(self.q_t)), self.q_t, rtol=1e-3)
    np.testing.assert_allclose(tx(inv_tx(self.q_t)), self.q_t, rtol=1e-3)

  def test_muzero_pair_is_consistent(self):
    a = np.array([1.03, 4.43, -3012.33, 0.0])
    tx = nonlinear_bellman.muzero_pair(
        num_bins=601,
        min_value=-300,
        max_value=300,
        tx=nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR)
    probs = tx.apply(a)
    scalar = tx.apply_inv(probs)
    np.testing.assert_allclose(a, scalar, rtol=1e-4, atol=1e-4)


class TransformedQLambdaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.lambda_ = 0.75

    self.q_tm1 = np.array(
        [[[1.1, 2.1], [-1.1, 1.1], [3.1, -3.1]],
         [[2.1, 3.1], [-1.1, 0.1], [-2.1, -1.1]]],
        dtype=np.float32)
    self.a_tm1 = np.array(
        [[0, 1, 0],
         [1, 0, 0]],
        dtype=np.int32)
    self.discount_t = np.array(
        [[0., 0.89, 0.85],
         [0.88, 1., 0.83]],
        dtype=np.float32)
    self.r_t = np.array(
        [[-1.3, -1.3, 2.3],
         [1.3, 5.3, -3.3]],
        dtype=np.float32)
    self.q_t = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2]]],
        dtype=np.float32)

    self.expected_td = np.array(
        [[[-2.4, 0.4280, 1.07], [0.6935, 3.478, -2.196]],
         [[-1.9329, 0.6643, -0.7854], [-0.20713, 2.1855, 0.27132]],
         [[-1.6179, 0.4633, -0.7576], [-1.1097, 1.6509, 0.3598]],
         [[-2.1785, 0.6562, -0.5938], [-0.0892, 2.6553, -0.1208]]],
        dtype=np.float32)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('identity0', nonlinear_bellman.IDENTITY_PAIR, 0),
      ('signed_logp11', nonlinear_bellman.SIGNED_LOGP1_PAIR, 1),
      ('signed_hyperbolic2', nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2),
      ('hyperbolic_sin3', nonlinear_bellman.HYPERBOLIC_SIN_PAIR, 3))
  def test_transformed_q_lambda_batch(self, tx_pair, td_index):
    """Tests correctness for full batch."""
    transformed_q_lambda = self.variant(jax.vmap(functools.partial(
        nonlinear_bellman.transformed_q_lambda, tx_pair=tx_pair,
        lambda_=self.lambda_)))
    # Compute vtrace output.
    actual_td = transformed_q_lambda(
        self.q_tm1, self.a_tm1, self.r_t, self.discount_t, self.q_t)
    # Test output.
    np.testing.assert_allclose(self.expected_td[td_index], actual_td, rtol=1e-3)


class TransformedNStepQLearningTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.n = 2

    self.q_tm1 = np.array(
        [[[1.1, 2.1], [-1.1, 1.1], [3.1, -3.1]],
         [[2.1, 3.1], [-1.1, 0.1], [-2.1, -1.1]]],
        dtype=np.float32)
    self.a_tm1 = np.array(
        [[0, 1, 0],
         [1, 0, 0]],
        dtype=np.int32)
    self.discount_t = np.array(
        [[0., 0.89, 0.85],
         [0.88, 1., 0.83]],
        dtype=np.float32)
    self.r_t = np.array(
        [[-1.3, -1.3, 2.3],
         [1.3, 5.3, -3.3]],
        dtype=np.float32)
    self.target_q_t = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2]]],
        dtype=np.float32)
    self.a_t = np.array(
        [[0, 1, 0],
         [1, 0, 0]],
        dtype=np.int32)

    self.expected_td = np.array([
        [[-2.4, 1.3112999, 1.0700002],
         [3.9199996, 2.104, -2.196]],
        [[-1.9329091, 0.9564189, -0.7853615],
         [-0.9021418, 1.1716722, 0.2713145]],
        [[-1.6178751, 0.85600746, -0.75762916],
         [-0.87689304, 0.6246443, 0.3598088]],
        [[-2.178451, 1.02313, -0.593768],
         [-0.415362, 1.790864, -0.120749]]
    ], dtype=np.float32)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('identity0', nonlinear_bellman.IDENTITY_PAIR, 0),
      ('signed_logp11', nonlinear_bellman.SIGNED_LOGP1_PAIR, 1),
      ('signed_hyperbolic2', nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2),
      ('hyperbolic_sin3', nonlinear_bellman.HYPERBOLIC_SIN_PAIR, 3))
  def test_transformed_q_lambda_batch(self, tx_pair, td_index):
    """Tests correctness for full batch."""
    transformed_n_step_q_learning = self.variant(jax.vmap(functools.partial(
        nonlinear_bellman.transformed_n_step_q_learning, tx_pair=tx_pair,
        n=self.n)))
    actual_td = transformed_n_step_q_learning(
        self.q_tm1, self.a_tm1, self.target_q_t, self.a_t, self.r_t,
        self.discount_t)
    np.testing.assert_allclose(self.expected_td[td_index], actual_td, rtol=1e-3)


class TransformedRetraceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._lambda = 0.9

    self._qs = np.array(
        [[[1.1, 2.1], [-1.1, 1.1], [3.1, -3.1], [-1.2, 0.0]],
         [[2.1, 3.1], [9.5, 0.1], [-2.1, -1.1], [0.1, 7.4]]],
        dtype=np.float32)
    self._targnet_qs = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2], [-2.25, -6.0]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2], [1.5, 1.0]]],
        dtype=np.float32)
    self._actions = np.array(
        [[0, 1, 0, 0],
         [1, 0, 0, 1]],
        dtype=np.int32)
    self._rewards = np.array(
        [[-1.3, -1.3, 2.3, 42.0],
         [1.3, 5.3, -3.3, -5.0]],
        dtype=np.float32)
    self._pcontinues = np.array(
        [[0., 0.89, 0.85, 0.99],
         [0.88, 1., 0.83, 0.95]],
        dtype=np.float32)
    self._target_policy_probs = np.array(
        [[[0.5, 0.5], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]],
         [[0.1, 0.9], [1.0, 0.0], [0.3, 0.7], [0.7, 0.3]]],
        dtype=np.float32)
    self._behavior_policy_probs = np.array(
        [[0.5, 0.1, 0.9, 0.3],
         [0.4, 0.6, 1.0, 0.9]],
        dtype=np.float32)

    self.expected_td = np.array(
        [[[-2.4, -2.7905, -3.0313], [0.7889, -6.3645, -0.0795]],
         [[-1.9329, -4.2626, -6.7738], [-2.3989, -9.9802, 1.4852]],
         [[-1.6179, -3.0165, -5.2699], [-2.7742, -9.9544, 2.3167]],
         [[-2.1785, -4.2530, -6.7081], [-1.3654, -8.2213, 0.7641]]],
        dtype=np.float32)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('identity0', nonlinear_bellman.IDENTITY_PAIR, 0),
      ('signed_logp11', nonlinear_bellman.SIGNED_LOGP1_PAIR, 1),
      ('signed_hyperbolic2', nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2),
      ('hyperbolic_sin3', nonlinear_bellman.HYPERBOLIC_SIN_PAIR, 3))
  def test_transformed_retrace_batch(self, tx_pair, td_index):
    """Tests correctness for full batch."""
    transformed_retrace = self.variant(jax.vmap(functools.partial(
        nonlinear_bellman.transformed_retrace,
        tx_pair=tx_pair, lambda_=self._lambda)))
    # Compute transformed vtrace td errors in batch.
    actual_td = transformed_retrace(
        self._qs[:, :-1], self._targnet_qs[:, 1:], self._actions[:, :-1],
        self._actions[:, 1:], self._rewards[:, :-1], self._pcontinues[:, :-1],
        self._target_policy_probs[:, 1:], self._behavior_policy_probs[:, 1:])
    # Test output.
    np.testing.assert_allclose(self.expected_td[td_index], actual_td, rtol=1e-3)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
