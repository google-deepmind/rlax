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
"""Unit tests for `nonlinear_bellman.py`."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from rlax._src import nonlinear_bellman


class IdentityTest(parameterized.TestCase):

  def setUp(self):
    super(IdentityTest, self).setUp()
    self.q_t = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2]]],
        dtype=np.float64)

  @parameterized.parameters(
      nonlinear_bellman.IDENTITY_PAIR,
      nonlinear_bellman.SIGNED_LOGP1_PAIR,
      nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR)
  def test_identity(self, tx, inv_tx):
    """Tests that tx(inv_tx(x)) == inv_tx(tx(x)) == x."""
    np.testing.assert_allclose(inv_tx(tx(self.q_t)), self.q_t, rtol=1e-3)
    np.testing.assert_allclose(tx(inv_tx(self.q_t)), self.q_t, rtol=1e-3)


class TransformedQLambdaTest(parameterized.TestCase):

  def setUp(self):
    super(TransformedQLambdaTest, self).setUp()
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
         [[-1.6179, 0.4633, -0.7576], [-1.1097, 1.6509, 0.3598]]],
        dtype=np.float32)

  @parameterized.parameters(
      (nonlinear_bellman.IDENTITY_PAIR, 0, lambda fn: fn),
      (nonlinear_bellman.IDENTITY_PAIR, 0, jax.jit),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, lambda fn: fn),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, jax.jit),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, lambda fn: fn),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, jax.jit))
  def test_transformed_q_lambda(self, tx_pair, td_index, compile_fn):
    """Tests correctness for single element."""
    # Optionally compile.
    transformed_q_lambda = functools.partial(
        nonlinear_bellman.transformed_q_lambda, tx_pair=tx_pair)
    transformed_q_lambda = compile_fn(transformed_q_lambda)
    # For each element in the batch.
    for expected_td, q_tm1, a_tm1, r_t, discount_t, q_t in zip(
        self.expected_td[td_index], self.q_tm1, self.a_tm1, self.r_t,
        self.discount_t, self.q_t):
      # Compute transformed Q-lambda td-errors.
      actual_td = transformed_q_lambda(
          q_tm1, a_tm1, r_t, discount_t, q_t, self.lambda_)
      # Test output.
      np.testing.assert_allclose(actual_td, expected_td, rtol=1e-3)

  @parameterized.parameters(
      (nonlinear_bellman.IDENTITY_PAIR, 0, lambda fn: fn),
      (nonlinear_bellman.IDENTITY_PAIR, 0, jax.jit),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, lambda fn: fn),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, jax.jit),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, lambda fn: fn),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, jax.jit))
  def test_transformed_q_lambda_batch(self, tx_pair, td_index, compile_fn):
    """Tests correctness for full batch."""
    # Vmap function and optionally compile.
    transformed_q_lambda = functools.partial(
        nonlinear_bellman.transformed_q_lambda, tx_pair=tx_pair)
    transformed_q_lambda = jax.vmap(
        transformed_q_lambda, in_axes=(0, 0, 0, 0, 0, None))
    transformed_q_lambda = compile_fn(transformed_q_lambda)
    # Compute vtrace output.
    actual_td = transformed_q_lambda(
        self.q_tm1, self.a_tm1, self.r_t, self.discount_t, self.q_t,
        self.lambda_)
    # Test output.
    np.testing.assert_allclose(self.expected_td[td_index], actual_td, rtol=1e-3)


class TransformedRetraceTest(parameterized.TestCase):

  def setUp(self):
    super(TransformedRetraceTest, self).setUp()
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
         [[-1.6179, -3.0165, -5.2699], [-2.7742, -9.9544, 2.3167]]],
        dtype=np.float32)

  @parameterized.parameters(
      (nonlinear_bellman.IDENTITY_PAIR, 0, lambda fn: fn),
      (nonlinear_bellman.IDENTITY_PAIR, 0, jax.jit),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, lambda fn: fn),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, jax.jit),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, lambda fn: fn),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, jax.jit))
  def test_transformed_retrace(self, tx_pair, td_index, compile_fn):
    """Tests correctness for single element."""
    # Optionally compile.
    transformed_retrace = functools.partial(
        nonlinear_bellman.transformed_retrace,
        tx_pair=tx_pair, lambda_=self._lambda)
    transformed_retrace = compile_fn(transformed_retrace)
    # For each element in the batch.
    for (expected_td, qs, targnet_qs, actions, rewards, pcontinues,
         target_policy_probs, behavior_policy_probs) in zip(
             self.expected_td[td_index], self._qs, self._targnet_qs,
             self._actions, self._rewards, self._pcontinues,
             self._target_policy_probs, self._behavior_policy_probs):
      # Compute transformed retrace td errors.
      actual_td = transformed_retrace(
          q_tm1=qs[:-1], q_t=targnet_qs[1:], a_tm1=actions[:-1],
          a_t=actions[1:], r_t=rewards[:-1], discount_t=pcontinues[:-1],
          pi_t=target_policy_probs[1:], mu_t=behavior_policy_probs[1:])
      # Test output.
      np.testing.assert_allclose(expected_td, actual_td, rtol=1e-3)

  @parameterized.parameters(
      (nonlinear_bellman.IDENTITY_PAIR, 0, lambda fn: fn),
      (nonlinear_bellman.IDENTITY_PAIR, 0, jax.jit),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, lambda fn: fn),
      (nonlinear_bellman.SIGNED_LOGP1_PAIR, 1, jax.jit),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, lambda fn: fn),
      (nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR, 2, jax.jit))
  def test_transformed_retrace_batch(self, tx_pair, td_index, compile_fn):
    """Tests correctness for full batch."""
    # Vmap function and optionally compile.
    transformed_retrace = functools.partial(
        nonlinear_bellman.transformed_retrace,
        tx_pair=tx_pair, lambda_=self._lambda)
    transformed_retrace = jax.vmap(transformed_retrace)
    transformed_retrace = compile_fn(transformed_retrace)
    # Compute transformed vtrace td errors in batch.
    actual_td = transformed_retrace(
        self._qs[:, :-1], self._targnet_qs[:, 1:], self._actions[:, :-1],
        self._actions[:, 1:], self._rewards[:, :-1], self._pcontinues[:, :-1],
        self._target_policy_probs[:, 1:], self._behavior_policy_probs[:, 1:])
    # Test output.
    np.testing.assert_allclose(self.expected_td[td_index], actual_td, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
