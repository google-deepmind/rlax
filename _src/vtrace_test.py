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
"""Tests for `value_learning.py`."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.tree_util import tree_multimap
import numpy as np
from rlax._src import distributions
from rlax._src import vtrace


class VTraceTest(parameterized.TestCase):

  def setUp(self):
    super(VTraceTest, self).setUp()

    behavior_policy_logits = np.array(
        [[[8.9, 0.7], [5.0, 1.0], [0.6, 0.1], [-0.9, -0.1]],
         [[0.3, -5.0], [1.0, -8.0], [0.3, 1.7], [4.7, 3.3]]],
        dtype=np.float32)
    target_policy_logits = np.array(
        [[[0.4, 0.5], [9.2, 8.8], [0.7, 4.4], [7.9, 1.4]],
         [[1.0, 0.9], [1.0, -1.0], [-4.3, 8.7], [0.8, 0.3]]],
        dtype=np.float32)
    actions = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.int32)
    self._rho_t = distributions.categorical_importance_sampling_ratios(
        target_policy_logits, behavior_policy_logits, actions)
    self._rewards = np.array([[-1.3, -1.3, 2.3, 42.0], [1.3, 5.3, -3.3, -5.0]],
                             dtype=np.float32)
    self._discounts = np.array([[0., 0.89, 0.85, 0.99], [0.88, 1., 0.83, 0.95]],
                               dtype=np.float32)
    self._values = np.array([[2.1, 1.1, -3.1, 0.0], [3.1, 0.1, -1.1, 7.4]],
                            dtype=np.float32)
    self._bootstrap_value = np.array([8.4, -1.2], dtype=np.float32)
    self._inputs = [
        self._rewards, self._discounts, self._rho_t,
        self._values, self._bootstrap_value]

    self._clip_rho_threshold = 1.0
    self._clip_pg_rho_threshold = 5.0
    self._lambda = 1.0

    self._expected_td = np.array(
        [[-1.6155143, -3.4973226, 1.8670533, 5.0316002e1],
         [1.4662437, 3.6116405, -8.3327293e-5, -1.3540000e1]],
        dtype=np.float32)
    self._expected_pg = np.array(
        [[-1.6155143, -3.4973226, 1.8670534, 5.0316002e1],
         [1.4662433, 3.6116405, -8.3369283e-05, -1.3540000e+1]],
        dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_vtrace_td_error_and_advantage(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Optionally compile.
    vtrace_td_error_and_advantage = functools.partial(
        vtrace.vtrace_td_error_and_advantage,
        clip_rho_threshold=self._clip_rho_threshold, lambda_=self._lambda)
    vtrace_td_error_and_advantage = vtrace.vtrace_td_error_and_advantage
    vtrace_td_error_and_advantage = compile_fn(vtrace_td_error_and_advantage)
    # Check each element in the batch.
    for idx, inputs in enumerate(zip(*self._inputs)):
      # Get batch slice.
      r_t, discount_t, rho_t, v_tm1, bootstrap_value = inputs
      v_t = np.concatenate([v_tm1[1:], bootstrap_value[None]], axis=0)
      # Optionally make inputs into device arrays.
      (v_tm1, v_t, r_t, discount_t, rho_t) = tree_multimap(
          place_fn, (v_tm1, v_t, r_t, discount_t, rho_t))
      # Compute vtrace output.
      vtrace_output = vtrace_td_error_and_advantage(
          v_tm1, v_t, r_t, discount_t, rho_t)
      # Test output.
      np.testing.assert_allclose(
          self._expected_td[idx], vtrace_output.errors, rtol=1e-3)
      np.testing.assert_allclose(
          self._expected_pg[idx], vtrace_output.pg_advantage, rtol=1e-3)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_vtrace_td_error_and_advantage_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap function and optionally compile.
    vtrace_td_error_and_advantage = functools.partial(
        vtrace.vtrace_td_error_and_advantage,
        clip_rho_threshold=self._clip_rho_threshold, lambda_=self._lambda)
    vtrace_td_error_and_advantage = jax.vmap(vtrace_td_error_and_advantage)
    vtrace_td_error_and_advantage = compile_fn(vtrace_td_error_and_advantage)
    # Get function arguments.
    r_t, discount_t, rho_t, v_tm1, bootstrap_value = self._inputs
    v_t = np.concatenate([v_tm1[:, 1:], bootstrap_value[:, None]], axis=1)
    # Optionally make inputs into device arrays.
    (v_tm1, v_t, r_t, discount_t, rho_t) = tree_multimap(
        place_fn, (v_tm1, v_t, r_t, discount_t, rho_t))
    # Compute vtrace output.
    vtrace_output = vtrace_td_error_and_advantage(
        v_tm1, v_t, r_t, discount_t, rho_t)
    # Test output.
    np.testing.assert_allclose(
        self._expected_td, vtrace_output.errors, rtol=1e-3)
    np.testing.assert_allclose(
        self._expected_pg, vtrace_output.pg_advantage, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
