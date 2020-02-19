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
"""Tests for schedules.py."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import numpy as np

from rlax._src import schedules


@parameterized.named_parameters(
    ('JitOnp', jax.jit, lambda t: t),
    ('NoJitOnp', lambda fn: fn, lambda t: t),
    ('JitJnp', jax.jit, jax.device_put),
    ('NoJitJnp', lambda fn: fn, jax.device_put))
class PolynomialTest(parameterized.TestCase):

  def test_linear(self, compile_fn, place_fn):
    """Check linear schedule."""
    # Get schedule function.
    schedule_fn = schedules.polynomial_schedule(10., 20., 1, 10)
    # Optionally compile.
    schedule_fn = compile_fn(schedule_fn)
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(15):
      # Optionally convert to device array.
      step_count = place_fn(count)
      # Compute next value.
      generated_vals.append(schedule_fn(step_count))
    # Test output.
    expected_vals = np.array(list(range(10, 20)) + [20] * 5, dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  def test_nonlinear(self, compile_fn, place_fn):
    """Check non-linear (quadratic) schedule."""
    # Get schedule function.
    schedule_fn = schedules.polynomial_schedule(25., 10., 2, 10)
    # Optionally compile.
    schedule_fn = compile_fn(schedule_fn)
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(15):
      # Optionally convert to device array.
      step_count = place_fn(count)
      # Compute next value.
      generated_vals.append(schedule_fn(step_count))
    # Test output.
    expected_vals = np.array(
        [10. + 15. * (1.-n/10)**2 for n in range(10)] + [10] * 5,
        dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)

  def test_with_decay_begin(self, compile_fn, place_fn):
    """Check quadratic schedule with non-zero schedule begin."""
    # Get schedule function.
    schedule_fn = schedules.polynomial_schedule(
        30., 10., 2, 10, transition_begin=4)
    # Optionally compile.
    schedule_fn = compile_fn(schedule_fn)
    # Test that generated values equal the expected schedule values.
    generated_vals = []
    for count in range(20):
      # Optionally convert to device array.
      step_count = place_fn(count)
      # Compute next value.
      generated_vals.append(schedule_fn(step_count))
    # Test output.
    expected_vals = np.array(
        [30.] * 4 + [10. + 20. * (1.-n/10)**2 for n in range(10)] + [10] * 6,
        dtype=np.float32)
    np.testing.assert_allclose(
        expected_vals, np.array(generated_vals), atol=1e-3)


if __name__ == '__main__':
  absltest.main()
