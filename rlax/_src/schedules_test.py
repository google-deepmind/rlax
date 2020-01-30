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


class PolynomialTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_deterministic(self, compile_fn, place_fn):
    """Check that noisy and noisless actions match for zero stddev."""
    # Get schedule function.
    schedule_fn = schedules.polynomial_decay(10., 20, 1, 10)
    # Optionally compile.
    schedule_fn = compile_fn(schedule_fn)
    # Test that noisy and noisless actions match for zero stddev
    generated_vals = []
    for count in range(10):
      # Optionally convert to device array.
      step_count = place_fn(count)
      # Compute next value.
      generated_vals.append(schedule_fn(step_count))
    # Test output.
    expected_vals = np.arange(10, 20, 1, dtype=np.float32)
    np.testing.assert_allclose(
        np.stack(generated_vals), expected_vals, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
