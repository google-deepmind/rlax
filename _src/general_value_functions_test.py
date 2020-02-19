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
"""Unit tests for `general_value_functions.py`."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from rlax._src import general_value_functions


class PixelControlTest(parameterized.TestCase):

  def setUp(self):
    super(PixelControlTest, self).setUp()
    self.cell_size = 2
    time, self.batch_size = 3, 2
    height, width, channels = 4, 4, 3

    shape = (time, self.batch_size, height, width, channels)
    hw = np.matmul(
        np.arange(0, 1, 0.25)[:, None],
        np.arange(0, 1, 0.25)[None, :])
    hwc = np.stack([hw, hw + 0.1, hw + 0.2], axis=-1)
    bhwc = np.stack([hwc, hwc + 0.1], axis=0)
    tbhwc = np.stack([bhwc, bhwc + 0.05, bhwc + 0.1], axis=0)
    assert tbhwc.shape == shape
    self.obs = tbhwc

    self.expected = 0.05 * np.ones((2, 2, 2, 2), dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_pixel_control_rewards(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Optionally compile.
    pixel_control_rewards = functools.partial(
        general_value_functions.pixel_control_rewards, cell_size=self.cell_size)
    pixel_control_rewards = compile_fn(pixel_control_rewards)
    # Optionally convert into device arrays.
    obs = place_fn(self.obs)
    # Test pseudo rewards.
    for i in range(self.batch_size):
      rs = pixel_control_rewards(obs[:, i])
      np.testing.assert_allclose(self.expected[:, i], rs, rtol=1e-5)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_pixel_control_rewards_batch(self, compile_fn, place_fn):
    """Tests for a single element, for non-default temperature."""
    # Vmap and optionally compile.
    pixel_control_rewards = functools.partial(
        general_value_functions.pixel_control_rewards, cell_size=self.cell_size)
    pixel_control_rewards = compile_fn(jax.vmap(
        pixel_control_rewards, in_axes=(1,), out_axes=1))
    # Optionally convert into device arrays.
    obs = place_fn(self.obs)
    # Test pseudo rewards.
    rs = pixel_control_rewards(obs)
    np.testing.assert_allclose(self.expected, rs, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
