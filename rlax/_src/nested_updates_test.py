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
"""Tests for perturbations.py."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import nested_updates


class NestedUpdatesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    old = jnp.zeros((3,), dtype=jnp.float32)
    new = jnp.ones((3,), dtype=jnp.float32)
    self._old_struct = ((old, old), old)
    self._new_struct = ((new, new), new)

  @chex.all_variants()
  def test_conditional_update_is_time(self):
    """Check periodic update enabled."""
    conditional_update = self.variant(nested_updates.conditional_update)

    is_time = jnp.array(True)
    output = conditional_update(self._new_struct, self._old_struct, is_time)
    for o, exp in zip(
        jax.tree_leaves(output), jax.tree_leaves(self._new_struct)):
      np.testing.assert_allclose(o, exp)

  @chex.all_variants()
  def test_conditional_update_is_not_time(self):
    """Check periodic update disables."""
    conditional_update = self.variant(nested_updates.conditional_update)

    is_not_time = jnp.array(False)
    output = conditional_update(self._new_struct, self._old_struct, is_not_time)
    for o, exp in zip(
        jax.tree_leaves(output), jax.tree_leaves(self._old_struct)):
      np.testing.assert_allclose(o, exp)

  @chex.all_variants()
  def test_incremental_update(self):
    """Check nested incremental updates."""
    incremental_update = self.variant(nested_updates.incremental_update)

    tau = jnp.array(0.1)
    output = incremental_update(self._new_struct, self._old_struct, tau)
    for o, exp in zip(
        jax.tree_leaves(output), jax.tree_leaves(self._new_struct)):
      np.testing.assert_allclose(o, tau * exp)


if __name__ == '__main__':
  absltest.main()
