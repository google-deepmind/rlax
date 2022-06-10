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
"""Tests for losses.py."""

from absl.testing import absltest
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import interruptions


class InterruptionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.correct_trajectory = jnp.array([
        [dm_env.StepType.MID],
        [dm_env.StepType.MID],
        [dm_env.StepType.MID],
        [dm_env.StepType.LAST],
        [dm_env.StepType.FIRST],
        [dm_env.StepType.MID],
    ])
    self.broken_trajectory = jnp.array([
        [dm_env.StepType.MID],
        [dm_env.StepType.MID],
        [dm_env.StepType.MID],
        [dm_env.StepType.MID],
        [dm_env.StepType.FIRST],
        [dm_env.StepType.MID],
    ])

  def test_fix_step_type_on_interruptions(self):
    output1 = interruptions.fix_step_type_on_interruptions(
        self.correct_trajectory)
    output2 = interruptions.fix_step_type_on_interruptions(
        self.broken_trajectory)
    # Test output.
    np.testing.assert_allclose(output1, self.correct_trajectory)
    np.testing.assert_allclose(output2, self.correct_trajectory)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
