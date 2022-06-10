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
"""Utilities for correct handling of interruptions."""

import chex
import dm_env
import jax.numpy as jnp


def fix_step_type_on_interruptions(step_type: chex.Array):
  """Returns step_type with a LAST step before almost every FIRST step.

  If the environment crashes or is interrupted while a trajectory is being
  written, the LAST step can be missing before a FIRST step. We add the LAST
  step before each FIRST step, if the step before the FIRST step is a MID step,
  to signal to the agent that the next observation is not connected to the
  current stream of data. Note that the agent must still then appropriately
  handle both `terminations` (e.g. game over in a game) and `interruptions` (a
  timeout or a reset for system maintenance): the value of the discount on LAST
  step will be > 0 on `interruptions`, while it will be 0 on `terminations`.
  Similar issues arise in hierarchical RL systems as well.

  Args:
    step_type: an array of `dm_env` step types, with shape `[T, B]`.

  Returns:
    Fixed step_type.
  """
  chex.assert_rank(step_type, 2)
  next_step_type = jnp.concatenate([
      step_type[1:],
      jnp.full(
          step_type[:1].shape, int(dm_env.StepType.MID), dtype=step_type.dtype),
  ],
                                   axis=0)
  return jnp.where(
      jnp.logical_and(
          jnp.equal(next_step_type, int(dm_env.StepType.FIRST)),
          jnp.equal(step_type, int(dm_env.StepType.MID)),
      ), int(dm_env.StepType.LAST), step_type)
