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
"""Schedules."""

import jax.numpy as jnp
from rlax._src import base


Scalar = base.Scalar


def polynomial_decay(
    init_value: Scalar, end_value: Scalar, power: Scalar, decay_steps: int):
  """Construct a schedule with polynomial decay."""
  def schedule(step_count):
    count = jnp.minimum(step_count, decay_steps)
    return (init_value - end_value)*(1 - count/decay_steps)**(power) + end_value
  return schedule
