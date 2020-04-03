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
"""JAX Schedules.

Schedules may be used to anneal the value of a hyper-parameter over time; for
instance, they may be used to anneal the learning rate used to update an agent's
parameters or the exploration factor used to select actions.
"""

from typing import Dict
import jax.numpy as jnp
from rlax._src import base


Scalar = base.Scalar


def polynomial_schedule(
    init_value: Scalar,
    end_value: Scalar,
    power: Scalar,
    transition_steps: int,
    transition_begin: int = 0):
  """Construct a schedule with polynomial transition from init to end value."""
  if transition_steps < 0:
    raise ValueError('transition_steps must be a non-negative integer.')

  elif transition_steps == 0:
    return lambda step_count: end_value

  else:
    def schedule(step_count):
      count = jnp.clip(step_count - transition_begin, 0, transition_steps)
      frac = 1 - count / transition_steps
      return (init_value - end_value) * (frac**power) + end_value
    return schedule


def piecewise_constant_schedule(
    init_value: float,
    boundaries_and_scales: Dict[int, float] = None):
  """Returns a function which implements a piecewise constant schedule.

  Args:
    init_value: An initial value `init_v`.
    boundaries_and_scales: A map from boundaries `b_i` to non-negative scaling
      factors `f_i`. For any step count `s`, the schedule returns `init_v`
      scaled by the product of all factors `f_i` such that `b_i` < `s`.

  schedule:
    step_size_fn: A function that maps step counts to values.
  """
  all_positive = all(scale >= 0. for scale in boundaries_and_scales.values())
  if not all_positive:
    raise ValueError(
        'The `piecewise_constant_schedule` expects non-negative scale factors')

  def schedule(count):
    v = init_value
    if boundaries_and_scales is not None:
      for threshold, scale in sorted(boundaries_and_scales.items()):
        indicator = jnp.max([0., jnp.sign(threshold - count)])
        v = v * indicator + (1 - indicator) * scale * v
    return v

  return schedule
