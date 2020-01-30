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
"""Functions for implementing forms of gradient clipping."""

import jax
import jax.numpy as jnp
from rlax._src import base

ArrayLike = base.ArrayLike


def huber_loss(x: ArrayLike, delta: float = 1.) -> ArrayLike:
  """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.

  See "Robust Estimation of a Location Parameter" by Huber.
  (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).

  Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.

  Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.

  Returns:
    a vector of same shape of `x`.
  """
  base.type_assert(x, float)

  # 0.5 * x^2                  if |x| <= d
  # 0.5 * d^2 + d * (|x| - d)  if |x| > d
  abs_x = jnp.abs(x)
  quadratic = jnp.minimum(abs_x, delta)
  # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
  linear = abs_x - quadratic
  return 0.5 * quadratic**2 + delta * linear


@jax.custom_gradient
def clip_gradient(x: ArrayLike, gradient_min: float, gradient_max: float):
  """Identity but the gradient in the backward pass is clipped.

  See "Human-level control through deep reinforcement learning" by Mnih et al,
  (https://www.nature.com/articles/nature14236)

  Note `grad(0.5 * clip_gradient(x)**2)` is equivalent to `grad(huber_loss(x))`.

  Args:
    x: a vector of arbitrary shape.
    gradient_min: min elementwise size of the gradient.
    gradient_max: max elementwise size of the gradient.

  Returns:
    a vector of same shape of `x`.
  """
  base.type_assert(x, float)
  return x, lambda g: (jnp.clip(g, gradient_min, gradient_max), 0., 0.)
