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
"""JAX functions for implementing forms of gradient clipping.

Gradient clipping is commonly used to avoid taking too large steps in parameter
space when updating the parameters of an agent's policy, value or model. Certain
forms of gradient clipping can be conveniently expressed as transformations of
the loss function optimized by a suitable gradient descent algorithm.
"""

import chex
import jax
import jax.numpy as jnp
from jax.tree_util import tree_multimap

Array = chex.Array


def huber_loss(x: Array, delta: float = 1.) -> Array:
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
  chex.assert_type(x, float)

  # 0.5 * x^2                  if |x| <= d
  # 0.5 * d^2 + d * (|x| - d)  if |x| > d
  abs_x = jnp.abs(x)
  quadratic = jnp.minimum(abs_x, delta)
  # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
  linear = abs_x - quadratic
  return 0.5 * quadratic**2 + delta * linear


@jax.custom_gradient
def clip_gradient(x, gradient_min: float, gradient_max: float):
  """Identity but the gradient in the backward pass is clipped.

  See "Human-level control through deep reinforcement learning" by Mnih et al,
  (https://www.nature.com/articles/nature14236)

  Note `grad(0.5 * clip_gradient(x)**2)` is equivalent to `grad(huber_loss(x))`.

  Note: x cannot be properly annotated because pytype does not support recursive
  types; we would otherwise use the chex.ArrayTree pytype annotation here. Most
  often x will be a single array of arbitrary shape, but the implementation
  supports pytrees.

  Args:
    x: a pytree of arbitrary shape.
    gradient_min: min elementwise size of the gradient.
    gradient_max: max elementwise size of the gradient.

  Returns:
    a vector of same shape of `x`.
  """
  chex.assert_type(x, float)

  def _compute_gradient(g):
    return (tree_multimap(lambda g: jnp.clip(g, gradient_min, gradient_max),
                          g), 0., 0.)

  return x, _compute_gradient
