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
"""JAX functions implementing custom non-linear transformations.

This is a collection of element-wise non-linear transformations that may be
used to transform losses, value estimates, or other multidimensional data.
"""

import chex
import jax
import jax.numpy as jnp
from rlax._src import base

Array = chex.Array
Numeric = chex.Numeric


def identity(x: Array) -> Array:
  """Identity transform."""
  chex.type_assert(x, float)
  return x


def sigmoid(x: Numeric) -> Array:
  """Sigmoid transform."""
  chex.type_assert(x, float)
  return jax.nn.sigmoid(x)


def logit(x: Array) -> Array:
  """Logit transform, inverse of sigmoid."""
  chex.type_assert(x, float)
  return -jnp.log(1. / x - 1.)


def signed_logp1(x: Array) -> Array:
  """Signed logarithm of x + 1."""
  chex.type_assert(x, float)
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def signed_expm1(x: Array) -> Array:
  """Signed exponential of x - 1, inverse of signed_logp1."""
  chex.type_assert(x, float)
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def signed_hyperbolic(x: Array, eps: float = 1e-3) -> Array:
  """Signed hyperbolic transform, inverse of signed_parabolic."""
  chex.type_assert(x, float)
  return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: Array, eps: float = 1e-3) -> Array:
  """Signed parabolic transform, inverse of signed_hyperbolic."""
  chex.type_assert(x, float)
  z = jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps - 1 / 2 / eps
  return jnp.sign(x) * (jnp.square(z) - 1)


def power(x: Array, p: float) -> Array:
  """Power transform; `power_tx(_, 1/p)` is the inverse of `power_tx(_, p)`."""
  chex.type_assert(x, float)
  q = jnp.sqrt(p)
  return jnp.sign(x) * (jnp.power(jnp.abs(x) / q + 1., p) - 1) / q


def transform_to_2hot(
    scalar: Array,
    min_value: float,
    max_value: float,
    num_bins: int) -> Array:
  """Transforms a scalar tensor to a 2 hot representation."""
  scalar = jnp.clip(scalar, min_value, max_value)
  scalar_bin = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
  lower, upper = jnp.floor(scalar_bin), jnp.ceil(scalar_bin)
  lower_value = (lower / (num_bins - 1.0)) * (max_value - min_value) + min_value
  upper_value = (upper / (num_bins - 1.0)) * (max_value - min_value) + min_value
  p_lower = (upper_value - scalar) / (upper_value - lower_value + 1e-5)
  p_upper = 1 - p_lower
  lower_one_hot = base.one_hot(lower, num_bins) * jnp.expand_dims(p_lower, -1)
  upper_one_hot = base.one_hot(upper, num_bins) * jnp.expand_dims(p_upper, -1)
  return lower_one_hot + upper_one_hot


def transform_from_2hot(
    probs: Array,
    min_value: float,
    max_value: float,
    num_bins: int) -> Array:
  """Transforms from a categorical distribution to a scalar."""
  support_space = jnp.linspace(min_value, max_value, num_bins)
  scalar = jnp.sum(probs * jnp.expand_dims(support_space, 0), -1)
  return scalar
