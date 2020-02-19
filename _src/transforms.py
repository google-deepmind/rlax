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
"""Custom non-linear transformations."""

import jax
import jax.numpy as jnp
from rlax._src import base

ArrayLike = base.ArrayLike


def identity(x: ArrayLike) -> ArrayLike:
  """Identity transform."""
  base.type_assert(x, float)
  return x


def sigmoid(x: ArrayLike) -> ArrayLike:
  """Sigmoid transform."""
  base.type_assert(x, float)
  return jax.nn.sigmoid(x)


def logit(x: ArrayLike) -> ArrayLike:
  """Logit transform, inverse of sigmoid."""
  base.type_assert(x, float)
  return -jnp.log(1. / x - 1.)


def signed_logp1(x: ArrayLike) -> ArrayLike:
  """Signed logarithm of x + 1."""
  base.type_assert(x, float)
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def signed_expm1(x: ArrayLike) -> ArrayLike:
  """Signed exponential of x - 1, inverse of signed_logp1."""
  base.type_assert(x, float)
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def signed_hyperbolic(x: ArrayLike, eps: float = 1e-3) -> ArrayLike:
  """Signed hyperbolic transform, inverse of signed_parabolic."""
  base.type_assert(x, float)
  return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: ArrayLike, eps: float = 1e-3) -> ArrayLike:
  """Signed parabolic transform, inverse of signed_hyperbolic."""
  base.type_assert(x, float)
  z = jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps - 1 / 2 / eps
  return jnp.sign(x) * (jnp.square(z) - 1)


def power(x: ArrayLike, p: float) -> ArrayLike:
  """Power transform; `power_tx(_, 1/p)` is the inverse of `power_tx(_, p)`."""
  base.type_assert(x, float)
  q = jnp.sqrt(p)
  return jnp.sign(x) * (jnp.power(jnp.abs(x) / q + 1., p) - 1) / q
