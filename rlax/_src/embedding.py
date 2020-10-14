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
"""JAX functions implementing custom embeddings."""

import chex
import jax
import jax.numpy as jnp

Array = chex.Array


def embed_oar(features: Array, action: Array, reward: Array,
              num_actions: int) -> Array:
  """Embed each of the (observation, action, reward) inputs & concatenate."""
  chex.assert_rank([features, action, reward], [2, 1, 1])
  action = jax.nn.one_hot(action, num_classes=num_actions)  # [B, A]

  reward = jnp.tanh(reward)
  while reward.ndim < action.ndim:
    reward = jnp.expand_dims(reward, axis=-1)

  embedding = jnp.concatenate([features, action, reward], axis=-1)  # [B, D+A+1]
  return embedding
