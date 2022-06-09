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
"""Functions to support model learning."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp


def extract_subsequences(
    trajectories: chex.Array,
    start_indices: chex.Array,
    subsequence_len: int = 1,
    max_valid_start_idx: Optional[int] = None,
) -> chex.Array:
  """Extract (potentially overlapping) subsequences from batch of trajectories.

  WARNING: If `max_valid_start_idx` is not provided, or incorrectly set,
  the function cannot check the validity of the chosen `start_idx` and no error
  will be raised if indexing outside the data boundaries.

  Args:
    trajectories: A batch of trajectories, shape `[T, B, ...]`.
    start_indices: Time indices of start points, shape `[B, num_start_indices]`.
    subsequence_len: The length of subsequences extracted from `trajectories`.
    max_valid_start_idx: The window used to construct the `start_idx`: i.e. the
      `start_indices` should be from {0, ..., max_valid_start_idx - 1}.

  Returns:
    A batch of subsequences, with
    `trajectories[start_indices[i, j]:start_indices[i, j] + n]` for each start
    index. Output shape is: `[subsequence_len, B, num_start_indices, ...]`.
  """
  if max_valid_start_idx is not None:
    min_len = max_valid_start_idx + subsequence_len - 1
    traj_len = trajectories.shape[0]
    if traj_len < min_len:
      raise AssertionError(
          f'Expected len >= {min_len}, but trajectories length is: {traj_len}.')
  batch_size = start_indices.shape[0]
  batch_range = jnp.arange(batch_size)
  num_subs = start_indices.shape[1]
  slices = []
  for i in range(num_subs):
    slices.append(jnp.stack(
        [trajectories[start_indices[:, i] + k, batch_range]
         for k in range(subsequence_len)], axis=0))
  return jnp.stack(slices, axis=2)


def sample_start_indices(
    rng_key: chex.PRNGKey,
    batch_size: int,
    num_start_indices: int,
    max_valid_start_idx: int
) -> chex.Array:
  """Sampling `batch_size x num_start_indices` starting indices.

  Args:
    rng_key: a pseudo random number generator's key.
    batch_size: the size of the batch of trajectories to index in.
    num_start_indices: how many starting points per trajectory in the batch.
    max_valid_start_idx: maximum valid time index for all starting points.

  Returns:
    an array of starting points with shape `[B, num_start_indices]`
  """

  @jax.vmap
  def _vchoose(key, entries):
    return jax.random.choice(
        key, entries, shape=(num_start_indices,), replace=False)

  rollout_window = jnp.arange(max_valid_start_idx)
  return _vchoose(
      jax.random.split(rng_key, batch_size),
      jnp.tile(rollout_window, (batch_size, 1)))
