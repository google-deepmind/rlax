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
"""JAX functions to construct and learn generalized value functions.

According to the reward hypothesis (Sutton et al., 2018) any goal might be
formulated as a suitable scalar `cumulant` to be maximized. Generalized value
functions (Sutton et al. 2011) extend the notion of value functions to include
estimates of discounted sums of `cumulants` different from the main task reward.
"""

import jax.numpy as jnp
from rlax._src import base

ArrayLike = base.ArrayLike


def pixel_control_rewards(
    observations: ArrayLike,
    cell_size: int,
) -> base.ArrayLike:
  """Calculates cumulants for pixel control tasks from an observation sequence.

  The observations are first split in a grid of KxK cells. For each cell a
  distinct pseudo reward is computed as the average absolute change in pixel
  intensity across all pixels in the cell. The change in intensity is averaged
  across both pixels and channels (e.g. RGB).

  The `observations` provided to this function should be cropped suitably, to
  ensure that the observations' height and width are a multiple of `cell_size`.
  The values of the `observations` tensor should be rescaled to [0, 1].

  See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
  Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).

  Args:
    observations: A tensor of shape `[T+1,H,W,C]`, where
      * `T` is the sequence length,
      * `H` is height,
      * `W` is width,
      * `C` is a channel dimension.
    cell_size: The size of each cell.

  Returns:
    A tensor of pixel control rewards calculated from the observation. The
    shape is `[T,H',W']`, where `H'=H/cell_size` and `W'=W/cell_size`.
  """
  base.rank_assert(observations, 4)
  base.type_assert(observations, float)

  # Shape info.
  h = observations.shape[1] // cell_size  # new height.
  w = observations.shape[2] // cell_size  # new width.
  # Calculate the absolute differences across the sequence.
  abs_diff = jnp.abs(observations[1:] - observations[:-1])
  # Average within cells to get the cumulants.
  abs_diff = abs_diff.reshape(
      (-1, h, cell_size, w, cell_size, observations.shape[3]))
  return abs_diff.mean(axis=(2, 4, 5))


def feature_control_rewards(
    features: ArrayLike,
    cumulant_type='absolute_change',
    discount=None,
) -> base.ArrayLike:
  """Calculates cumulants for feature control tasks from a sequence of features.

  For each feature dimension, a distinct pseudo reward is computed based on the
  change in the feature value between consecutive timesteps. Depending on
  `cumulant_type`, cumulants may be equal the features themselves, the absolute
  difference between their values in consecutive steps, their increase/decrease,
  or may take the form of a potential-based reward discounted by `discount`.

  See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
  Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).

  Args:
    features: A tensor of shape `[T+1,D]` of features.
    cumulant_type: either 'feature' (feature is the reward), `absolute_change`
      (the reward equals the absolute difference between consecutive
      timesteps), `increase` (the reward equals the increase in the
      value of the feature), `decrease` (the reward equals the decrease in the
      value of the feature), or 'potential' (r=gamma*phi_{t+1} - phi_t).
    discount: (optional) discount for potential based rewards.

  Returns:
    A tensor of cumulants calculated from the features. The shape is `[T,D]`.
  """
  base.rank_assert(features, 2)
  base.type_assert(features, float)

  if cumulant_type == 'feature':
    return features[1:]
  elif cumulant_type == 'absolute_change':
    return jnp.abs(features[1:] - features[:-1])
  elif cumulant_type == 'increase':
    return features[1:] - features[:-1]
  elif cumulant_type == 'decrease':
    return features[:-1] - features[1:]
  elif cumulant_type == 'potential':
    return discount * features[1:] - features[:-1]
  else:
    raise ValueError('Unknown cumulant_type {}'.format(cumulant_type))
