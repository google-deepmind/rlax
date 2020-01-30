# Lint as: python3
# coding=utf8
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
"""Common losses."""

from typing import Optional
import jax.numpy as jnp
from rlax._src import base

ArrayLike = base.ArrayLike


def l2_loss(predictions: ArrayLike,
            targets: Optional[ArrayLike] = None) -> ArrayLike:
  """Caculates the L2 loss of predictions wrt targets.

  If targets are not provided this function acts as an L2-regularizer for preds.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  if targets is None:
    targets = jnp.zeros_like(predictions)
  base.type_assert([predictions, targets], float)
  return 0.5 * (predictions - targets)**2


def likelihood(predictions: ArrayLike, targets: ArrayLike) -> ArrayLike:
  """Calculates the likelihood of predictions wrt targets.

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  base.type_assert([predictions, targets], float)
  return targets * predictions + (1. - targets) * (1. - predictions)


def log_loss(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> ArrayLike:
  """Calculates the log loss of predictions wrt targets.

  Args:
    predictions: a vector of probabilities of arbitrary shape.
    targets: a vector of probabilities of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  base.type_assert([predictions, targets], float)
  return -jnp.log(likelihood(predictions, targets))
