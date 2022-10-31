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
"""Utilities to construct and learn from policy targets."""

import chex
import distrax
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class PolicyTarget:
  """A dataclass to hold (possibly sampled) policy targets."""
  # The sampled target actions. These may not cover all actions.
  # The shape is [N_targets, ...]
  actions: chex.Array
  # Probabilities for the corresponding actions. They may have been
  # importance weighted. The shape matches log_prob(actions).shape.
  weights: chex.Array


def sampled_policy_distillation_loss(
    distribution: distrax.DistributionLike,
    policy_targets: PolicyTarget,
) -> chex.Numeric:
  """Compute a sampled cross-entropy-like loss.

  The loss corresponds to taking the mean of `-weights * log_prob(actions)`,
  where the weights and actions come from a `PolicyTarget` object, and the mean
  is computed over the N samples in the `policy_target` as well as any batch
  dimension. The loss is suitable for both discrete and continuous actions.

  Args:
    distribution: a predicted `distrax` or `tfp` distribution.
    policy_targets: a policy target to learn from.

  Returns:
    a scalar loss.
  """
  # Stop gradients from propagating into the targets and into the actions,
  # the latter is mostly relevant in continuous control.
  weights = jax.lax.stop_gradient(policy_targets.weights)
  actions = jax.lax.stop_gradient(policy_targets.actions)
  # Compute log-probabilities.
  log_probs = distribution.log_prob(actions)
  # Assert shapes are compatible.
  chex.assert_equal_shape([weights, log_probs])
  # We avoid NaNs from `0 * (-inf)` by using `0 * min_logp` in that case.
  min_logp = jnp.finfo(log_probs.dtype).min
  # We average over the samples, over time and batch, and if the actions are
  # a continuous vector also over the actions.
  return -jnp.mean(weights * jnp.maximum(log_probs, min_logp))
