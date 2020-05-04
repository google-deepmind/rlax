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
"""RLax: building blocks for RL, in JAX."""

from rlax._src.base import ArrayLike
from rlax._src.base import batched_index
from rlax._src.base import one_hot
from rlax._src.base import rank_assert
from rlax._src.base import Scalar
from rlax._src.base import type_assert
from rlax._src.clipping import clip_gradient
from rlax._src.clipping import huber_loss
from rlax._src.distributions import categorical_cross_entropy
from rlax._src.distributions import categorical_importance_sampling_ratios
from rlax._src.distributions import categorical_kl_divergence
from rlax._src.distributions import epsilon_greedy
from rlax._src.distributions import epsilon_softmax
from rlax._src.distributions import gaussian_diagonal
from rlax._src.distributions import greedy
from rlax._src.distributions import multivariate_normal_kl_divergence
from rlax._src.distributions import safe_epsilon_softmax
from rlax._src.distributions import softmax
from rlax._src.general_value_functions import feature_control_rewards
from rlax._src.general_value_functions import pixel_control_rewards
from rlax._src.losses import l2_loss
from rlax._src.losses import likelihood
from rlax._src.losses import log_loss
from rlax._src.losses import pixel_control_loss
from rlax._src.multistep import discounted_returns
from rlax._src.multistep import general_off_policy_returns_from_action_values
from rlax._src.multistep import general_off_policy_returns_from_q_and_v
from rlax._src.multistep import lambda_returns
from rlax._src.nested_updates import incremental_update
from rlax._src.nested_updates import periodic_update
from rlax._src.nonlinear_bellman import IDENTITY_PAIR
from rlax._src.nonlinear_bellman import SIGNED_HYPERBOLIC_PAIR
from rlax._src.nonlinear_bellman import SIGNED_LOGP1_PAIR
from rlax._src.nonlinear_bellman import transformed_general_off_policy_returns_from_action_values
from rlax._src.nonlinear_bellman import transformed_lambda_returns
from rlax._src.nonlinear_bellman import transformed_q_lambda
from rlax._src.nonlinear_bellman import transformed_retrace
from rlax._src.nonlinear_bellman import TxPair
from rlax._src.perturbations import add_gaussian_noise
from rlax._src.perturbations import add_ornstein_uhlenbeck_noise
from rlax._src.policy_gradients import dpg_loss
from rlax._src.policy_gradients import entropy_loss
from rlax._src.policy_gradients import policy_gradient_loss
from rlax._src.schedules import piecewise_constant_schedule
from rlax._src.schedules import polynomial_schedule
from rlax._src.transforms import identity
from rlax._src.transforms import logit
from rlax._src.transforms import power
from rlax._src.transforms import sigmoid
from rlax._src.transforms import signed_expm1
from rlax._src.transforms import signed_hyperbolic
from rlax._src.transforms import signed_logp1
from rlax._src.transforms import signed_parabolic
from rlax._src.tree_util import tree_split_key
from rlax._src.value_learning import categorical_double_q_learning
from rlax._src.value_learning import categorical_q_learning
from rlax._src.value_learning import categorical_td_learning
from rlax._src.value_learning import double_q_learning
from rlax._src.value_learning import expected_sarsa
from rlax._src.value_learning import persistent_q_learning
from rlax._src.value_learning import q_lambda
from rlax._src.value_learning import q_learning
from rlax._src.value_learning import quantile_q_learning
from rlax._src.value_learning import qv_learning
from rlax._src.value_learning import qv_max
from rlax._src.value_learning import sarsa
from rlax._src.value_learning import sarsa_lambda
from rlax._src.value_learning import td_lambda
from rlax._src.value_learning import td_learning
from rlax._src.value_learning import vtrace
from rlax._src.vtrace import vtrace_td_error_and_advantage

__version__ = "0.0.1"

__all__ = (
    "add_gaussian_noise",
    "add_ornstein_uhlenbeck_noise",
    "ArrayLike",
    "batched_index",
    "categorical_cross_entropy",
    "categorical_double_q_learning",
    "categorical_importance_sampling_ratios",
    "categorical_kl_divergence",
    "categorical_q_learning",
    "categorical_td_learning",
    "clip_gradient",
    "discounted_returns",
    "double_q_learning",
    "dpg_loss",
    "entropy_loss",
    "epsilon_greedy",
    "epsilon_softmax",
    "expected_sarsa",
    "feature_control_rewards",
    "gaussian_diagonal",
    "greedy",
    "huber_loss",
    "identity",
    "IDENTITY_PAIR",
    "incremental_update",
    "l2_loss",
    "lambda_returns",
    "likelihood",
    "log_loss",
    "logit",
    "multivariate_normal_kl_divergence",
    "one_hot",
    "periodic_update",
    "persistent_q_learning",
    "piecewise_constant_schedule",
    "pixel_control_rewards",
    "policy_gradient_loss",
    "polynomial_schedule",
    "power",
    "q_lambda",
    "q_learning",
    "general_off_policy_returns_from_action_values",
    "general_off_policy_returns_from_q_and_v",
    "quantile_q_learning",
    "qv_learning",
    "qv_max",
    "rank_assert",
    "sarsa",
    "sarsa_lambda",
    "Scalar",
    "sigmoid",
    "signed_expm1",
    "signed_hyperbolic",
    "SIGNED_HYPERBOLIC_PAIR",
    "SIGNED_LOGP1_PAIR",
    "signed_logp1",
    "signed_parabolic",
    "softmax",
    "td_lambda",
    "td_learning",
    "transformed_general_off_policy_returns_from_action_values",
    "transformed_lambda_returns",
    "transformed_q_lambda",
    "transformed_retrace",
    "tree_split_key",
    "type_assert",
    "TxPair",
    "vtrace",
    "vtrace_td_error_and_advantage",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the RLax public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
