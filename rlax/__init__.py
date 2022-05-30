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

from rlax._src.base import AllSum
from rlax._src.base import batched_index
from rlax._src.base import lhs_broadcast
from rlax._src.base import one_hot
from rlax._src.clipping import clip_gradient
from rlax._src.clipping import huber_loss
from rlax._src.distributions import categorical_cross_entropy
from rlax._src.distributions import categorical_importance_sampling_ratios
from rlax._src.distributions import categorical_kl_divergence
from rlax._src.distributions import categorical_sample
from rlax._src.distributions import clipped_entropy_softmax
from rlax._src.distributions import epsilon_greedy
from rlax._src.distributions import epsilon_softmax
from rlax._src.distributions import gaussian_diagonal
from rlax._src.distributions import greedy
from rlax._src.distributions import multivariate_normal_kl_divergence
from rlax._src.distributions import safe_epsilon_softmax
from rlax._src.distributions import softmax
from rlax._src.distributions import squashed_gaussian
from rlax._src.embedding import embed_oar
from rlax._src.episodic_memory import knn_query
from rlax._src.exploration import add_dirichlet_noise
from rlax._src.exploration import add_gaussian_noise
from rlax._src.exploration import add_ornstein_uhlenbeck_noise
from rlax._src.exploration import episodic_memory_intrinsic_rewards
from rlax._src.general_value_functions import feature_control_rewards
from rlax._src.general_value_functions import pixel_control_rewards
from rlax._src.losses import l2_loss
from rlax._src.losses import likelihood
from rlax._src.losses import log_loss
from rlax._src.losses import pixel_control_loss
from rlax._src.mpo_ops import compute_parametric_kl_penalty_and_dual_loss
from rlax._src.mpo_ops import LagrangePenalty
from rlax._src.mpo_ops import mpo_compute_weights_and_temperature_loss
from rlax._src.mpo_ops import mpo_loss
from rlax._src.mpo_ops import vmpo_compute_weights_and_temperature_loss
from rlax._src.mpo_ops import vmpo_loss
from rlax._src.multistep import discounted_returns
from rlax._src.multistep import general_off_policy_returns_from_action_values
from rlax._src.multistep import general_off_policy_returns_from_q_and_v
from rlax._src.multistep import lambda_returns
from rlax._src.multistep import n_step_bootstrapped_returns
from rlax._src.multistep import truncated_generalized_advantage_estimation
from rlax._src.nested_updates import conditional_update
from rlax._src.nested_updates import incremental_update
from rlax._src.nested_updates import periodic_update
from rlax._src.nonlinear_bellman import compose_tx
from rlax._src.nonlinear_bellman import DISCOUNT_TRANSFORM_PAIR
from rlax._src.nonlinear_bellman import HYPERBOLIC_SIN_PAIR
from rlax._src.nonlinear_bellman import IDENTITY_PAIR
from rlax._src.nonlinear_bellman import muzero_pair
from rlax._src.nonlinear_bellman import SIGNED_HYPERBOLIC_PAIR
from rlax._src.nonlinear_bellman import SIGNED_LOGP1_PAIR
from rlax._src.nonlinear_bellman import transformed_general_off_policy_returns_from_action_values
from rlax._src.nonlinear_bellman import transformed_lambda_returns
from rlax._src.nonlinear_bellman import transformed_n_step_q_learning
from rlax._src.nonlinear_bellman import transformed_n_step_returns
from rlax._src.nonlinear_bellman import transformed_q_lambda
from rlax._src.nonlinear_bellman import transformed_retrace
from rlax._src.nonlinear_bellman import twohot_pair
from rlax._src.nonlinear_bellman import TxPair
from rlax._src.nonlinear_bellman import unbiased_transform_pair
from rlax._src.policy_gradients import clipped_surrogate_pg_loss
from rlax._src.policy_gradients import dpg_loss
from rlax._src.policy_gradients import entropy_loss
from rlax._src.policy_gradients import policy_gradient_loss
from rlax._src.policy_gradients import qpg_loss
from rlax._src.policy_gradients import rm_loss
from rlax._src.policy_gradients import rpg_loss
from rlax._src.pop_art import art
from rlax._src.pop_art import normalize
from rlax._src.pop_art import pop
from rlax._src.pop_art import popart
from rlax._src.pop_art import PopArtState
from rlax._src.pop_art import unnormalize
from rlax._src.pop_art import unnormalize_linear
from rlax._src.transforms import identity
from rlax._src.transforms import logit
from rlax._src.transforms import power
from rlax._src.transforms import sigmoid
from rlax._src.transforms import signed_expm1
from rlax._src.transforms import signed_hyperbolic
from rlax._src.transforms import signed_logp1
from rlax._src.transforms import signed_parabolic
from rlax._src.transforms import transform_from_2hot
from rlax._src.transforms import transform_to_2hot
from rlax._src.tree_util import tree_map_zipped
from rlax._src.tree_util import tree_select
from rlax._src.tree_util import tree_split_key
from rlax._src.tree_util import tree_split_leaves
from rlax._src.value_learning import categorical_double_q_learning
from rlax._src.value_learning import categorical_l2_project
from rlax._src.value_learning import categorical_q_learning
from rlax._src.value_learning import categorical_td_learning
from rlax._src.value_learning import double_q_learning
from rlax._src.value_learning import expected_sarsa
from rlax._src.value_learning import persistent_q_learning
from rlax._src.value_learning import q_lambda
from rlax._src.value_learning import q_learning
from rlax._src.value_learning import quantile_expected_sarsa
from rlax._src.value_learning import quantile_q_learning
from rlax._src.value_learning import quantile_regression_loss
from rlax._src.value_learning import qv_learning
from rlax._src.value_learning import qv_max
from rlax._src.value_learning import retrace
from rlax._src.value_learning import retrace_continuous
from rlax._src.value_learning import sarsa
from rlax._src.value_learning import sarsa_lambda
from rlax._src.value_learning import td_lambda
from rlax._src.value_learning import td_learning
from rlax._src.vtrace import leaky_vtrace
from rlax._src.vtrace import leaky_vtrace_td_error_and_advantage
from rlax._src.vtrace import vtrace
from rlax._src.vtrace import vtrace_td_error_and_advantage

__version__ = "0.1.2"

__all__ = (
    "add_gaussian_noise",
    "add_ornstein_uhlenbeck_noise",
    "add_dirichlet_noise",
    "AllSum",
    "batched_index",
    "categorical_cross_entropy",
    "categorical_double_q_learning",
    "categorical_importance_sampling_ratios",
    "categorical_kl_divergence",
    "categorical_l2_project",
    "categorical_q_learning",
    "categorical_td_learning",
    "clip_gradient",
    "clipped_surrogate_pg_loss",
    "compose_tx",
    "conditional_update",
    "discounted_returns",
    "DISCOUNT_TRANSFORM_PAIR",
    "double_q_learning",
    "dpg_loss",
    "entropy_loss",
    "episodic_memory_intrinsic_rewards",
    "epsilon_greedy",
    "epsilon_softmax",
    "expected_sarsa",
    "feature_control_rewards",
    "gaussian_diagonal",
    "HYPERBOLIC_SIN_PAIR",
    "squashed_gaussian",
    "clipped_entropy_softmax",
    "art",
    "compute_parametric_kl_penalty_and_dual_loss",
    "general_off_policy_returns_from_action_values",
    "general_off_policy_returns_from_q_and_v",
    "greedy",
    "huber_loss",
    "identity",
    "IDENTITY_PAIR",
    "incremental_update",
    "knn_query",
    "l2_loss",
    "LagrangePenalty",
    "lambda_returns",
    "leaky_vtrace",
    "leaky_vtrace_td_error_and_advantage",
    "lhs_broadcast",
    "likelihood",
    "logit",
    "log_loss",
    "mpo_compute_weights_and_temperature_loss",
    "mpo_loss",
    "multivariate_normal_kl_divergence",
    "muzero_pair",
    "normalize",
    "n_step_bootstrapped_returns",
    "one_hot",
    "periodic_update",
    "persistent_q_learning",
    "pixel_control_rewards",
    "policy_gradient_loss",
    "pop",
    "popart",
    "PopArtState",
    "power",
    "qpg_loss",
    "quantile_expected_sarsa",
    "quantile_q_learning",
    "quantile_regression_loss",
    "qv_learning",
    "qv_max",
    "q_lambda",
    "q_learning",
    "retrace",
    "retrace_continuous",
    "rm_loss",
    "rpg_loss",
    "sarsa",
    "sarsa_lambda",
    "sigmoid",
    "signed_expm1",
    "signed_hyperbolic",
    "SIGNED_HYPERBOLIC_PAIR",
    "signed_logp1",
    "SIGNED_LOGP1_PAIR",
    "signed_parabolic",
    "softmax",
    "td_lambda",
    "td_learning",
    "transformed_general_off_policy_returns_from_action_values",
    "transformed_lambda_returns",
    "transformed_n_step_q_learning",
    "transformed_n_step_returns",
    "transformed_q_lambda",
    "transformed_retrace",
    "transform_from_2hot",
    "transform_to_2hot",
    "tree_map_zipped",
    "tree_select",
    "tree_split_key",
    "tree_split_leaves",
    "truncated_generalized_advantage_estimation",
    "twohot_pair",
    "TxPair",
    "unbiased_transform_pair",
    "unnormalize",
    "unnormalize_linear",
    "vmpo_compute_weights_and_temperature_loss",
    "vmpo_loss",
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
