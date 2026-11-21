* Values, including both state and action-values;
* Values for Non-linear generalizations of the Bellman equations.
* Return Distributions, aka distributional value functions;
* General Value Functions, for cumulants other than the main reward;
* Policies, via policy-gradients in both continuous and discrete action spaces.

Value Learning
==============

.. currentmodule:: rlax

.. autosummary::

    categorical_double_q_learning
    categorical_l2_project
    categorical_q_learning
    categorical_td_learning
    discounted_returns
    double_q_learning
    expected_sarsa
    general_off_policy_returns_from_action_values
    general_off_policy_returns_from_q_and_v
    lambda_returns
    leaky_vtrace
    leaky_vtrace_td_error_and_advantage
    n_step_bootstrapped_returns
    persistent_q_learning
    q_lambda
    q_learning
    quantile_expected_sarsa
    quantile_q_learning
    quantile_regression_loss
    qv_learning
    qv_max
    retrace
    retrace_continuous
    sarsa
    sarsa_lambda
    td_lambda
    td_learning
    transformed_general_off_policy_returns_from_action_values
    transformed_lambda_returns
    transformed_n_step_q_learning
    transformed_n_step_returns
    transformed_q_lambda
    transformed_retrace
    vtrace
    vtrace_td_error_and_advantage


Categorical Double Q Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_double_q_learning

Categorical L2 Project
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_l2_project

Categorical Q Learning
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_q_learning

Categorical TD Learning
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_td_learning

Discounted Returns
~~~~~~~~~~~~~~~~~~

.. autofunction:: discounted_returns

Double Q Learning
~~~~~~~~~~~~~~~~~

.. autofunction:: double_q_learning

Expected SARSA
~~~~~~~~~~~~~~

.. autofunction:: expected_sarsa

General Off Policy Returns From Action Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: general_off_policy_returns_from_action_values

General Off Policy Returns From Q and V
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: general_off_policy_returns_from_q_and_v

Lambda Returns
~~~~~~~~~~~~~~

.. autofunction:: lambda_returns

Leaky VTrace
~~~~~~~~~~~~

.. autofunction:: leaky_vtrace

N Step Bootstrapped Returns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: n_step_bootstrapped_returns

Leaky VTrace TD Error and Advantage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: leaky_vtrace_td_error_and_advantage

Persistent Q Learning
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: persistent_q_learning

Q-Lambda
~~~~~~~~

.. autofunction:: q_lambda

Q Learning
~~~~~~~~~~

.. autofunction:: q_learning


Quantile Expected Sarsa
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: quantile_expected_sarsa

Quantile Q Learning
~~~~~~~~~~~~~~~~~~~

.. autofunction:: quantile_q_learning

QV Learning
~~~~~~~~~~~

.. autofunction:: qv_learning

QV Max
~~~~~~~

.. autofunction:: qv_max

Retrace
~~~~~~~

.. autofunction:: retrace

Retrace Continuous
~~~~~~~~~~~~~~~~~~

.. autofunction:: retrace_continuous

SARSA
~~~~~

.. autofunction:: sarsa

SARSA Lambda
~~~~~~~~~~~~

.. autofunction:: sarsa_lambda

TD Lambda
~~~~~~~~~

.. autofunction:: td_lambda

TD Learning
~~~~~~~~~~~

.. autofunction:: td_learning

Transformed General Off Policy Returns from Action Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformed_general_off_policy_returns_from_action_values

Transformed Lambda Returns
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformed_lambda_returns

Transformed N Step Q Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformed_n_step_q_learning

Transformed N Step Returns
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformed_n_step_returns

Transformed Q Lambda
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformed_q_lambda


Transformed Retrace
~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformed_retrace

Truncated Generalized Advantage Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: truncated_generalized_advantage_estimation

VTrace
~~~~~~

.. autofunction:: vtrace


Policy Optimization
===================

.. currentmodule:: rlax

.. autosummary::

    clipped_surrogate_pg_loss
    cmpo_policy_targets
    constant_policy_targets
    dpg_loss
    entropy_loss
    mpo_loss
    mpo_compute_weights_and_temperature_loss
    policy_gradient_loss
    qpg_loss
    rm_loss
    rpg_loss
    sampled_cmpo_policy_targets
    sampled_policy_distillation_loss
    zero_policy_targets

Clipped Surrogate PG Loss
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: clipped_surrogate_pg_loss


CMPO Policy Targets
~~~~~~~~~~~~~~~~~~~

.. autofunction:: cmpo_policy_targets


Sampled CMPO Policy Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sampled_cmpo_policy_targets


Compute Parametric KL Penalty and Dual Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: compute_parametric_kl_penalty_and_dual_loss

DPG Loss
~~~~~~~~

.. autofunction:: dpg_loss

Entropy Loss
~~~~~~~~~~~~

.. autofunction:: entropy_loss


MPO Compute Weights and Temperature Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mpo_compute_weights_and_temperature_loss

MPO Loss
~~~~~~~~

.. autofunction:: mpo_loss

Policy Gradient Loss
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: policy_gradient_loss

QPG Loss
~~~~~~~~

.. autofunction:: qpg_loss

RM Loss
~~~~~~~~

.. autofunction:: rm_loss

RPG Loss
~~~~~~~~

.. autofunction:: rpg_loss

Constant Policy Targets
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: constant_policy_targets

Zero Policy Targets
~~~~~~~~~~~~~~~~~~~

.. autofunction:: zero_policy_targets

Sampled Policy Distillation Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sampled_policy_distillation_loss

VMPO Compute Weights and Temperature Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vmpo_compute_weights_and_temperature_loss

VMPO Loss
~~~~~~~~~

.. autofunction:: vmpo_loss


Exploration
===========

.. currentmodule:: rlax

.. autosummary::

    add_dirichlet_noise
    add_gaussian_noise
    add_ornstein_uhlenbeck_noise
    episodic_memory_intrinsic_rewards
    knn_query


Add Dirichlet Noise
~~~~~~~~~~~~~~~~~~~

.. autofunction:: add_dirichlet_noise

Add Gaussian Noise
~~~~~~~~~~~~~~~~~~

.. autofunction:: add_gaussian_noise

Add Ornstein Uhlenbeck Noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: add_ornstein_uhlenbeck_noise

Episodic Memory Intrinsic Rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: episodic_memory_intrinsic_rewards

KNN Query
~~~~~~~~~

.. autofunction:: knn_query


Utilities
=========


.. currentmodule:: rlax

.. autosummary::

    AllSum
    batched_index
    clip_gradient
    create_ema
    fix_step_type_on_interruptions
    lhs_broadcast
    one_hot
    embed_oar
    replace_masked
    transpose_first_axis_to_last
    transpose_last_axis_to_first
    tree_fn
    tree_map_zipped
    tree_replace_masked
    tree_select
    tree_split_key
    tree_split_leaves
    conditional_update
    periodic_update


All Sum
~~~~~~~

.. autoclass:: AllSum
    :members:

Batched Index
~~~~~~~~~~~~~

.. autofunction:: batched_index

Clip Gradient
~~~~~~~~~~~~~

.. autofunction:: clip_gradient

Create Ema
~~~~~~~~~~~~~

.. autofunction:: create_ema

LHS Broadcast
~~~~~~~~~~~~~

.. autofunction:: lhs_broadcast

One Hot
~~~~~~~

.. autofunction:: one_hot

Embed OAR
~~~~~~~~~

.. autofunction:: embed_oar

Fix Step Type
~~~~~~~~~~~~~

.. autofunction:: fix_step_type_on_interruptions

Transpose First Axis To Last
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transpose_first_axis_to_last

Transpose Last Axis to First
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transpose_last_axis_to_first

Replace Masked
~~~~~~~~~~~~~~

.. autofunction:: replace_masked

Tree Map Zipped
~~~~~~~~~~~~~~~

.. autofunction:: tree_map_zipped

Tree Replace Masked
~~~~~~~~~~~~~~~~~~~

.. autofunction:: tree_replace_masked

Tree Select
~~~~~~~~~~~

.. autofunction:: tree_select

Tree Split Key
~~~~~~~~~~~~~~

.. autofunction:: tree_split_key

Tree Split Leaves
~~~~~~~~~~~~~~~~~

.. autofunction:: tree_split_leaves

Conditional Update
~~~~~~~~~~~~~~~~~~

.. autofunction:: conditional_update

Periodic Update
~~~~~~~~~~~~~~~

.. autofunction:: periodic_update


General Value Functions
=======================


.. currentmodule:: rlax

.. autosummary::

    pixel_control_rewards
    feature_control_rewards


Pixel Control Rewards
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pixel_control_rewards

Feature Control Rewards
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: feature_control_rewards


Model Learning
==============


.. currentmodule:: rlax

.. autosummary::

    extract_subsequences
    sample_start_indices

Extract model training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: extract_subsequences

.. autofunction:: sample_start_indices


Pop Art
========


.. currentmodule:: rlax

.. autosummary::

    art
    normalize
    pop
    popart
    unnormalize
    unnormalize_linear

Art
~~~

.. autofunction:: art

Normalize
~~~~~~~~~

.. autofunction:: normalize

Pop
~~~

.. autofunction:: pop

PopArt
~~~~~~

.. autofunction:: popart

Unnormalize
~~~~~~~~~~~

.. autofunction:: unnormalize

Unnormalize Linear
~~~~~~~~~~~~~~~~~~

.. autofunction:: unnormalize_linear



Transforms
==========


.. currentmodule:: rlax

.. autosummary::

    compose_tx
    DISCOUNT_TRANSFORM_PAIR
    HYPERBOLIC_SIN_PAIR
    identity
    IDENTITY_PAIR
    logit
    muzero_pair
    power
    sigmoid
    signed_expm1
    signed_hyperbolic
    SIGNED_HYPERBOLIC_PAIR
    signed_logp1
    SIGNED_LOGP1_PAIR
    signed_parabolic
    transform_from_2hot
    transform_to_2hot
    twohot_pair
    TxPair
    unbiased_transform_pair


Identity
~~~~~~~~

.. autofunction:: identity


Logit
~~~~~

.. autofunction:: logit


Power
~~~~~

.. autofunction:: power


Sigmoid
~~~~~~~

.. autofunction:: sigmoid


Signed Exponential
~~~~~~~~~~~~~~~~~~

.. autofunction:: signed_expm1


Signed Hyperbolic
~~~~~~~~~~~~~~~~~

.. autofunction:: signed_hyperbolic


Signed Logarithm
~~~~~~~~~~~~~~~~

.. autofunction:: signed_logp1


Signed Parabolic
~~~~~~~~~~~~~~~~

.. autofunction:: signed_parabolic


Transform from 2 Hot
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transform_from_2hot

Transform to 2 Hot
~~~~~~~~~~~~~~~~~~

.. autofunction:: transform_to_2hot


Losses
======

.. currentmodule:: rlax

.. autosummary::

    l2_loss
    likelihood
    log_loss
    huber_loss
    pixel_control_loss


L2 Loss
~~~~~~~

.. autofunction:: l2_loss


Likelihood
~~~~~~~~~~

.. autofunction:: likelihood

Log Loss
~~~~~~~~

.. autofunction:: log_loss

Huber Loss
~~~~~~~~~~

.. autofunction:: huber_loss

Pixel Control Loss
~~~~~~~~~~~~~~~~~~

 .. autofunction:: pixel_control_loss


Distributions
=============

.. currentmodule:: rlax

.. autosummary::

    categorical_cross_entropy
    categorical_importance_sampling_ratios
    categorical_kl_divergence
    categorical_sample
    clipped_entropy_softmax
    epsilon_greedy
    gaussian_diagonal
    greedy
    multivariate_normal_kl_divergence
    softmax
    squashed_gaussian


Categorical Cross Entropy
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_cross_entropy


Categorical Importance Sampling Ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_importance_sampling_ratios

Categorical KL Divergence
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_kl_divergence

Categorical Sample
~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_sample

Clipped Entropy Softmax
~~~~~~~~~~~~~~~~~~~~~~~

 .. autofunction:: clipped_entropy_softmax

Epsilon Greedy
~~~~~~~~~~~~~~

.. autofunction:: epsilon_greedy

Gaussian Diagonal
~~~~~~~~~~~~~~~~~

.. autofunction:: gaussian_diagonal

Greedy
~~~~~~

 .. autofunction:: greedy

Multivariate Normal KL Divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: multivariate_normal_kl_divergence

Softmax
~~~~~~~

.. autofunction:: softmax

Squashed Gaussian
~~~~~~~~~~~~~~~~~

.. autofunction:: squashed_gaussian

