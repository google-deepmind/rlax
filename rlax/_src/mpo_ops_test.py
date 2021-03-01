# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for mpo_ops.py."""

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rlax._src import distributions
from rlax._src import mpo_ops

NUM_SAMPLES = 10
ACTION_DIM = 3
TIME_DIM = 8
BATCH_DIM = 100

# NOTE: These are not typical values used for MPO. In the test case, we know the
# Q function perfectly so we loosen the bound on the mean to zone in to the
# optimal policy very quickly. Similarly, we maintain a high variance to sample
# distinct actions to explore and learn from.
_INIT_TEMPERATURE = 0.2
_INIT_ALPHA_MEAN = 0.001
_INIT_ALPHA_COVARIANCE = float(1e6)

_EPSILON_BOUND = 0.01
_EPSILON_MEAN_BOUND = 10.0
_EPSILON_COVARIANCE_BOUND = 1e-12

_NUM_ITERATIONS = 5000
_TARGET_UPDATE_PERIOD = 100
_RANDOM_SEED = 42

# The offset to ensure initially the policy is not close to 0
_MEAN_OFFSET = 2.0

# The final action should optimize down to be close to 0.0
_MAX_ACTION_ERROR = 0.2
_MAX_KL_ERROR = 1e-6

_DIAGONAL_GAUSSIAN_DIST = distributions.gaussian_diagonal()
_PROJECTION_OPERATOR = functools.partial(jnp.clip, a_min=1e-10)


def _hk_mock_policy_params(s_tm1):
  """Returns mock policy params."""
  # Outputs of the network are mu and sigma. Both shaped [B, ACTION_DIM].
  pi_out = hk.nets.MLP(
      output_sizes=[2 * ACTION_DIM],
      w_init=hk.initializers.VarianceScaling(1e-3),
      activation=jnp.tanh,
      activate_final=False,
      name='online_policy')(s_tm1)
  pi_mean, pi_cov = jnp.split(pi_out, 2, axis=-1)
  pi_cov = jax.nn.softplus(pi_cov)
  pi_mean = pi_mean + _MEAN_OFFSET
  return {'mean': pi_mean, 'stddev': pi_cov}


def _init_params(key):
  init_fn, _ = hk.transform(_hk_mock_policy_params, apply_rng=True)
  key_seq = hk.PRNGSequence(key)
  s_tm1 = jax.random.normal(
      next(key_seq), (TIME_DIM, BATCH_DIM, ACTION_DIM), jnp.float32)
  online_params = init_fn(next(key_seq), s_tm1)
  return dict(
      online=online_params,
      target=online_params,
      mpo=dict(
          temperature=_INIT_TEMPERATURE,
          alpha_mean=_INIT_ALPHA_MEAN,
          alpha_covariance=_INIT_ALPHA_COVARIANCE),
  )


def _mock_outputs(online_params, target_params, key, target_name):
  """Returns mock network outputs."""
  _, policy_params_fn = hk.transform(_hk_mock_policy_params, apply_rng=True)
  key_seq = hk.PRNGSequence(key)

  state_size = ACTION_DIM

  # Input state: [TIME_DIM, BATCH_DIM, DIM_STATE]
  s_tm1 = jax.random.normal(
      next(key_seq), (TIME_DIM, BATCH_DIM, state_size), jnp.float32)
  policy_params = policy_params_fn(online_params, None, s_tm1)
  target_policy_params = policy_params_fn(target_params, None, s_tm1)

  # Shape for actions: [NUM_SAMPLES, TIME_DIM, BATCH_DIM, ACTION_DIM]
  mean, stddev = target_policy_params['mean'], target_policy_params['stddev']
  mean_repeated = jnp.repeat(
      mean.reshape((1,) + mean.shape), NUM_SAMPLES, axis=0)
  stddev_repeated = jnp.repeat(
      stddev.reshape((1,) + stddev.shape), NUM_SAMPLES, axis=0)
  target_actions = _DIAGONAL_GAUSSIAN_DIST.sample(
      next(key_seq), mean_repeated, stddev_repeated)
  # If the target is advantages then num samples is 1.
  if target_name == 'advantages':
    target_actions = target_actions[0, ...]

  # Shape for Q: [NUM_SAMPLES, TIME_DIM, BATCH_DIM]
  # Setting Q = -a_t * tf.transpose(a_t) where a_t = s_t + a.
  # The solution to optimizing this is basically for the policy to output
  # 0 actions thereby minimizing the cost. Since this is a convex
  # optimization problem, the algorithm should get to a good solution quickly.

  # First compute a_t = s_t + a with shape: [NUM_SAMPLES, TIME_DIM, BATCH_DIM,
  # ACTION_DIM] since action dim is the same as shape dim here and then compute
  # the quadratic form.
  a_t = target_actions + jnp.expand_dims(s_tm1, 0)
  sample_q_values = -jnp.sum(a_t ** 2, axis=-1)
  # Set the advantage to the same as the q value.
  # Shape for advantages: [TIME_DIM, BATCH_DIM]
  advantages = sample_q_values[0, :, :]

  return dict(
      pi_params=policy_params,
      target_pi_params=target_policy_params,
      sample_q_values=sample_q_values,
      advantages=advantages,
      target_actions=target_actions,
  )


def get_common_loss_fn_inputs(params, key, target_name):
  out = _mock_outputs(params['online'], params['target'], key, target_name)
  pi_sample_log_probs = _DIAGONAL_GAUSSIAN_DIST.logprob(
      out['target_actions'], out['pi_params']['mean'],
      out['pi_params']['stddev'])

  return out, {
      'sample_log_probs': pi_sample_log_probs,
      target_name: out[target_name],
      'temperature_constraint': mpo_ops.LagrangePenalty(
          params['mpo']['temperature'], _EPSILON_BOUND)}


def get_decoupled_kl_constraints(out, params, per_dimension):
  # Factorize KL for Gaussian.
  kl_mean, kl_covariance = (
      distributions.decoupled_multivariate_normal_kl_divergence(
          out['target_pi_params']['mean'], out['target_pi_params']['stddev'],
          out['pi_params']['mean'], out['pi_params']['stddev'],
          per_dimension=per_dimension))
  alpha_mean = params['mpo']['alpha_mean'] * jnp.ones_like(kl_mean)
  alpha_covariance = params['mpo']['alpha_covariance'] * jnp.ones_like(
      kl_covariance)

  return [
      (kl_mean, mpo_ops.LagrangePenalty(
          alpha=alpha_mean, epsilon=_EPSILON_MEAN_BOUND,
          per_dimension=per_dimension)),
      (kl_covariance, mpo_ops.LagrangePenalty(
          alpha=alpha_covariance, epsilon=_EPSILON_COVARIANCE_BOUND,
          per_dimension=per_dimension)),
  ]


def get_coupled_kl_constraints(out, params, per_dimension):
  kl_mean, kl_covariance = (
      distributions.decoupled_multivariate_normal_kl_divergence(
          out['target_pi_params']['mean'], out['target_pi_params']['stddev'],
          out['pi_params']['mean'], out['pi_params']['stddev'],
          per_dimension=per_dimension))
  alpha_mean = params['mpo']['alpha_mean'] * jnp.ones_like(kl_mean)
  return [
      (kl_mean + kl_covariance, mpo_ops.LagrangePenalty(
          alpha=alpha_mean,
          epsilon=_EPSILON_MEAN_BOUND + _EPSILON_COVARIANCE_BOUND,
          per_dimension=per_dimension))
  ]


def vmpo_e_step_without_restarting_or_importance_weights(advantages, **kwargs):
  restarting_weights = jnp.ones_like(advantages)
  importance_weights = jnp.ones_like(advantages)
  return mpo_ops.vmpo_compute_weights_and_temperature_loss(
      advantages=advantages, restarting_weights=restarting_weights,
      importance_weights=importance_weights, **kwargs)


class MPOTest(parameterized.TestCase):
  """Tests for the MPO losses."""

  @parameterized.parameters(
      {'target_name': 'sample_q_values',
       'loss_fn': mpo_ops.mpo_loss,
       'get_kl_constraints': get_decoupled_kl_constraints,
       'per_dimension': False},
      {'target_name': 'advantages',
       'loss_fn': mpo_ops.vmpo_loss,
       'get_kl_constraints': get_decoupled_kl_constraints,
       'per_dimension': False},
      {'target_name': 'sample_q_values',
       'loss_fn': mpo_ops.mpo_loss,
       'get_kl_constraints': get_coupled_kl_constraints,
       'per_dimension': False},
      {'target_name': 'advantages',
       'loss_fn': mpo_ops.vmpo_loss,
       'get_kl_constraints': get_coupled_kl_constraints,
       'per_dimension': False},
      {'target_name': 'sample_q_values',
       'loss_fn': mpo_ops.mpo_loss,
       'get_kl_constraints': get_decoupled_kl_constraints,
       'per_dimension': True},
      {'target_name': 'advantages',
       'loss_fn': mpo_ops.vmpo_loss,
       'get_kl_constraints': get_decoupled_kl_constraints,
       'per_dimension': True},
      {'target_name': 'sample_q_values',
       'loss_fn': mpo_ops.mpo_loss,
       'get_kl_constraints': get_coupled_kl_constraints,
       'per_dimension': True},
      {'target_name': 'advantages',
       'loss_fn': mpo_ops.vmpo_loss,
       'get_kl_constraints': get_coupled_kl_constraints,
       'per_dimension': True},
  )
  def test_optimization(
      self, target_name, loss_fn, get_kl_constraints, per_dimension):
    """Tests that the policy optimization works correctly."""

    def _loss(params, key):
      out, loss_fn_inputs = get_common_loss_fn_inputs(params, key, target_name)
      kl_constraints = get_kl_constraints(out, params, per_dimension)
      loss_fn_inputs.update({'kl_constraints': kl_constraints})
      loss, mpo_stats = loss_fn(**loss_fn_inputs)
      loss = jnp.mean(loss)
      temperature_bound = jnp.mean(mpo_stats.normalized_weights * jnp.log(
          mpo_stats.num_samples * mpo_stats.normalized_weights + 1e-8))
      return loss, {'outputs': out, 'temperature_bound': temperature_bound}

    key = jax.random.PRNGKey(_RANDOM_SEED)
    grad_fn = jax.jit(jax.grad(_loss, has_aux=True))
    optimizer = optax.adam(1e-3)
    key, new_key = jax.random.split(key)
    params = _init_params(new_key)
    opt_state = optimizer.init((params['online'], params['mpo']))

    @jax.jit
    def _update(params_, opt_state_, key_):
      next_key, key_ = jax.random.split(key_)
      grad, stats = grad_fn(params_, key_)
      updates, opt_state_ = optimizer.update(
          (grad['online'], grad['mpo']), opt_state_)
      online_params, mpo_params = optax.apply_updates(
          (params_['online'], params_['mpo']), updates)
      params_['online'] = online_params
      params_['mpo'] = mpo_params
      return params_, opt_state_, stats, next_key

    for iter_idx in range(_NUM_ITERATIONS):
      params, opt_state, extra, key = _update(params, opt_state, key)
      if iter_idx % _TARGET_UPDATE_PERIOD == 0:
        params['target'] = params['online']

    # Test the bounds are within tolerance.
    key, new_key = jax.random.split(key)
    _, extra = _loss(params, new_key)
    action_mean = jnp.mean(extra['outputs']['pi_params']['mean'])
    # Check action mean is close to 0.
    self.assertBetween(action_mean, -_MAX_ACTION_ERROR, _MAX_ACTION_ERROR)

    # Check the temperature are within the bounds.
    self.assertLess(extra['temperature_bound'], _EPSILON_BOUND)

  @parameterized.parameters(
      {'e_step_fn': mpo_ops.mpo_compute_weights_and_temperature_loss,
       'additional_inputs': {},
       # dL/dq == 1 and dL/dt == epsilon (for one sample)
       'expected_deriv_of_target': [[[1]]],
       'sample_dimension': True},
      {'e_step_fn': vmpo_e_step_without_restarting_or_importance_weights,
       'additional_inputs': {'top_k_fraction': 1.0},
       'expected_deriv_of_target': [[1]],
       'sample_dimension': False},
  )
  def test_e_step_gradient_computation(
      self, e_step_fn, additional_inputs, expected_deriv_of_target,
      sample_dimension):
    """Tests the gradients from the E-step against the analytic ones."""
    # Target has shape [NUM_SAMPLES, T, B] => [1, 1, 1]
    target = jnp.array([[3]], jnp.float32)
    if sample_dimension:
      target = jnp.expand_dims(target, axis=0)
    temperature = jnp.array(0.1, jnp.float32)
    def fn(target_, temperature_):
      temperature_constraint = mpo_ops.LagrangePenalty(
          temperature_, _EPSILON_BOUND)
      temperature_loss, _, _ = e_step_fn(
          target_, temperature_constraint=temperature_constraint,
          projection_operator=_PROJECTION_OPERATOR,
          **additional_inputs)
      return jnp.mean(temperature_loss)
    grad = jax.grad(fn, argnums=(0, 1))(target, temperature)

    np.testing.assert_almost_equal(np.array(grad[0]), np.array(
        expected_deriv_of_target, np.float32), decimal=4)
    self.assertAlmostEqual(grad[1], _EPSILON_BOUND, places=4)

  @parameterized.parameters(
      {'e_step_fn': mpo_ops.mpo_compute_weights_and_temperature_loss,
       'additional_inputs': {},
       'sample_dimension': True},
      {'e_step_fn': vmpo_e_step_without_restarting_or_importance_weights,
       'additional_inputs': {'top_k_fraction': 1.0},
       'sample_dimension': False},
  )
  def test_e_step_stop_gradient(
      self, e_step_fn, additional_inputs, sample_dimension):
    """Tests no gradients flow through `weights` in the E-Step."""
    # Target has shape [NUM_SAMPLES, T, B] => [1, 1, 1]
    target = jnp.array([[3]], jnp.float32)
    if sample_dimension:
      target = jnp.expand_dims(target, axis=0)
    temperature = 0.1
    # pylint: disable=g-long-lambda
    def mean_weights_fn(target_, temperature_):
      temperature_constraint = mpo_ops.LagrangePenalty(
          temperature_, _EPSILON_BOUND)
      _, weights, _ = e_step_fn(
          target_, temperature_constraint=temperature_constraint,
          projection_operator=_PROJECTION_OPERATOR,
          **additional_inputs)
      return jnp.mean(weights)
    grad = jax.grad(mean_weights_fn, argnums=(0, 1))(target, temperature)
    np.testing.assert_almost_equal(
        np.array(grad[0]), np.zeros_like(grad[0]), decimal=4)
    self.assertAlmostEqual(grad[1], 0., places=4)

  def test_kl_constraint_loss_gradients(self):
    """Tests the gradients in the `_kl_constraint_loss` method."""
    kl = jnp.array(1., jnp.float32)
    alpha = jnp.array(1., jnp.float32)
    _, _, alpha = mpo_ops.kl_constraint_loss(kl, mpo_ops.LagrangePenalty(
        alpha=alpha, epsilon=_EPSILON_MEAN_BOUND, per_dimension=False),
                                             _PROJECTION_OPERATOR)

    def alpha_loss_fn(alpha_):
      penalty = mpo_ops.LagrangePenalty(
          alpha=alpha_, epsilon=_EPSILON_MEAN_BOUND, per_dimension=False)
      _, alpha_loss, _ = mpo_ops.kl_constraint_loss(
          kl, penalty, _PROJECTION_OPERATOR)
      return alpha_loss
    alpha_gradients = jax.grad(alpha_loss_fn)(alpha)
    actual_alpha_gradients = _EPSILON_MEAN_BOUND - kl

    def kl_loss_fn(kl_):
      penalty = mpo_ops.LagrangePenalty(
          alpha=alpha, epsilon=_EPSILON_MEAN_BOUND, per_dimension=False)
      kl_loss, _, _ = mpo_ops.kl_constraint_loss(
          kl_, penalty, _PROJECTION_OPERATOR)
      return kl_loss
    kl_gradients = jax.grad(kl_loss_fn)(kl)
    actual_kl_gradients = alpha

    self.assertAlmostEqual(kl_gradients, actual_kl_gradients)
    self.assertAlmostEqual(alpha_gradients, actual_alpha_gradients)

  def test_kl_constraint_loss_stop_gradients(self):
    """Tests the stop gradients in the `kl_constraint_loss` function.

      The `alpha_loss` term should not affect the KL and the `kl` term should
      not affect `alpha`.
    """
    kl = jnp.array(1., jnp.float32)
    alpha = jnp.array(1., jnp.float32)
    _, _, alpha = mpo_ops.kl_constraint_loss(kl, mpo_ops.LagrangePenalty(
        alpha=alpha, epsilon=_EPSILON_MEAN_BOUND, per_dimension=False),
                                             _PROJECTION_OPERATOR)

    def kl_loss_fn(alpha_):
      penalty = mpo_ops.LagrangePenalty(
          alpha=alpha_, epsilon=_EPSILON_MEAN_BOUND, per_dimension=False)
      kl_loss, _, _ = mpo_ops.kl_constraint_loss(
          kl, penalty, _PROJECTION_OPERATOR)
      return kl_loss

    kl_gradients = jax.grad(kl_loss_fn)(alpha)

    def alpha_loss_fn(kl_):
      penalty = mpo_ops.LagrangePenalty(
          alpha=alpha, epsilon=_EPSILON_MEAN_BOUND, per_dimension=False)
      _, alpha_loss, _ = mpo_ops.kl_constraint_loss(
          kl_, penalty, _PROJECTION_OPERATOR)
      return alpha_loss
    alpha_gradients = jax.grad(alpha_loss_fn)(kl)

    # Test that there are no gradients of KL w.r.t alpha
    self.assertEqual(kl_gradients, 0.)

    # Test that there are no gradients of alpha w.r.t kl
    self.assertEqual(alpha_gradients, 0.)

  @parameterized.parameters(
      # With restarting weights of 1 (and temperature of 1) the weights should
      # be e^-1, 1, max advantage is 2 and num samples is 2 so temperature loss
      # is log(1 + e^-1) + 2 - log(2) + temperature epsilon
      {'advantages': np.array([[1.0, 2.0]]),
       'restarting_weights': np.array([[1.0, 1.0]]),
       'expected_temperature_loss': (math.log(1.0 + math.exp(-1.0)) + 2.0 -
                                     math.log(2.0) + _EPSILON_BOUND)},
      # With the second restarting weight set to 0 the weights become 1, 0
      # max advantage is 1 and num samples is 1 so temperature loss is
      # log(1) + 1 - log(1) + temperature epsilon
      {'advantages': np.array([[1.0, 2.0]]),
       'restarting_weights': np.array([[1.0, 0.0]]),
       'expected_temperature_loss': 1.0 + _EPSILON_BOUND},
  )
  def test_restarting_weights(
      self, advantages, restarting_weights, expected_temperature_loss):
    """Test that calculation is correct if restarting weight is set to 0."""
    temperature_loss, _, _ = mpo_ops.vmpo_compute_weights_and_temperature_loss(
        advantages, restarting_weights, np.ones_like(restarting_weights),
        mpo_ops.LagrangePenalty(1.0, _EPSILON_BOUND),
        functools.partial(np.clip, a_min=1e-8, a_max=None), 1.0)
    self.assertAlmostEqual(
        temperature_loss, expected_temperature_loss, places=4)

  @parameterized.parameters(
      # When the top k fraction is 1.0 all of the weights should be 1
      {'top_k_fraction': 1.0,
       'scaled_advantages': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
       'expected_top_k_weights': np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])},
      # When the top k fraction is 0.5 it will take the bottom row as these are
      # the highest.
      {'top_k_fraction': 0.5,
       'scaled_advantages': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
       'expected_top_k_weights': np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])}
  )
  def test_top_k_fraction(
      self, top_k_fraction, scaled_advantages, expected_top_k_weights):
    """Test that only the top k fraction are used."""
    top_k_weights = mpo_ops.get_top_k_weights(
        top_k_fraction, jnp.ones_like(scaled_advantages), scaled_advantages)
    np.testing.assert_allclose(top_k_weights, expected_top_k_weights)

  def test_top_k_fraction_too_low(self):
    """Test if the top k fraction returns 0 advantages we raise an error."""
    with self.assertRaises(ValueError):
      mpo_ops.get_top_k_weights(0.01, jnp.ones((3, 2)), jnp.ones((3, 2)))

  @parameterized.parameters(
      # With importance weights of 1 (and temperature of 1) the weights should
      # be e^-1, 1, max advantage is 2 and num samples is 2 so temperature loss
      # is log(1 + e^-1) + 2 - log(2) + temperature epsilon
      {'advantages': np.array([[1.0, 2.0]]),
       'importance_weights': np.array([[1.0, 1.0]]),
       'expected_temperature_loss': (math.log(1.0 + math.exp(-1.0)) + 2.0 -
                                     math.log(2.0) + _EPSILON_BOUND)},
      # If the second importance weight is 0.5 temperature loss becomes
      # log(0.5 + e^-1) + 2 - log(2) + temperature epsilon
      {'advantages': np.array([[1.0, 2.0]]),
       'importance_weights': np.array([[1.0, 0.5]]),
       'expected_temperature_loss': (math.log(0.5 + math.exp(-1.0)) + 2.0 -
                                     math.log(2.0) + _EPSILON_BOUND)},
  )
  def test_importance_weights(
      self, advantages, importance_weights, expected_temperature_loss):
    """Test that importance weights have the correct effect."""
    temperature_loss, _, _ = mpo_ops.vmpo_compute_weights_and_temperature_loss(
        advantages, np.ones_like(importance_weights), importance_weights,
        mpo_ops.LagrangePenalty(1.0, _EPSILON_BOUND),
        functools.partial(np.clip, a_min=1e-8, a_max=None), 1.0)
    self.assertAlmostEqual(
        temperature_loss, expected_temperature_loss, places=4)

  @parameterized.parameters({'per_dimension': True}, {'per_dimension': False})
  def test_mpo_input_axis_order_equivalence(self, per_dimension):
    """Test loss functions are equivalent regardless of axis order."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    key, new_key = jax.random.split(key)
    params = _init_params(new_key)
    out, mpo_inputs = get_common_loss_fn_inputs(params, key, 'sample_q_values')
    kl_constraints = get_coupled_kl_constraints(out, params,
                                                per_dimension=per_dimension)
    mpo_inputs.update({'kl_constraints': kl_constraints})

    # Original loss fn inputs are [S T B],
    stb_loss, stb_outputs = mpo_ops.mpo_loss(**mpo_inputs)
    mean_stb_loss = jnp.mean(stb_loss)

    # Swap axes and try [S B T]
    mpo_inputs.update({
        'sample_log_probs': jnp.swapaxes(mpo_inputs['sample_log_probs'], 1, 2),
        'sample_q_values': jnp.swapaxes(mpo_inputs['sample_q_values'], 1, 2),
        'kl_constraints': [(jnp.swapaxes(kl, 0, 1), mpo_ops.LagrangePenalty(
            alpha=jnp.swapaxes(pen.alpha, 0, 1), epsilon=pen.epsilon,
            per_dimension=pen.per_dimension)) for (kl, pen) in kl_constraints],
    })
    sbt_loss, sbt_outputs = mpo_ops.mpo_loss(**mpo_inputs)
    mean_sbt_loss = jnp.mean(sbt_loss)

    # Try [T B S] denoting sample_axis at 2 instead of 0.
    mpo_inputs.update({
        'sample_log_probs': jnp.swapaxes(mpo_inputs['sample_log_probs'], 0, 2),
        'sample_q_values': jnp.swapaxes(mpo_inputs['sample_q_values'], 0, 2),
        'kl_constraints': kl_constraints,  # T B
        'sample_axis': 2
    })
    tbs_loss, tbs_outputs = mpo_ops.mpo_loss(**mpo_inputs)
    mean_tbs_loss = jnp.mean(tbs_loss)

    self.assertAlmostEqual(mean_stb_loss, mean_sbt_loss, places=4)
    self.assertAlmostEqual(mean_tbs_loss, mean_sbt_loss, places=4)
    self.assertEqual(tbs_outputs.num_samples, sbt_outputs.num_samples)
    self.assertEqual(tbs_outputs.num_samples, stb_outputs.num_samples)

  @parameterized.parameters({'per_dimension': True}, {'per_dimension': False})
  def test_vmpo_input_axis_order_equivalence(self, per_dimension):
    """Test loss functions are equivalent regardless of axis order."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    key, new_key = jax.random.split(key)
    params = _init_params(new_key)
    out, vmpo_inputs = get_common_loss_fn_inputs(params, key, 'advantages')
    kl_constraints = get_coupled_kl_constraints(out, params,
                                                per_dimension=per_dimension)
    vmpo_inputs.update({'kl_constraints': kl_constraints})

    # Original loss fn inputs are [T B],
    tb_loss, tb_outputs = mpo_ops.vmpo_loss(**vmpo_inputs)
    mean_tb_loss = jnp.mean(tb_loss)

    # Swap axes and try [B T]
    vmpo_inputs.update({
        'sample_log_probs': jnp.swapaxes(vmpo_inputs['sample_log_probs'], 0, 1),
        'advantages': jnp.swapaxes(vmpo_inputs['advantages'], 0, 1),
        'kl_constraints': [(jnp.swapaxes(kl, 0, 1), mpo_ops.LagrangePenalty(
            alpha=jnp.swapaxes(pen.alpha, 0, 1), epsilon=pen.epsilon,
            per_dimension=pen.per_dimension)) for (kl, pen) in kl_constraints],
    })
    bt_loss, bt_outputs = mpo_ops.vmpo_loss(**vmpo_inputs)
    mean_bt_loss = jnp.mean(bt_loss)

    self.assertAlmostEqual(mean_tb_loss, mean_bt_loss, places=4)
    self.assertEqual(tb_outputs.num_samples, bt_outputs.num_samples)


if __name__ == '__main__':
  absltest.main()
