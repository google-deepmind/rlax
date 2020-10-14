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
"""Tests for pop_art.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import chex
import haiku as hk
from haiku import data_structures
from haiku import initializers
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
from rlax._src import pop_art

_INPUT_DIM = 3


def setUpModule():
  chex.set_n_cpu_devices(n=4)
  chex.assert_devices_available(n=4, devtype='cpu', backend='cpu')


def get_constant_linear_params(num_outputs):
  w = np.ones([_INPUT_DIM, num_outputs])
  b = np.zeros([num_outputs])
  return dict(w=w, b=b)


def get_fake_pop_art_state(num_outputs):
  """Returns a fake PopArtState."""
  shift = np.arange(num_outputs).astype(np.float32) + 1
  scale = shift
  second_moment = np.square(scale) + np.square(shift)
  return pop_art.PopArtState(shift, scale, second_moment)


class PopArtTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, np.array([[[0., 0., 1.],
                        [0., 5., 0.]],
                       [[9., 0., 0.],
                        [0., 0., 0.]]])),
      ([], np.array([[[0., 0., 1.],
                      [0., 5., 0.]],
                     [[9., 0., 0.],
                      [0., 0., 0.]]])),
      (['i'], np.array([[[0., 5., 1.],
                         [0., 5., 1.]],
                        [[9., 0., 0.],
                         [9., 0., 0.]]])),
      (['j'], np.array([[[9., 0., 1.],
                         [0., 5., 0.]],
                        [[9., 0., 1.],
                         [0., 5., 0.]]])),
      (['i', 'j'], np.array([[[9., 5., 1.],
                              [9., 5., 1.]],
                             [[9., 5., 1.],
                              [9., 5., 1.]]])))
  def test_cross_replica_scatter_add(self, axes, expected):
    shape = (2, 2, 3)
    source = np.zeros(shape)
    fn = functools.partial(pop_art._cross_replica_scatter_add, axis_name=axes)
    mapped_fn = jax.pmap(fn, axis_name='i', backend='cpu')
    mapped_fn = jax.pmap(mapped_fn, axis_name='j', backend='cpu')

    updates = np.array([[[1., 0., 0.],
                         [0., 5., 0.]],
                        [[0., 0., 9.],
                         [0., 0., 0.]]])
    indices = np.array([[[2, 2, 2],
                         [1, 1, 1]],
                        [[0, 0, 0],
                         [0, 1, 2]]])
    np.testing.assert_equal(mapped_fn(source, indices, updates), expected)

  def test_normalize_unnormalize_is_identity(self):
    num_outputs = 3
    state = get_fake_pop_art_state(num_outputs)
    np.random.seed(42)
    # Draws random numbers with the same magnitude as the pop art shift and
    # scales to ensure numerical stability.
    values = np.random.randint(-10, 10, size=(100,)).astype(np.float32)
    indices = np.random.randint(0, num_outputs, size=(100,))
    np.testing.assert_allclose(
        values,
        pop_art.normalize(state, pop_art.unnormalize(state, values, indices),
                          indices))
    np.testing.assert_allclose(
        values,
        pop_art.unnormalize(state, pop_art.normalize(state, values, indices),
                            indices))

  def test_unnormalize_linear(self):
    num_outputs = 3
    state = get_fake_pop_art_state(num_outputs)

    # Verify that it denormalizes as planned.
    inputs = np.ones([num_outputs, num_outputs])
    indices = np.arange(num_outputs)[::-1]
    out = pop_art.unnormalize_linear(state, inputs, indices)

    expected_normalized = np.asarray([1, 1, 1])
    expected_unnormalized = np.asarray([6, 4, 2])
    np.testing.assert_allclose(out.normalized, expected_normalized)
    np.testing.assert_allclose(out.unnormalized, expected_unnormalized)

  def test_learn_scale_shift(self):
    num_outputs = 2
    initial_state, update = pop_art.popart(
        num_outputs, step_size=1e-1, scale_lb=1e-6, scale_ub=1e6)
    state = initial_state()
    params = get_constant_linear_params(num_outputs)
    targets = np.arange(6) - 3
    indices = np.asarray([0, 0, 0, 1, 1, 1])
    # Learn the parameters.
    for _ in range(10):
      _, state = update(params, state, targets, indices)

    expected_scale = np.std(targets[:3])
    expected_scale = np.asarray([expected_scale, expected_scale])
    expected_shift = np.asarray([-2., 1.])
    # Loose tolerances; just get close.
    np.testing.assert_allclose(
        state.scale, expected_scale, atol=1e-1, rtol=1e-1)
    np.testing.assert_allclose(
        state.shift, expected_shift, atol=1e-1, rtol=1e-1)

  def test_slow_update(self):
    num_outputs = 2
    # Two step sizes: 0.1, and 0.8
    kwargs = dict(
        num_outputs=num_outputs,
        scale_lb=1e-6,
        scale_ub=1e6,
    )
    initial_state, slow_update = pop_art.popart(step_size=1e-2, **kwargs)
    _, fast_update = pop_art.popart(step_size=1e-1, **kwargs)
    state = initial_state()
    params = get_constant_linear_params(num_outputs)
    targets = np.arange(6) * 3  # standard deviation > 1 and mean > 0
    indices = np.asarray([0, 0, 0, 1, 1, 1])
    _, slow_state = slow_update(params, state, targets, indices)
    _, fast_state = fast_update(params, state, targets, indices)

    # Faster step size means faster adjustment.
    np.testing.assert_array_less(slow_state.shift, fast_state.shift)
    np.testing.assert_array_less(slow_state.scale, fast_state.scale)

  def test_scale_bounded(self):
    num_outputs = 1
    # Set scale_lb and scale_ub to 1 and verify this is obeyed.
    initial_state, update = pop_art.popart(
        num_outputs, step_size=1e-1, scale_lb=1., scale_ub=1.)
    state = initial_state()
    params = get_constant_linear_params(num_outputs)
    targets = np.ones((4, 2))
    indices = np.zeros((4, 2), dtype=np.int32)
    for _ in range(4):
      _, state = update(params, state, targets, indices)
      self.assertAlmostEqual(float(state.scale[0]), 1.)

  def test_outputs_preserved(self):
    num_outputs = 2
    initial_state, update = pop_art.popart(
        num_outputs, step_size=1e-3, scale_lb=1e-6, scale_ub=1e6)
    state = initial_state()
    key = jax.random.PRNGKey(428)

    def net(x):
      linear = hk.Linear(
          num_outputs, b_init=initializers.RandomUniform(), name='head')
      return linear(x)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(net))
    key, subkey1, subkey2 = jax.random.split(key, 3)
    fixed_data = jax.random.uniform(subkey1, (4, 3))
    params = init_fn(subkey2, fixed_data)
    initial_result = apply_fn(params, fixed_data)
    indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    # Repeatedly update state and verify that params still preserve outputs.
    for _ in range(30):
      key, subkey1, subkey2 = jax.random.split(key, 3)
      targets = jax.random.uniform(subkey1, (8,))
      linear_params, state = update(params['head'], state, targets, indices)
      params = data_structures.to_mutable_dict(params)
      params['head'] = linear_params

      # Apply updated linear transformation and unnormalize outputs.
      transform = apply_fn(params, fixed_data)
      out = jnp.broadcast_to(state.scale,
                             transform.shape) * transform + jnp.broadcast_to(
                                 state.shift, transform.shape)
      np.testing.assert_allclose(initial_result, out, atol=1e-2)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
