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
"""Tests for `value_learning.py`."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import distributions
from rlax._src import value_learning


class TDLearningTest(parameterized.TestCase):

  def setUp(self):
    super(TDLearningTest, self).setUp()

    self.v_tm1 = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    self.r_t = np.array(
        [-1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
    self.discount_t = np.array(
        [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], dtype=np.float32)
    self.v_t = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.float32)

    self.expected_td = np.array(
        [-2., -2., -2., -2., -1.5, -1., -2., -1., 0.], dtype=np.float32)

  @chex.all_variants()
  def test_td_learning_batch(self):
    """Tests for a full batch."""
    td_learning = self.variant(jax.vmap(value_learning.td_learning))
    # Compute errors in batch.
    actual_td = td_learning(self.v_tm1, self.r_t, self.discount_t, self.v_t)
    # Tets output.
    np.testing.assert_allclose(self.expected_td, actual_td)


class TDLambdaTest(parameterized.TestCase):

  def setUp(self):
    super(TDLambdaTest, self).setUp()
    self.lambda_ = 0.75

    self.v_tm1 = np.array(
        [[1.1, -1.1, 3.1], [2.1, -1.1, -2.1]], dtype=np.float32)
    self.discount_t = np.array(
        [[0., 0.89, 0.85], [0.88, 1., 0.83]], dtype=np.float32)
    self.r_t = np.array(
        [[-1.3, -1.3, 2.3], [1.3, 5.3, -3.3]], dtype=np.float32)
    self.bootstrap_v = np.array([2.2, -1.2], np.float32)

    self.expected = np.array(
        [[-2.4, 3.2732253, 1.0700002],
         [-0.01701999, 2.6529999, -2.196]],
        dtype=np.float32)

  @chex.all_variants()
  def test_batch_compatibility(self):
    """Tests for a full batch."""
    td_lambda = self.variant(jax.vmap(functools.partial(
        value_learning.td_lambda, lambda_=self.lambda_)))
    # Get arguments.
    v_t = np.concatenate([self.v_tm1[:, 1:], self.bootstrap_v[:, None]], axis=1)
    # Test output
    actual = td_lambda(self.v_tm1, self.r_t, self.discount_t, v_t)
    np.testing.assert_allclose(self.expected, actual, rtol=1e-4)


class SarsaTest(parameterized.TestCase):

  def setUp(self):
    super(SarsaTest, self).setUp()

    self.q_tm1 = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float32)
    self.a_tm1 = np.array([0, 1], dtype=np.int32)
    self.r_t = np.array([1, 1], dtype=np.float32)
    self.discount_t = np.array([0, 1], dtype=np.float32)
    self.q_t = np.array([[0, 1, 0], [3, 2, 0]], dtype=np.float32)
    self.a_t = np.array([1, 0], dtype=np.int32)

    self.expected = np.array([0., 3.], dtype=np.float32)

  @chex.all_variants()
  def test_sarsa_batch(self):
    """Tests for a full batch."""
    batch_sarsa = self.variant(jax.vmap(value_learning.sarsa))
    # Test outputs.
    actual = batch_sarsa(self.q_tm1, self.a_tm1, self.r_t, self.discount_t,
                         self.q_t, self.a_t)
    np.testing.assert_allclose(self.expected, actual)


class ExpectedSarsaTest(parameterized.TestCase):

  def setUp(self):
    super(ExpectedSarsaTest, self).setUp()

    self.q_tm1 = np.array(
        [[1, 1, 0.5], [1, 1, 3]], dtype=np.float32)
    self.a_tm1 = np.array(
        [0, 1], dtype=np.int32)
    self.r_t = np.array(
        [4, 1], dtype=np.float32)
    self.discount_t = np.array(
        [1, 1], dtype=np.float32)
    self.q_t = np.array(
        [[1.5, 1, 2], [3, 2, 1]], dtype=np.float32)
    self.probs_a_t = np.array(
        [[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]], dtype=np.float32)

    self.expected = np.array(
        [4.4, 2.], dtype=np.float32)

  @chex.all_variants()
  def test_expected_sarsa_batch(self):
    """Tests for a full batch."""
    expected_sarsa = self.variant(jax.vmap(value_learning.expected_sarsa))
    # Test outputs.
    actual = expected_sarsa(self.q_tm1, self.a_tm1, self.r_t, self.discount_t,
                            self.q_t, self.probs_a_t)
    np.testing.assert_allclose(self.expected, actual)


class SarsaLambdaTest(parameterized.TestCase):

  def setUp(self):
    super(SarsaLambdaTest, self).setUp()
    self.lambda_ = 0.75

    self.q_tm1 = np.array(
        [[[1.1, 2.1], [-1.1, 1.1], [3.1, -3.1]],
         [[2.1, 3.1], [-1.1, 0.1], [-2.1, -1.1]]],
        dtype=np.float32)
    self.a_tm1 = np.array(
        [[0, 1, 0],
         [1, 0, 0]],
        dtype=np.int32)
    self.discount_t = np.array(
        [[0., 0.89, 0.85],
         [0.88, 1., 0.83]],
        dtype=np.float32)
    self.r_t = np.array(
        [[-1.3, -1.3, 2.3],
         [1.3, 5.3, -3.3]],
        dtype=np.float32)
    self.q_t = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2]]],
        dtype=np.float32)
    self.a_t = np.array(
        [[1, 0, 1],
         [1, 1, 0]],
        dtype=np.int32)

    self.expected = np.array(
        [[-2.4, -1.8126001, -1.8200002], [0.25347996, 3.4780002, -2.196]],
        dtype=np.float32)

  @chex.all_variants()
  def test_sarsa_lambda_batch(self):
    """Tests for a full batch."""
    sarsa_lambda = self.variant(jax.vmap(functools.partial(
        value_learning.sarsa_lambda, lambda_=self.lambda_)))
    # Test outputs.
    actual = sarsa_lambda(self.q_tm1, self.a_tm1, self.r_t, self.discount_t,
                          self.q_t, self.a_t)
    np.testing.assert_allclose(self.expected, actual, rtol=1e-4)


class QLearningTest(parameterized.TestCase):

  def setUp(self):
    super(QLearningTest, self).setUp()

    self.q_tm1 = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    self.a_tm1 = np.array([0, 1], dtype=np.int32)
    self.r_t = np.array([1, 1], dtype=np.float32)
    self.discount_t = np.array([0, 1], dtype=np.float32)
    self.q_t = np.array([[0, 1, 0], [1, 2, 0]], dtype=np.float32)

    self.expected = np.array([0., 1.], dtype=np.float32)

  @chex.all_variants()
  def test_q_learning_batch(self):
    """Tests for a full batch."""
    q_learning = self.variant(jax.vmap(value_learning.q_learning))
    # Test outputs.
    actual = q_learning(self.q_tm1, self.a_tm1, self.r_t, self.discount_t,
                        self.q_t)
    np.testing.assert_allclose(self.expected, actual)


class DoubleQLearningTest(parameterized.TestCase):

  def setUp(self):
    super(DoubleQLearningTest, self).setUp()

    self.q_tm1 = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    self.a_tm1 = np.array([0, 1], dtype=np.int32)
    self.r_t = np.array([1, 1], dtype=np.float32)
    self.discount_t = np.array([0, 1], dtype=np.float32)
    self.q_t_value = np.array([[99, 1, 98], [91, 2, 66]], dtype=np.float32)
    self.q_t_selector = np.array([[2, 10, 1], [11, 20, 1]], dtype=np.float32)

    self.expected = np.array([0., 1.], dtype=np.float32)

  @chex.all_variants()
  def test_double_q_learning_batch(self):
    """Tests for a full batch."""
    double_q_learning = self.variant(jax.vmap(value_learning.double_q_learning))
    # Test outputs.
    actual = double_q_learning(self.q_tm1, self.a_tm1, self.r_t,
                               self.discount_t, self.q_t_value,
                               self.q_t_selector)
    np.testing.assert_allclose(self.expected, actual)


class PersistentQLearningTest(parameterized.TestCase):

  def setUp(self):
    super(PersistentQLearningTest, self).setUp()
    self.action_gap_scale = 0.25

    self.q_tm1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    self.a_tm1 = np.array([0, 1, 1], dtype=np.int32)
    self.r_t = np.array([3, 2, 7], dtype=np.float32)
    self.discount_t = np.array([0, 1, 0.5], dtype=np.float32)
    self.q_t = np.array([[11, 12], [20, 16], [-8, -4]], dtype=np.float32)

    self.expected = np.array([2., 17., -1.], dtype=np.float32)

  @chex.all_variants()
  def test_persistent_q_learning_batch(self):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    persistent_q_learning = self.variant(jax.vmap(functools.partial(
        value_learning.persistent_q_learning,
        action_gap_scale=self.action_gap_scale)))
    # Test outputs.
    actual = persistent_q_learning(self.q_tm1, self.a_tm1, self.r_t,
                                   self.discount_t, self.q_t)
    np.testing.assert_allclose(self.expected, actual)


class QVLearningTest(parameterized.TestCase):

  def setUp(self):
    super(QVLearningTest, self).setUp()

    self.q_tm1 = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    self.a_tm1 = np.array([0, 1], dtype=np.int32)
    self.r_t = np.array([1, 1], dtype=np.float32)
    self.discount_t = np.array([0, 1], dtype=np.float32)
    self.v_t = np.array([1, 3], dtype=np.float32)

    self.expected = np.array([0., 2.], dtype=np.float32)

  @chex.all_variants()
  def test_qv_learning_batch(self):
    """Tests for a full batch."""
    batch_qv_learning = self.variant(jax.vmap(value_learning.qv_learning))
    # Test outputs.
    actual = batch_qv_learning(self.q_tm1, self.a_tm1, self.r_t,
                               self.discount_t, self.v_t)
    np.testing.assert_allclose(self.expected, actual)


class QVMaxTest(parameterized.TestCase):

  def setUp(self):
    super(QVMaxTest, self).setUp()

    self.v_tm1 = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        dtype=np.float32)
    self.r_t = np.array(
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        dtype=np.float32)
    self.discount_t = np.array(
        [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1],
        dtype=np.float32)
    self.q_t = np.array(
        [[0, -1], [-2, 0], [0, -3], [1, 0], [1, 1], [0, 1],
         [1, 2], [2, -2], [2, 2]],
        dtype=np.float32)

    self.expected = np.array(
        [-2., -2., -2., -2., -1.5, -1., -2., -1., 0.],
        dtype=np.float32)

  @chex.all_variants()
  def test_qv_max_batch(self):
    """Tests for a full batch."""
    qv_max = self.variant(jax.vmap(value_learning.qv_max))
    # Test outputs.
    actual = qv_max(self.v_tm1, self.r_t, self.discount_t, self.q_t)
    np.testing.assert_allclose(self.expected, actual)


class QLambdaTest(parameterized.TestCase):

  def setUp(self):
    super(QLambdaTest, self).setUp()
    self.lambda_ = 0.75

    self.q_tm1 = np.array(
        [[[1.1, 2.1], [-1.1, 1.1], [3.1, -3.1]],
         [[2.1, 3.1], [-1.1, 0.1], [-2.1, -1.1]]],
        dtype=np.float32)
    self.a_tm1 = np.array(
        [[0, 1, 0],
         [1, 0, 0]],
        dtype=np.int32)
    self.discount_t = np.array(
        [[0., 0.89, 0.85],
         [0.88, 1., 0.83]],
        dtype=np.float32)
    self.r_t = np.array(
        [[-1.3, -1.3, 2.3],
         [1.3, 5.3, -3.3]],
        dtype=np.float32)
    self.q_t = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2]]],
        dtype=np.float32)

    self.expected = np.array(
        [[-2.4, 0.427975, 1.07],
         [0.69348, 3.478, -2.196]],
        dtype=np.float32)

  @chex.all_variants()
  def test_q_lambda_batch(self):
    """Tests for a full batch."""
    q_lambda = self.variant(jax.vmap(functools.partial(
        value_learning.q_lambda, lambda_=self.lambda_)))
    # Test outputs.
    actual = q_lambda(self.q_tm1, self.a_tm1, self.r_t, self.discount_t,
                      self.q_t)
    np.testing.assert_allclose(self.expected, actual, rtol=1e-5)


class RetraceTest(parameterized.TestCase):

  def setUp(self):
    super(RetraceTest, self).setUp()
    self._lambda = 0.9

    self._qs = np.array(
        [[[1.1, 2.1], [-1.1, 1.1], [3.1, -3.1], [-1.2, 0.0]],
         [[2.1, 3.1], [9.5, 0.1], [-2.1, -1.1], [0.1, 7.4]]],
        dtype=np.float32)
    self._targnet_qs = np.array(
        [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2], [-2.25, -6.0]],
         [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2], [1.5, 1.0]]],
        dtype=np.float32)
    self._actions = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 1]],
        dtype=np.int32)
    self._rewards = np.array(
        [[-1.3, -1.3, 2.3, 42.0],
         [1.3, 5.3, -3.3, -5.0]],
        dtype=np.float32)
    self._pcontinues = np.array(
        [[0., 0.89, 0.85, 0.99],
         [0.88, 1., 0.83, 0.95]],
        dtype=np.float32)
    self._target_policy_probs = np.array(
        [[[0.5, 0.5], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]],
         [[0.1, 0.9], [1.0, 0.0], [0.3, 0.7], [0.7, 0.3]]],
        dtype=np.float32)
    self._behavior_policy_probs = np.array(
        [[0.5, 0.1, 0.9, 0.3], [0.4, 0.6, 1.0, 0.9]],
        dtype=np.float32)
    self._inputs = [
        self._qs, self._targnet_qs, self._actions,
        self._rewards, self._pcontinues,
        self._target_policy_probs, self._behavior_policy_probs]

    self.expected = np.array(
        [[2.8800001, 3.8934109, 4.5942383],
         [3.1121615e-1, 2.0253206e1, 3.1601219e-3]],
        dtype=np.float32)

  @chex.all_variants()
  def test_retrace_batch(self):
    """Tests for a full batch."""
    retrace = self.variant(jax.vmap(functools.partial(
        value_learning.retrace, lambda_=self._lambda)))
    # Test outputs.
    actual_td = retrace(self._qs[:, :-1], self._targnet_qs[:, 1:],
                        self._actions[:, :-1], self._actions[:, 1:],
                        self._rewards[:, :-1], self._pcontinues[:, :-1],
                        self._target_policy_probs[:, 1:],
                        self._behavior_policy_probs[:, 1:])
    actual_loss = 0.5 * np.square(actual_td)
    np.testing.assert_allclose(self.expected, actual_loss, rtol=1e-5)


def _generate_sorted_support(size):
  """Generate a random support vector."""
  support = np.random.normal(-1.0, 1.0, size=size).astype(np.float32)
  return np.sort(support, axis=-1)


def _generate_weights(size):
  """Generates a weight distribution where half of entries are zero."""
  normal = np.random.normal(-1.0, 1.0, size=size).astype(np.float32)
  mask = (np.random.random(size=size) > 0.5).astype(np.float32)
  return normal * mask


class L2ProjectTest(parameterized.TestCase):

  def setUp(self):
    super(L2ProjectTest, self).setUp()

    old_supports = np.arange(-1, 1., 0.25)
    self.old_supports = np.stack([old_supports, old_supports + 1.])
    weights = self.old_supports.copy()
    weights[0, ::2] = 0.
    weights[1, 1::2] = 0.
    self.weights = weights
    new_supports = np.arange(-1, 1., 0.5)
    self.new_supports = np.stack([new_supports, new_supports + 1.])

    self.expected = np.array([[-0.375, -0.5, 0., 0.875], [0., 0.5, 1., 1.5]],
                             dtype=np.float32)

  @chex.all_variants()
  def test_categorical_l2_project_batch(self):
    """Testsfor a full batch."""
    l2_project = self.variant(jax.vmap(functools.partial(
        value_learning._categorical_l2_project)))
    # Compute projection in batch.
    actual = l2_project(self.old_supports, self.weights, self.new_supports)
    # Test outputs.
    np.testing.assert_allclose(actual, self.expected)


class CategoricalTDLearningTest(parameterized.TestCase):

  def setUp(self):
    super(CategoricalTDLearningTest, self).setUp()
    self.atoms = np.array([.5, 1., 1.5], dtype=np.float32)

    self.logits_tm1 = np.array(
        [[0, 9, 0], [9, 0, 9], [0, 9, 0], [9, 9, 0], [9, 0, 9]],
        dtype=np.float32)
    self.r_t = np.array(
        [0.5, 0., 0.5, 0.8, -0.1],
        dtype=np.float32)
    self.discount_t = np.array(
        [0.8, 1., 0.8, 0., 1.],
        dtype=np.float32)
    self.logits_t = np.array(
        [[0, 0, 9], [1, 1, 1], [0, 0, 9], [1, 1, 1], [0, 9, 9]],
        dtype=np.float32)

    self.expected = np.array(
        [8.998915, 3.6932087, 8.998915, 0.69320893, 5.1929307],
        dtype=np.float32)

  @chex.all_variants()
  def test_categorical_td_learning_batch(self):
    """Tests for a full batch."""
    # Not using vmap for atoms.
    def fn(v_logits_tm1, r_t, discount_t, v_logits_t):
      return value_learning.categorical_td_learning(
          v_atoms_tm1=self.atoms,
          v_logits_tm1=v_logits_tm1,
          r_t=r_t,
          discount_t=discount_t,
          v_atoms_t=self.atoms,
          v_logits_t=v_logits_t)
    categorical_td_learning = self.variant(jax.vmap(fn))
    # Test outputs.
    actual = categorical_td_learning(
        self.logits_tm1, self.r_t, self.discount_t, self.logits_t)
    np.testing.assert_allclose(self.expected, actual, rtol=1e-5)


class CategoricalQLearningTest(parameterized.TestCase):

  def setUp(self):
    super(CategoricalQLearningTest, self).setUp()
    self.atoms = np.array([.5, 1., 1.5], dtype=np.float32)

    self.q_logits_tm1 = np.array(
        [[[1, 1, 1], [0, 9, 9], [0, 9, 0], [0, 0, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
         [[1, 1, 1], [0, 9, 9], [0, 0, 0], [0, 9, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]]],
        dtype=np.float32)
    self.q_logits_t = np.array(
        [[[1, 1, 1], [9, 0, 9], [1, 0, 0], [0, 0, 9]],
         [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
         [[1, 1, 1], [9, 0, 9], [0, 0, 9], [1, 0, 0]],
         [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
         [[9, 9, 0], [9, 0, 0], [0, 9, 9], [9, -9, 0]]],
        dtype=np.float32)
    self.a_tm1 = np.array(
        [2, 1, 3, 0, 1],
        dtype=np.int32)
    self.r_t = np.array(
        [0.5, 0., 0.5, 0.8, -0.1],
        dtype=np.float32)
    self.discount_t = np.array(
        [0.8, 1., 0.8, 0., 1.],
        dtype=np.float32)
    self.inputs = (
        self.q_logits_tm1, self.a_tm1, self.r_t,
        self.discount_t, self.q_logits_t)

    self.expected = np.array(
        [8.998915, 3.6932087, 8.998915, 0.69320893, 5.1929307],
        dtype=np.float32)

  @chex.all_variants()
  def test_categorical_q_learning_batch(self):
    """Tests for a full batch."""
    # Not using vmap for atoms.
    def fn(q_logits_tm1, a_tm1, r_t, discount_t, q_logits_t):
      return value_learning.categorical_q_learning(
          q_atoms_tm1=self.atoms,
          q_logits_tm1=q_logits_tm1,
          a_tm1=a_tm1,
          r_t=r_t,
          discount_t=discount_t,
          q_atoms_t=self.atoms,
          q_logits_t=q_logits_t)
    categorical_q_learning = self.variant(jax.vmap(fn))
    # Test outputs.
    actual = categorical_q_learning(*self.inputs)
    np.testing.assert_allclose(self.expected, actual, rtol=1e-5)


class CategoricalDoubleQLearningTest(parameterized.TestCase):

  def setUp(self):
    super(CategoricalDoubleQLearningTest, self).setUp()
    self.atoms = np.array([.5, 1., 1.5], dtype=np.float32)

    self.q_logits_tm1 = np.array(
        [[[1, 1, 1], [0, 9, 9], [0, 9, 0], [0, 0, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
         [[1, 1, 1], [0, 9, 9], [0, 0, 0], [0, 9, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]]],
        dtype=np.float32)
    self.q_logits_t = np.array(
        [[[1, 1, 1], [9, 0, 9], [1, 0, 0], [0, 0, 9]],
         [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
         [[1, 1, 1], [9, 0, 9], [0, 0, 9], [1, 0, 0]],
         [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
         [[9, 9, 0], [9, 0, 0], [0, 9, 9], [9, -9, 0]]],
        dtype=np.float32)
    self.q_t_selector = np.array(
        [[1, 0, 0, 9], [9, 0, 1, 1], [1, 9, 1, 1], [0, 1, 0, 9], [1, 1, 1, 9]],
        dtype=np.float32)
    self.a_tm1 = np.array(
        [2, 1, 3, 0, 1],
        dtype=np.int32)
    self.r_t = np.array(
        [0.5, 0., 0.5, 0.8, -0.1],
        dtype=np.float32)
    self.discount_t = np.array(
        [0.8, 1., 0.8, 0., 1.],
        dtype=np.float32)
    self.inputs = (
        self.q_logits_tm1, self.a_tm1, self.r_t,
        self.discount_t, self.q_logits_t, self.q_t_selector)

    self.expected = np.array(
        [8.998915, 5.192931, 5.400247, 0.693209, 0.693431],
        dtype=np.float32)

  @chex.all_variants()
  def test_categorical_double_q_learning_batch(self):
    """Tests for a full batch."""
    # Not using vmap for atoms.
    def fn(q_logits_tm1, a_tm1, r_t, discount_t, q_logits_t, q_t_selector):
      return value_learning.categorical_double_q_learning(
          q_atoms_tm1=self.atoms,
          q_logits_tm1=q_logits_tm1,
          a_tm1=a_tm1,
          r_t=r_t,
          discount_t=discount_t,
          q_atoms_t=self.atoms,
          q_logits_t=q_logits_t,
          q_t_selector=q_t_selector)
    categorical_double_q_learning = self.variant(jax.vmap(fn))
    # Test outputs.
    actual = categorical_double_q_learning(*self.inputs)
    np.testing.assert_allclose(self.expected, actual, rtol=1e-5)

  @chex.all_variants()
  def test_single_double_q_learning_eq_batch(self):
    """Tests equivalence to categorical_q_learning when q_t_selector == q_t."""
    # Not using vmap for atoms.
    @self.variant
    @jax.vmap
    def batch_categorical_double_q_learning(
        q_logits_tm1, a_tm1, r_t, discount_t, q_logits_t, q_t_selector):
      return value_learning.categorical_double_q_learning(
          q_atoms_tm1=self.atoms,
          q_logits_tm1=q_logits_tm1,
          a_tm1=a_tm1,
          r_t=r_t,
          discount_t=discount_t,
          q_atoms_t=self.atoms,
          q_logits_t=q_logits_t,
          q_t_selector=q_t_selector)

    @self.variant
    @jax.vmap
    def batch_categorical_q_learning(
        q_logits_tm1, a_tm1, r_t, discount_t, q_logits_t):
      return value_learning.categorical_q_learning(
          q_atoms_tm1=self.atoms,
          q_logits_tm1=q_logits_tm1,
          a_tm1=a_tm1,
          r_t=r_t,
          discount_t=discount_t,
          q_atoms_t=self.atoms,
          q_logits_t=q_logits_t)
    # Double Q-learning estimate with q_t_selector=q_t
    distrib = distributions.softmax()
    # Add batch and time dimension to atoms.
    atoms = jnp.expand_dims(jnp.expand_dims(self.atoms, 0), 0)
    q_t_selector = jnp.sum(distrib.probs(self.q_logits_t) * atoms, axis=-1)
    actual = batch_categorical_double_q_learning(
        self.q_logits_tm1, self.a_tm1, self.r_t, self.discount_t,
        self.q_logits_t, q_t_selector)
    # Q-learning estimate.
    expected = batch_categorical_q_learning(
        self.q_logits_tm1, self.a_tm1, self.r_t, self.discount_t,
        self.q_logits_t)
    # Test equivalence.
    np.testing.assert_allclose(expected, actual)


class QuantileRegressionLossTest(parameterized.TestCase):

  def setUp(self):
    super(QuantileRegressionLossTest, self).setUp()
    self.dist_src = np.array([[-1., 3.], [-1., 3.]])
    self.tau_src = np.array([[0.2, 0.7], [0., 0.4]])
    self.dist_target = np.array([[-3., 4., 2.], [-3., 4., 2.]])

    # delta = [[ -2  5  3 ], [ -6  1 -1 ]]
    # Huber(2.)-delta = [[  2  8  4 ], [ 10 .5 .5 ]]
    #
    # First batch element:
    # |tau - Id_{d<0}| = [[ .8 .2 .2 ], [ .3 .7 .3 ]]
    # Loss = 1/3 sum( |delta| . |tau - Id_{d<0}| )  = 2.0
    # Huber(2.)-loss = 2.5
    #
    # Second batch element:
    # |tau - Id_{d<0}| = [[ 1. 0. 0. ], [ .6 .4 .6 ]]
    # Loss = 2.2
    # Huber(2.)-loss = 8.5 / 3
    self.expected_loss = {
        0.: np.array([2.0, 2.2]),
        2.: np.array([2.5, 8.5 / 3.])
    }

  @chex.all_variants()
  @parameterized.named_parameters(
      ('nohuber', 0.),
      ('huber', 2.))
  def test_quantile_regression_loss_batch(self, huber_param):
    """Tests for a full batch."""
    loss_fn = value_learning._quantile_regression_loss
    loss_fn = self.variant(jax.vmap(functools.partial(
        loss_fn, huber_param=huber_param)))
    # Compute quantile regression loss.
    actual = loss_fn(self.dist_src, self.tau_src, self.dist_target)
    # Test outputs in batch.
    np.testing.assert_allclose(actual, self.expected_loss[huber_param])


class QuantileLearningTest(parameterized.TestCase):

  def setUp(self):
    super(QuantileLearningTest, self).setUp()

    self.dist_q_tm1 = np.array(  # n_batch = 3, n_taus = 2, n_actions = 4
        [[[0, 1, -5, 6], [-1, 3, 0, -2]],
         [[-5, 9, -5, 6], [2, 3, 1, -4]],
         [[5, 1, -5, 7], [-1, 3, 0, -2]]],
        dtype=np.float32)
    self.tau_q_tm1 = np.array(
        [[0.2, 0.7],
         [0.1, 0.5],
         [0.3, 0.4]],
        dtype=np.float32)
    self.a_tm1 = np.array(
        [1, 2, 0],
        dtype=np.int32)
    self.r_t = np.array(
        [0.5, -1., 0.],
        dtype=np.float32)
    self.discount_t = np.array(
        [0.5, 0., 1],
        dtype=np.float32)
    self.dist_q_t = np.array(
        [[[0, 5, 2, 2], [0, -3, 2, 2]],
         [[-3, -1, 4, -3], [1, 3, 1, -4]],
         [[-2, 2, -5, -7], [1, 3, 2, -2]]],
        dtype=np.float32)
    self.dist_q_t_selector = np.array(
        [[[0, 7, 2, -2], [0, 4, 2, 2]],
         [[-3, -1, 4, 3], [1, 3, 1, 4]],
         [[-1, -2, -5, -6], [-1, -5, 2, -2]]],
        dtype=np.float32)

    # Scenario 1: these match the dist_q_t_selector above
    self.greedy_probs_a_t = np.array(
        [[0, 1, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 0, 0]],
        dtype=np.float32)

    dist_qa_tm1 = np.array(
        [[1, 3], [-5, 1], [5, -1]],
        dtype=np.float32)
    # dist_qa_tm1                                      [ 1,  3]
    #     (batch x n_tau)                          =   [-5,  1]
    #                                                  [ 5, -1]
    # dist_q_t_selector[mean]                          [ 0.0,  5.5,  2.0,  0.0]
    #     (batch x n_actions)                      =   [-1.0,  1.0,  2.5,  3.5]
    #                                                  [-1.0, -3.5, -1.5, -4.0]
    # a_t = argmax_a dist_q_t_selector                 [1]
    #     (batch)                                  =   [3]
    #                                                  [0]
    # dist_qa_t                                        [ 5, -3]
    #     (batch x n_taus)                         =   [-3, -4]
    #                                                  [-2,  1]
    # target = r + gamma * dist_qa_t                   [ 3, -1]
    #     (batch x n_taus)                         =   [-1, -1]
    #                                                  [-2,  1]
    dist_target = np.array(
        [[3, -1], [-1, -1], [-2, 1]],
        dtype=np.float32)

    # Use qr loss to compute expected results (itself tested explicitly in
    # distributions_test.py).
    self.expected = {}
    for huber_param in [0.0, 1.0]:
      self.expected[huber_param] = np.array(  # loop over batch
          [value_learning._quantile_regression_loss(dqa, tau, dt, huber_param)
           for (dqa, tau, dt) in zip(dist_qa_tm1, self.tau_q_tm1, dist_target)],
          dtype=np.float32)

    # Scenario 2:
    # bootstrap targets are nor an argmax, but averaging across actions
    self.uniform_probs_a_t = np.ones((3, 4), dtype=np.float32) / 4.
    #                                             [ 2.25, 0.25]
    # dist_qa_t                              =    [-0.75, 0.25]
    #                                             [-3.00, 1.00]
    uniform_dist_target = np.array(
        [[1.625, 0.625],
         [-1, -1],
         [-3, 1]],
        dtype=np.float32)

    self.uniform_expected = {}
    for huber_param in [0.0, 1.0]:
      self.uniform_expected[huber_param] = np.array(  # loop over batch
          [value_learning._quantile_regression_loss(dqa, tau, dt, huber_param)
           for (dqa, tau, dt) in zip(dist_qa_tm1, self.tau_q_tm1,
                                     uniform_dist_target)],
          dtype=np.float32)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('nohuber', 0.0),
      ('huber', 1.0))
  def test_quantile_q_learning_batch(self, huber_param):
    """Tests for a full batch."""
    quantile_q_learning = self.variant(jax.vmap(functools.partial(
        value_learning.quantile_q_learning, huber_param=huber_param)))
    # Test outputs.
    actual = quantile_q_learning(
        self.dist_q_tm1, self.tau_q_tm1, self.a_tm1, self.r_t, self.discount_t,
        self.dist_q_t_selector, self.dist_q_t)
    np.testing.assert_allclose(self.expected[huber_param], actual, rtol=1e-5)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('nohuber', 0.0),
      ('huber', 1.0))
  def test_quantile_expected_sarsa_batch_greedy(self, huber_param):
    """Tests for a full batch."""
    quantile_expected_sarsa = self.variant(jax.vmap(functools.partial(
        value_learning.quantile_expected_sarsa, huber_param=huber_param)))
    # Test outputs.
    actual = quantile_expected_sarsa(
        self.dist_q_tm1, self.tau_q_tm1, self.a_tm1, self.r_t, self.discount_t,
        self.dist_q_t, self.greedy_probs_a_t)
    np.testing.assert_allclose(self.expected[huber_param], actual, rtol=1e-5)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('nohuber', 0.0),
      ('huber', 1.0))
  def test_quantile_expected_sarsa_batch_uniform(self, huber_param):
    """Tests for a full batch."""
    quantile_expected_sarsa = self.variant(jax.vmap(functools.partial(
        value_learning.quantile_expected_sarsa, huber_param=huber_param)))
    # Test outputs.
    actual = quantile_expected_sarsa(
        self.dist_q_tm1, self.tau_q_tm1, self.a_tm1, self.r_t, self.discount_t,
        self.dist_q_t, self.uniform_probs_a_t)
    np.testing.assert_allclose(
        self.uniform_expected[huber_param], actual, rtol=1e-5)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
