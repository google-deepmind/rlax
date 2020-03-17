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
"""Unit tests for `distributions.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.tree_util import tree_map
import numpy as np
from rlax._src import distributions


class CategoricalSampleTest(parameterized.TestCase):

  def test_categorical_sample(self):
    key = np.array([1, 2], dtype=np.uint32)
    probs = np.array([0.2, 0.3, 0.5])
    sample = distributions._categorical_sample(key, probs)
    self.assertEqual(sample, 0)


class SoftmaxTest(parameterized.TestCase):

  def setUp(self):
    super(SoftmaxTest, self).setUp()

    self.logits = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    self.samples = np.array([0, 1], dtype=np.int32)

    self.expected_probs = np.array(  # softmax with temperature=10
        [[0.34425336, 0.34425336, 0.31149334],
         [0.332225, 0.3671654, 0.3006096]],
        dtype=np.float32)
    probs = np.array(  # softmax with temperature=1
        [[0.42231882, 0.42231882, 0.15536241],
         [0.24472848, 0.66524094, 0.09003057]],
        dtype=np.float32)
    logprobs = np.log(probs)
    self.expected_logprobs = np.array(
        [logprobs[0][self.samples[0]], logprobs[1][self.samples[1]]])
    self.expected_entropy = -np.sum(probs * logprobs, axis=-1)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_probs(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.softmax(temperature=10.)
    # Optionally compile.
    softmax = compile_fn(distrib.probs)
    # For each element in the batch.
    for logits, expected in zip(self.logits, self.expected_probs):
      # Optionally convert to device array.
      logits = place_fn(logits)
      # Test outputs.
      actual = softmax(logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_probs_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.softmax(temperature=10.)
    # Optionally compile.
    softmax = compile_fn(distrib.probs)
    # Optionally convert to device array.
    logits = place_fn(self.logits)
    # Test softmax output in batch.
    actual = softmax(logits)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_logprob(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.softmax()
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # For each element in the batch.
    for logits, samples, expected in zip(
        self.logits, self.samples, self.expected_logprobs):
      # Optionally convert to device array.
      logits, samples = tree_map(place_fn, (logits, samples))
      # Test output.
      actual = logprob_fn(samples, logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_logprob_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.softmax()
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # Optionally convert to device array.
    logits, samples = tree_map(place_fn, (self.logits, self.samples))
    # Test softmax output in batch.
    actual = logprob_fn(samples, logits)
    np.testing.assert_allclose(self.expected_logprobs, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_entropy(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.softmax()
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # For each element in the batch.
    for logits, expected in zip(self.logits, self.expected_entropy):
      # Optionally convert to device array.
      logits = place_fn(logits)
      # Test outputs.
      actual = entropy_fn(logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_entropy_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.softmax()
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # Optionally convert to device array.
    logits = place_fn(self.logits)
    # Test softmax output in batch.
    actual = entropy_fn(logits)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)


class EpsilonSoftmaxTest(parameterized.TestCase):

  def setUp(self):
    super(EpsilonSoftmaxTest, self).setUp()

    self.logits = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    self.samples = np.array([0, 1], dtype=np.int32)

    self.expected_probs = np.array(  # softmax with temperature=10
        [[0.34316134, 0.34316134, 0.3136773],
         [0.3323358, 0.36378217, 0.30388197]],
        dtype=np.float32)
    probs = np.array(  # softmax with temperature=10
        [[0.34316134, 0.34316134, 0.3136773],
         [0.3323358, 0.36378217, 0.30388197]],
        dtype=np.float32)
    probs = distributions._mix_with_uniform(probs, epsilon=0.1)
    logprobs = np.log(probs)
    self.expected_logprobs = np.array(
        [logprobs[0][self.samples[0]], logprobs[1][self.samples[1]]])
    self.expected_entropy = -np.sum(probs * logprobs, axis=-1)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_probs(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.epsilon_softmax(epsilon=0.1,
                                            temperature=10.)
    # Optionally compile.
    softmax = compile_fn(distrib.probs)
    # For each element in the batch.
    for logits, expected in zip(self.logits, self.expected_probs):
      # Optionally convert to device array.
      logits = place_fn(logits)
      # Test outputs.
      actual = softmax(logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_softmax_probs_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.epsilon_softmax(epsilon=0.1,
                                            temperature=10.)
    # Optionally compile.
    softmax = compile_fn(distrib.probs)
    # Optionally convert to device array.
    logits = place_fn(self.logits)
    # Test softmax output in batch.
    actual = softmax(logits)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_safe_epsilon_softmax_equivalence(self, compile_fn, place_fn):
    distrib = distributions.safe_epsilon_softmax(epsilon=0.1,
                                                 temperature=10.)
    # Optionally compile.
    softmax = compile_fn(distrib.probs)
    # Optionally convert to device array.
    logits = place_fn(self.logits)
    # Test softmax output in batch.
    actual = softmax(logits)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)


class GreedyTest(parameterized.TestCase):

  def setUp(self):
    super(GreedyTest, self).setUp()

    self.preferences = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    self.samples = np.array([0, 1], dtype=np.int32)

    self.expected_probs = np.array(
        [[0.5, 0.5, 0.], [0., 1., 0.]], dtype=np.float32)
    self.expected_logprob = np.array(
        [-0.6931472, 0.], dtype=np.float32)
    self.expected_entropy = np.array(
        [0.6931472, 0.], dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_probs(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.greedy()
    # Optionally compile.
    greedy = compile_fn(distrib.probs)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_probs):
      # Optionally convert to device array.
      preferences = place_fn(preferences)
      # Test outputs.
      actual = greedy(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_probs_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.greedy()
    # Optionally compile.
    greedy = compile_fn(distrib.probs)
    # Optionally convert to device array.
    preferences = place_fn(self.preferences)
    # Test greedy output in batch.
    actual = greedy(preferences)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_logprob(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.greedy()
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # For each element in the batch.
    for preferences, samples, expected in zip(
        self.preferences, self.samples, self.expected_logprob):
      # Optionally convert to device array.
      preferences, samples = tree_map(place_fn, (preferences, samples))
      # Test output.
      actual = logprob_fn(samples, preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_logprob_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.greedy()
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # Optionally convert to device array.
    preferences, samples = tree_map(place_fn, (self.preferences, self.samples))
    # Test greedy output in batch.
    actual = logprob_fn(samples, preferences)
    np.testing.assert_allclose(self.expected_logprob, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_entropy(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.greedy()
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_entropy):
      # Optionally convert to device array.
      preferences = place_fn(preferences)
      # Test outputs.
      actual = entropy_fn(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_entropy_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.greedy()
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # Optionally convert to device array.
    preferences = place_fn(self.preferences)
    # Test greedy output in batch.
    actual = entropy_fn(preferences)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)


class EpsilonGreedyTest(parameterized.TestCase):

  def setUp(self):
    super(EpsilonGreedyTest, self).setUp()
    self.epsilon = 0.2

    self.preferences = np.array([[1, 1, 0, 0], [1, 2, 0, 0]], dtype=np.float32)
    self.samples = np.array([0, 1], dtype=np.int32)

    self.expected_probs = np.array(
        [[0.45, 0.45, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05]], dtype=np.float32)
    self.expected_logprob = np.array(
        [-0.7985077, -0.1625189], dtype=np.float32)
    self.expected_entropy = np.array(
        [1.01823008, 0.58750093], dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_probs(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    # Optionally compile.
    probs_fn = compile_fn(distrib.probs)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_probs):
      # Optionally convert to device array.
      preferences = place_fn(preferences)
      # Test outputs.
      actual = probs_fn(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_probs_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    # Optionally compile.
    probs_fn = compile_fn(distrib.probs)
    # Optionally convert to device array.
    preferences = place_fn(self.preferences)
    # Test greedy output in batch.
    actual = probs_fn(preferences)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_logprob(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # For each element in the batch.
    for preferences, samples, expected in zip(
        self.preferences, self.samples, self.expected_logprob):
      # Optionally convert to device array.
      preferences, samples = tree_map(place_fn, (preferences, samples))
      # Test output.
      actual = logprob_fn(samples, preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_logprob_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # Optionally convert to device array.
    preferences, samples = tree_map(place_fn, (self.preferences, self.samples))
    # Test greedy output in batch.
    actual = logprob_fn(samples, preferences)
    np.testing.assert_allclose(self.expected_logprob, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_entropy(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_entropy):
      # Optionally convert to device array.
      preferences = place_fn(preferences)
      # Test outputs.
      actual = entropy_fn(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_greedy_entropy_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # Optionally convert to device array.
    preferences = place_fn(self.preferences)
    # Test greedy output in batch.
    actual = entropy_fn(preferences)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_safe_epsilon_softmax_equivalence(self, compile_fn, place_fn):
    distrib = distributions.safe_epsilon_softmax(epsilon=self.epsilon,
                                                 temperature=0)
    # Optionally compile.
    probs_fn = compile_fn(distrib.probs)
    # Optionally convert to device array.
    preferences = place_fn(self.preferences)
    # Test greedy output in batch.
    actual = probs_fn(preferences)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # Optionally convert to device array.
    preferences, samples = tree_map(place_fn, (self.preferences, self.samples))
    # Test greedy output in batch.
    actual = logprob_fn(samples, preferences)
    np.testing.assert_allclose(self.expected_logprob, actual, atol=1e-4)

    # Optionally compile.
    sample_fn = compile_fn(distrib.sample)
    # Optionally convert to device array.
    preferences = place_fn(self.preferences)
    key = np.array([1, 2], dtype=np.uint32)
    actions = sample_fn(key, preferences)
    # test just the shape
    self.assertEqual(actions.shape, (2,))


class GaussianDiagonalTest(parameterized.TestCase):

  def setUp(self):
    super(GaussianDiagonalTest, self).setUp()

    self.mu = np.array([[1., -1], [0.1, -0.1]], dtype=np.float32)
    self.sigma = np.array([[0.1, 0.1], [0.2, 0.3]], dtype=np.float32)
    self.sample = np.array([[1.2, -1.1], [-0.1, 0.]], dtype=np.float32)

    # Expected values for the distribution's function were computed using
    # tfd.MultivariateNormalDiag (from the tensorflow_probability package).
    self.expected_prob_a = np.array(
        [1.3064219, 1.5219283], dtype=np.float32)
    self.expected_logprob_a = np.array(
        [0.26729202, 0.41997814], dtype=np.float32)
    self.expected_entropy = np.array(
        [-1.7672932, 0.02446628], dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_gaussian_prob(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.gaussian_diagonal()
    # Optionally compile.
    prob_fn = compile_fn(distrib.prob)
    # For each element in the batch.
    for mu, sigma, sample, expected in zip(
        self.mu, self.sigma, self.sample, self.expected_prob_a):
      # Optionally convert to device array.
      mu, sigma, sample = tree_map(place_fn, (mu, sigma, sample))
      # Test outputs.
      actual = prob_fn(sample, mu, sigma)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_gaussian_prob_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    # Optionally compile.
    prob_fn = compile_fn(distrib.prob)
    # Optionally convert to device array.
    mu, sigma, sample = tree_map(place_fn, (self.mu, self.sigma, self.sample))
    # Test greedy output in batch.
    actual = prob_fn(sample, mu, sigma)
    np.testing.assert_allclose(self.expected_prob_a, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_gaussian_logprob(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.gaussian_diagonal()
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # For each element in the batch.
    for mu, sigma, sample, expected in zip(
        self.mu, self.sigma, self.sample, self.expected_logprob_a):
      # Optionally convert to device array.
      mu, sigma, sample = tree_map(place_fn, (mu, sigma, sample))
      # Test output.
      actual = logprob_fn(sample, mu, sigma)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_gaussian_logprob_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    # Optionally compile.
    logprob_fn = compile_fn(distrib.logprob)
    # Optionally convert to device array.
    mu, sigma, sample = tree_map(place_fn, (self.mu, self.sigma, self.sample))
    # Test greedy output in batch.
    actual = logprob_fn(sample, mu, sigma)
    np.testing.assert_allclose(self.expected_logprob_a, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_gaussian_entropy(self, compile_fn, place_fn):
    """Tests for a single element."""
    distrib = distributions.gaussian_diagonal()
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # For each element in the batch.
    for mu, sigma, sample, expected in zip(
        self.mu, self.sigma, self.sample, self.expected_entropy):
      # Optionally convert to device array.
      mu, sigma, sample = tree_map(place_fn, (mu, sigma, sample))
      # Test outputs.
      actual = entropy_fn(mu, sigma)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_gaussian_entropy_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    # Optionally compile.
    entropy_fn = compile_fn(distrib.entropy)
    # Optionally convert to device array.
    mu, sigma = tree_map(place_fn, (self.mu, self.sigma))
    # Test greedy output in batch.
    actual = entropy_fn(mu, sigma)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)


class ImportanceSamplingTest(parameterized.TestCase):

  def setUp(self):
    super(ImportanceSamplingTest, self).setUp()

    self.pi_logits = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    self.mu_logits = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
    self.actions = np.array([1, 0], dtype=np.int32)

    pi = jax.nn.softmax(self.pi_logits)
    mu = jax.nn.softmax(self.mu_logits)
    self.expected_rhos = np.array(
        [pi[0][1] / mu[0][1], pi[1][0] / mu[1][0]], dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_importance_sampling_ratios(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Optionally compile.
    ratios_fn = compile_fn(distributions.categorical_importance_sampling_ratios)
    # For each element in the batch.
    for pi_logits, mu_logits, actions, expected in zip(
        self.pi_logits, self.mu_logits, self.actions, self.expected_rhos):
      # Optionally convert to device array.
      pi_logits, mu_logits, actions = tree_map(
          place_fn, (pi_logits, mu_logits, actions))
      # Test outputs.
      actual = ratios_fn(pi_logits, mu_logits, actions)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_importance_sampling_ratios_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    ratios_fn = compile_fn(
        jax.vmap(distributions.categorical_importance_sampling_ratios))
    # Optionally convert to device array.
    pi_logits, mu_logits, actions = tree_map(
        place_fn, (self.pi_logits, self.mu_logits, self.actions))
    # Test softmax output in batch.
    actual = ratios_fn(pi_logits, mu_logits, actions)
    np.testing.assert_allclose(self.expected_rhos, actual, atol=1e-4)


class CategoricalKLTest(parameterized.TestCase):

  def setUp(self):
    super(CategoricalKLTest, self).setUp()
    self.p_logits = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    p_probs = np.array([[0.42231882, 0.42231882, 0.15536241],
                        [0.24472848, 0.66524094, 0.09003057]],
                       dtype=np.float32)
    p_logprobs = np.log(p_probs)
    self.q_logits = np.array([[1, 2, 0], [1, 1, 0]], dtype=np.float32)
    q_probs = np.array([[0.24472848, 0.66524094, 0.09003057],
                        [0.42231882, 0.42231882, 0.15536241]],
                       dtype=np.float32)
    q_logprobs = np.log(q_probs)

    self.expected_kl = np.sum(p_probs * (p_logprobs - q_logprobs), axis=-1)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_categorical_kl_divergence(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Optionally compile.
    kl_fn = compile_fn(distributions.categorical_kl_divergence)
    # For each element in the batch.
    for p_logits, q_logits, expected in zip(
        self.p_logits, self.q_logits, self.expected_kl):
      # Optionally convert to device array.
      p_logits, q_logits = tree_map(place_fn, (p_logits, q_logits))
      # Test outputs.
      actual = kl_fn(p_logits, q_logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_categorical_kl_divergence_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    kl_fn = compile_fn(jax.vmap(distributions.categorical_kl_divergence))
    # Optionally convert to device array.
    p_logits, q_logits = tree_map(place_fn, (self.p_logits, self.q_logits))
    # Test softmax output in batch.
    actual = kl_fn(p_logits, q_logits)
    np.testing.assert_allclose(self.expected_kl, actual, atol=1e-4)


class CategoricalCrossEntropyTest(parameterized.TestCase):

  def setUp(self):
    super(CategoricalCrossEntropyTest, self).setUp()

    self.labels = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
    self.logits = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)

    self.expected = np.array([9.00013, 3.0696733], dtype=np.float32)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_categorical_cross_entropy(self, compile_fn, place_fn):
    """Tests for a single element."""
    # Optionally compile.
    cross_entropy = compile_fn(distributions.categorical_cross_entropy)
    # Test outputs.
    for labels, logits, expected in zip(
        self.labels, self.logits, self.expected):
      # Optionally convert to device array.
      labels, logits = tree_map(place_fn, (labels, logits))
      # Test outputs.
      actual = cross_entropy(labels=labels, logits=logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @parameterized.named_parameters(
      ('JitOnp', jax.jit, lambda t: t),
      ('NoJitOnp', lambda fn: fn, lambda t: t),
      ('JitJnp', jax.jit, jax.device_put),
      ('NoJitJnp', lambda fn: fn, jax.device_put))
  def test_categorical_cross_entropy_batch(self, compile_fn, place_fn):
    """Tests for a full batch."""
    # Vmap and optionally compile.
    cross_entropy = jax.vmap(distributions.categorical_cross_entropy)
    cross_entropy = compile_fn(cross_entropy)
    # Optionally convert to device array.
    labels, logits = tree_map(place_fn, (self.labels, self.logits))
    # Test outputs.
    actual = cross_entropy(labels, logits)
    np.testing.assert_allclose(self.expected, actual, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
