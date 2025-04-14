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
import chex
import jax
import numpy as np
from rlax._src import distributions

jax.config.update('jax_threefry_partitionable', False)


@chex.dataclass(frozen=True)
class _MockActionSpec:
  minimum: chex.Array
  maximum: chex.Array


class CategoricalSampleTest(parameterized.TestCase):

  @chex.all_variants()
  def test_categorical_sample(self):
    key = np.array([1, 2], dtype=np.uint32)
    probs = np.array([0.2, 0.3, 0.5])
    sample = self.variant(distributions.categorical_sample)(key, probs)
    self.assertEqual(sample, 0)

  @chex.all_variants()
  @parameterized.parameters(
      ((-1., 10., -1.),),
      ((0., 0., 0.),),
      ((1., np.inf, 3.),),
      ((1., 2., -np.inf),),
      ((1., 2., np.nan),),
  )
  def test_categorical_sample_on_invalid_distributions(self, probs):
    key = np.array([1, 2], dtype=np.uint32)
    probs = np.asarray(probs)
    sample = self.variant(distributions.categorical_sample)(key, probs)
    self.assertEqual(sample, -1)


class SoftmaxTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

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
    self.expected_clipped_entropy = {0.5: 0.549306, 0.9: 0.988751}

  @chex.all_variants()
  @parameterized.named_parameters(
      ('softmax', distributions.softmax),
      ('clipped_entropy_softmax', distributions.clipped_entropy_softmax))
  def test_softmax_probs(self, softmax_dist):
    """Tests for a single element."""
    distrib = softmax_dist(temperature=10.)
    softmax = self.variant(distrib.probs)
    # For each element in the batch.
    for logits, expected in zip(self.logits, self.expected_probs):
      # Test outputs.
      actual = softmax(logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('softmax', distributions.softmax),
      ('clipped_entropy_softmax', distributions.clipped_entropy_softmax))
  def test_softmax_probs_batch(self, softmax_dist):
    """Tests for a full batch."""
    distrib = softmax_dist(temperature=10.)
    softmax = self.variant(distrib.probs)
    # Test softmax output in batch.
    actual = softmax(self.logits)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('softmax', distributions.softmax),
      ('clipped_entropy_softmax', distributions.clipped_entropy_softmax))
  def test_softmax_logprob(self, softmax_dist):
    """Tests for a single element."""
    distrib = softmax_dist()
    logprob_fn = self.variant(distrib.logprob)
    # For each element in the batch.
    for logits, samples, expected in zip(
        self.logits, self.samples, self.expected_logprobs):
      # Test output.
      actual = logprob_fn(samples, logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('softmax', distributions.softmax),
      ('clipped_entropy_softmax', distributions.clipped_entropy_softmax))
  def test_softmax_logprob_batch(self, softmax_dist):
    """Tests for a full batch."""
    distrib = softmax_dist()
    logprob_fn = self.variant(distrib.logprob)
    # Test softmax output in batch.
    actual = logprob_fn(self.samples, self.logits)
    np.testing.assert_allclose(self.expected_logprobs, actual, atol=1e-4)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('softmax', distributions.softmax),
      ('clipped_entropy_softmax', distributions.clipped_entropy_softmax))
  def test_softmax_entropy(self, softmax_dist):
    """Tests for a single element."""
    distrib = softmax_dist()
    entropy_fn = self.variant(distrib.entropy)
    # For each element in the batch.
    for logits, expected in zip(self.logits, self.expected_entropy):
      # Test outputs.
      actual = entropy_fn(logits)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  @parameterized.parameters((0.9, [0.988751, 0.832396]),
                            (0.5, [0.549306, 0.549306]))
  def test_softmax_clipped_entropy_batch(self, entropy_clip, expected_clipped):
    """Tests for a single element."""
    distrib = distributions.clipped_entropy_softmax(entropy_clip=entropy_clip)
    entropy_fn = self.variant(distrib.entropy)
    # Test softmax output in batch.
    actual = entropy_fn(self.logits)
    np.testing.assert_allclose(expected_clipped, actual, atol=1e-4)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('softmax', distributions.softmax),
      ('clipped_entropy_softmax', distributions.clipped_entropy_softmax))
  def test_softmax_entropy_batch(self, softmax_dist):
    """Tests for a full batch."""
    distrib = softmax_dist()
    entropy_fn = self.variant(distrib.entropy)
    # Test softmax output in batch.
    actual = entropy_fn(self.logits)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)


class GreedyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.preferences = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    self.samples = np.array([0, 1], dtype=np.int32)

    self.expected_probs = np.array(
        [[0.5, 0.5, 0.], [0., 1., 0.]], dtype=np.float32)
    self.expected_logprob = np.array(
        [-0.6931472, 0.], dtype=np.float32)
    self.expected_entropy = np.array(
        [0.6931472, 0.], dtype=np.float32)

  @chex.all_variants()
  def test_greedy_probs(self):
    """Tests for a single element."""
    distrib = distributions.greedy()
    greedy = self.variant(distrib.probs)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_probs):
      # Test outputs.
      actual = greedy(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_probs_batch(self):
    """Tests for a full batch."""
    distrib = distributions.greedy()
    greedy = self.variant(distrib.probs)
    # Test greedy output in batch.
    actual = greedy(self.preferences)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_logprob(self):
    """Tests for a single element."""
    distrib = distributions.greedy()
    logprob_fn = self.variant(distrib.logprob)
    # For each element in the batch.
    for preferences, samples, expected in zip(
        self.preferences, self.samples, self.expected_logprob):
      # Test output.
      actual = logprob_fn(samples, preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_logprob_batch(self):
    """Tests for a full batch."""
    distrib = distributions.greedy()
    logprob_fn = self.variant(distrib.logprob)
    # Test greedy output in batch.
    actual = logprob_fn(self.samples, self.preferences)
    np.testing.assert_allclose(self.expected_logprob, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_entropy(self):
    """Tests for a single element."""
    distrib = distributions.greedy()
    entropy_fn = self.variant(distrib.entropy)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_entropy):
      # Test outputs.
      actual = entropy_fn(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_entropy_batch(self):
    """Tests for a full batch."""
    distrib = distributions.greedy()
    entropy_fn = self.variant(distrib.entropy)
    # Test greedy output in batch.
    actual = entropy_fn(self.preferences)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)


class EpsilonGreedyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.epsilon = 0.2

    self.preferences = np.array([[1, 1, 0, 0], [1, 2, 0, 0]], dtype=np.float32)
    self.samples = np.array([0, 1], dtype=np.int32)

    self.expected_probs = np.array(
        [[0.45, 0.45, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05]], dtype=np.float32)
    self.expected_logprob = np.array(
        [-0.7985077, -0.1625189], dtype=np.float32)
    self.expected_entropy = np.array(
        [1.01823008, 0.58750093], dtype=np.float32)

  @chex.all_variants()
  def test_greedy_probs(self):
    """Tests for a single element."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    probs_fn = self.variant(distrib.probs)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_probs):
      # Test outputs.
      actual = probs_fn(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_probs_batch(self):
    """Tests for a full batch."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    probs_fn = self.variant(distrib.probs)
    # Test greedy output in batch.
    actual = probs_fn(self.preferences)
    np.testing.assert_allclose(self.expected_probs, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_logprob(self):
    """Tests for a single element."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    logprob_fn = self.variant(distrib.logprob)
    # For each element in the batch.
    for preferences, samples, expected in zip(
        self.preferences, self.samples, self.expected_logprob):
      # Test output.
      actual = logprob_fn(samples, preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_logprob_batch(self):
    """Tests for a full batch."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    logprob_fn = self.variant(distrib.logprob)
    # Test greedy output in batch.
    actual = logprob_fn(self.samples, self.preferences)
    np.testing.assert_allclose(self.expected_logprob, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_entropy(self):
    """Tests for a single element."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    entropy_fn = self.variant(distrib.entropy)
    # For each element in the batch.
    for preferences, expected in zip(self.preferences, self.expected_entropy):
      # Test outputs.
      actual = entropy_fn(preferences)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_greedy_entropy_batch(self):
    """Tests for a full batch."""
    distrib = distributions.epsilon_greedy(self.epsilon)
    entropy_fn = self.variant(distrib.entropy)
    # Test greedy output in batch.
    actual = entropy_fn(self.preferences)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)


class GaussianDiagonalTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.mu = np.array([[1., -1], [0.1, -0.1]], dtype=np.float32)
    self.sigma = np.array([[0.1, 0.1], [0.2, 0.3]], dtype=np.float32)
    self.sample = np.array([[1.2, -1.1], [-0.1, 0.]], dtype=np.float32)
    self.other_mu = np.array([[1., -10.], [0.3, -0.2]], dtype=np.float32)
    self.other_sigma = np.array([[0.1, 0.1], [0.8, 0.3]], dtype=np.float32)

    # Expected values for the distribution's function were computed using
    # tfd.MultivariateNormalDiag (from the tensorflow_probability package).
    self.expected_prob_a = np.array(
        [1.3064219, 1.5219283], dtype=np.float32)
    self.expected_logprob_a = np.array(
        [0.26729202, 0.41997814], dtype=np.float32)
    self.expected_entropy = np.array(
        [-1.7672932, 0.02446628], dtype=np.float32)
    self.expected_kl = np.array(
        [4050.00, 1.00435], dtype=np.float32)
    self.expected_kl_to_std_normal = np.array(
        [4.6151705, 1.8884108], dtype=np.float32)

  @chex.all_variants()
  def test_gaussian_prob(self):
    """Tests for a single element."""
    distrib = distributions.gaussian_diagonal()
    prob_fn = self.variant(distrib.prob)
    # For each element in the batch.
    for mu, sigma, sample, expected in zip(
        self.mu, self.sigma, self.sample, self.expected_prob_a):
      # Test outputs.
      actual = prob_fn(sample, mu, sigma)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_gaussian_prob_batch(self):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    prob_fn = self.variant(distrib.prob)
    # Test greedy output in batch.
    actual = prob_fn(self.sample, self.mu, self.sigma)
    np.testing.assert_allclose(self.expected_prob_a, actual, atol=1e-4)

  @chex.all_variants()
  def test_gaussian_logprob(self):
    """Tests for a single element."""
    distrib = distributions.gaussian_diagonal()
    logprob_fn = self.variant(distrib.logprob)
    # For each element in the batch.
    for mu, sigma, sample, expected in zip(
        self.mu, self.sigma, self.sample, self.expected_logprob_a):
      # Test output.
      actual = logprob_fn(sample, mu, sigma)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_gaussian_logprob_batch(self):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    logprob_fn = self.variant(distrib.logprob)
    # Test greedy output in batch.
    actual = logprob_fn(self.sample, self.mu, self.sigma)
    np.testing.assert_allclose(self.expected_logprob_a, actual, atol=1e-4)

  @chex.all_variants()
  def test_gaussian_entropy(self):
    """Tests for a single element."""
    distrib = distributions.gaussian_diagonal()
    entropy_fn = self.variant(distrib.entropy)
    # For each element in the batch.
    for mu, sigma, expected in zip(
        self.mu, self.sigma, self.expected_entropy):
      # Test outputs.
      actual = entropy_fn(mu, sigma)
      np.testing.assert_allclose(expected, actual, atol=1e-4)

  @chex.all_variants()
  def test_gaussian_entropy_batch(self):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    entropy_fn = self.variant(distrib.entropy)
    # Test greedy output in batch.
    actual = entropy_fn(self.mu, self.sigma)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-4)

  @chex.all_variants()
  def test_gaussian_kl_batch(self):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    kl_fn = self.variant(distrib.kl)
    # Test greedy output in batch.
    actual = kl_fn(self.mu, self.sigma, self.other_mu, self.other_sigma)
    np.testing.assert_allclose(self.expected_kl, actual, atol=1e-3, rtol=1e-6)

  @chex.all_variants()
  def test_gaussian_kl_to_std_normal_batch(self):
    """Tests for a full batch."""
    distrib = distributions.gaussian_diagonal()
    kl_fn = self.variant(distrib.kl_to_standard_normal)
    # Test greedy output in batch.
    actual = kl_fn(self.mu, self.sigma)
    np.testing.assert_allclose(self.expected_kl_to_std_normal, actual,
                               atol=1e-4)


class SquashedGaussianTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.mu = np.array([[1., -1], [0.1, -0.1]], dtype=np.float32)
    self.sigma = np.array([[0.1, 0.1], [0.2, 0.3]], dtype=np.float32)
    self.sample = np.array([[0.5, -0.6], [-.4, -.2]], dtype=np.float32)
    self.other_mu = np.array([[1., -10.], [0.3, -0.2]], dtype=np.float32)
    self.other_sigma = np.array([[0.1, 0.1], [0.8, 0.3]], dtype=np.float32)
    self.action_spec = _MockActionSpec(minimum=np.array([-1.0]),
                                       maximum=np.array([2.0]))
    self.sigma_min = 0.0
    self.sigma_max = 2.0
    # Expected values for the distribution's function were computed using
    # tfd.MultivariateNormalDiag (from the tensorflow_probability package).
    self.expected_prob_a = np.array(
        [0.016403, 0.011328], dtype=np.float32)
    self.expected_logprob_a = np.array(
        [-4.110274, -4.480485], dtype=np.float32)
    self.expected_entropy = np.array(
        [5.037213, 5.326565], dtype=np.float32)
    self.expected_kl = np.array(
        [0.003151, 0.164303], dtype=np.float32)
    self.expected_kl_to_std_normal = np.array(
        [6.399713, 8.61989], dtype=np.float32)

  @chex.all_variants()
  def test_squashed_gaussian_prob(self):
    """Tests for a full batch."""
    distrib = distributions.squashed_gaussian(sigma_min=self.sigma_min,
                                              sigma_max=self.sigma_max)
    prob_fn = self.variant(distrib.prob)
    # Test greedy output in batch.
    actual = prob_fn(self.sample, self.mu, self.sigma, self.action_spec)
    np.testing.assert_allclose(self.expected_prob_a, actual, atol=1e-4)

  @chex.all_variants()
  def test_squashed_gaussian_logprob(self):
    """Tests for a full batch."""
    distrib = distributions.squashed_gaussian(sigma_min=self.sigma_min,
                                              sigma_max=self.sigma_max)
    logprob_fn = self.variant(distrib.logprob)
    # Test greedy output in batch.
    actual = logprob_fn(self.sample, self.mu, self.sigma, self.action_spec)
    np.testing.assert_allclose(self.expected_logprob_a, actual, atol=1e-3,
                               rtol=1e-6)

  @chex.all_variants()
  def test_squashed_gaussian_entropy(self):
    """Tests for a full batch."""
    distrib = distributions.squashed_gaussian(sigma_min=self.sigma_min,
                                              sigma_max=self.sigma_max)
    entropy_fn = self.variant(distrib.entropy)
    # Test greedy output in batch.
    actual = entropy_fn(self.mu, self.sigma)
    np.testing.assert_allclose(self.expected_entropy, actual, atol=1e-3,
                               rtol=1e-6)

  @chex.all_variants()
  def test_squashed_gaussian_kl(self):
    """Tests for a full batch."""
    distrib = distributions.squashed_gaussian(sigma_min=self.sigma_min,
                                              sigma_max=self.sigma_max)
    kl_fn = self.variant(distrib.kl)
    # Test greedy output in batch.
    actual = kl_fn(self.mu, self.sigma, self.other_mu, self.other_sigma)
    np.testing.assert_allclose(self.expected_kl, actual, atol=1e-3, rtol=1e-6)

  @chex.all_variants()
  def test_squashed_gaussian_kl_to_std_normal(self):
    """Tests for a full batch."""
    distrib = distributions.squashed_gaussian(sigma_min=self.sigma_min,
                                              sigma_max=self.sigma_max)
    kl_fn = self.variant(distrib.kl_to_standard_normal)
    # Test greedy output in batch.
    actual = kl_fn(self.mu, self.sigma)
    np.testing.assert_allclose(self.expected_kl_to_std_normal, actual,
                               atol=1e-3, rtol=1e-5)


class ImportanceSamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.pi_logits = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    self.mu_logits = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
    self.actions = np.array([1, 0], dtype=np.int32)

    pi = jax.nn.softmax(self.pi_logits)
    mu = jax.nn.softmax(self.mu_logits)
    self.expected_rhos = np.array(
        [pi[0][1] / mu[0][1], pi[1][0] / mu[1][0]], dtype=np.float32)

  @chex.all_variants()
  def test_importance_sampling_ratios_batch(self):
    """Tests for a full batch."""
    ratios_fn = self.variant(
        distributions.categorical_importance_sampling_ratios)
    # Test softmax output in batch.
    actual = ratios_fn(self.pi_logits, self.mu_logits, self.actions)
    np.testing.assert_allclose(self.expected_rhos, actual, atol=1e-4)


class CategoricalKLTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
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

  @chex.all_variants()
  def test_categorical_kl_divergence_batch(self):
    """Tests for a full batch."""
    kl_fn = self.variant(distributions.categorical_kl_divergence)
    # Test softmax output in batch.
    actual = kl_fn(self.p_logits, self.q_logits)
    np.testing.assert_allclose(self.expected_kl, actual, atol=1e-4)


class CategoricalCrossEntropyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.labels = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
    self.logits = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)

    self.expected = np.array([9.00013, 3.0696733], dtype=np.float32)

  @chex.all_variants()
  def test_categorical_cross_entropy_batch(self):
    """Tests for a full batch."""
    cross_entropy = self.variant(jax.vmap(
        distributions.categorical_cross_entropy))
    # Test outputs.
    actual = cross_entropy(self.labels, self.logits)
    np.testing.assert_allclose(self.expected, actual, atol=1e-4)


class MultivariateNormalKLTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Test numbers taken from tfd.MultivariateNormalDiag
    self.mu0 = np.array([[5., -1], [0.1, -0.1]], dtype=np.float32)
    self.sigma0 = np.array([[0.3, 0.1], [0.2, 0.3]], dtype=np.float32)
    self.mu1 = np.array([[0.005, -11.], [-0.25, -0.2]], dtype=np.float32)
    self.sigma1 = np.array([[0.1, 0.1], [0.6, 0.3]], dtype=np.float32)
    self.expected_kl = np.array([6.2504023e+03, 8.7986231e-01],
                                dtype=np.float32)

  @chex.all_variants()
  def test_multivariate_normal_kl_divergence_batch(self):
    kl_fn = self.variant(distributions.multivariate_normal_kl_divergence)
    actual = kl_fn(self.mu0, self.sigma0, self.mu1, self.sigma1)
    np.testing.assert_allclose(self.expected_kl, actual, atol=1e-3, rtol=1e-6)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
