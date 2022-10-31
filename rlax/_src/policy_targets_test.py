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
"""Unit tests for `policy_targets.py`."""

from absl.testing import absltest
import distrax
import jax.numpy as jnp
import numpy as np

from rlax._src import policy_targets


class PolicyTargetsTest(absltest.TestCase):

  def test_sampled_policy_distillation_loss(self):
    targets = policy_targets.PolicyTarget(
        actions=jnp.array([0, 1], dtype=jnp.int32),
        weights=jnp.array([0.5, 0.0], dtype=jnp.float32))
    distribution = distrax.Categorical(
        probs=jnp.array([[0.7, 0.3], [0.9, 0.1]], dtype=jnp.float32))
    loss = policy_targets.sampled_policy_distillation_loss(
        distribution=distribution,
        policy_targets=targets)
    expected_loss = 0.089169
    np.testing.assert_allclose(expected_loss, loss, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
