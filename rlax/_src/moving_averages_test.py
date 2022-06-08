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
"""Tests for moving_averages.py."""

from absl.testing import absltest
import numpy as np

from rlax._src import moving_averages


class EmaTest(absltest.TestCase):

  def test_moments(self):
    values = [5.0, 7.0]
    decay = 0.9
    d = decay

    init_state, update = moving_averages.create_ema(decay=decay)
    state = init_state(values[0])
    moments, state = update(values[0], state)

    np.testing.assert_allclose(moments.mean, values[0], atol=1e-5)
    np.testing.assert_allclose(moments.variance, 0.0, atol=1e-5)

    moments, state = update(values[1], state)
    np.testing.assert_allclose(
        moments.mean,
        (d * (1 - d) * values[0] + (1 - d) * values[1]) / (1 - d**2), atol=1e-5)
    np.testing.assert_allclose(
        moments.variance,
        (d * (1 - d) * values[0]**2 + (1 - d) * values[1]**2) / (1 - d**2) -
        moments.mean**2, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
