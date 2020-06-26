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
"""Utilities for target network switching."""

from typing import Any
import chex
import jax

Numeric = chex.Numeric


def periodic_update(new_tensors: Any, old_tensors: Any, is_time: Numeric):
  """Periodically switch all elements from a nested struct with new elements."""
  return jax.tree_multimap(
      lambda new, old: jax.lax.select(is_time, new, old),
      new_tensors, old_tensors)


def incremental_update(new_tensors, old_tensors, tau: Numeric):
  """Incrementally update all elements from a nested struct."""
  return jax.tree_multimap(
      lambda new, old: tau * new + (1.0 - tau) * old,
      new_tensors, old_tensors)
