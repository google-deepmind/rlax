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
"""Utilities for tests."""

import functools
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np


def parameterize_variant(*testcases):
  """A decorator to test each test case with all variants.

  This decorator is an enhanced version of `parameterized.named_parameters`.
  The variant factory is appended to the end of the tuple of the function
  parameters.

  Args:
    *testcases: Tuples to pass to `parameterized.named_parameters`.
      An empty list of testcases will produce one test for each variant.

  Returns:
    A test generator to test each test case with all variants.
  """
  factories = _get_variant_factories()
  return _enhance_named_parameters(factories, testcases)


def parameterize_vmap_variant(*testcases):
  """A decorator to test each test case with all variants of vmap.

  This decorator is an enhanced version of `parameterized.named_parameters`.
  The variant factory is appended to the end of the tuple of the function
  parameters.

  Args:
    *testcases: Tuples to pass to `parameterized.named_parameters`.
      An empty list of testcases will produce one test for each variant.

  Returns:
    A test generator to test each test case with all variants.
  """
  factories = _get_vmap_variant_factories()
  return _enhance_named_parameters(factories, testcases)


def _enhance_named_parameters(factories, testcases):
  """Calls parameterized.named_parameters() with enhanced testcases."""
  if not testcases:
    testcases = [("variant",)]
  enhanced_testcases = []
  for testcase in testcases:
    name = testcase[0]
    test_args = tuple(testcase[1:])
    for variant_name, raw_factory in factories.items():
      variant_factory = _produce_variant_factory(raw_factory)
      # The variant_factory will be the last argument.
      case = (name + "_" + variant_name,) + test_args + (variant_factory,)
      enhanced_testcases.append(case)
  return parameterized.named_parameters(
      *enhanced_testcases)


def _produce_variant_factory(raw_factory):
  def variant_factory(fn, *args, **kwargs):
    return raw_factory(functools.partial(fn, *args, **kwargs))
  return variant_factory


def _get_variant_factories():
  factories = dict(
      jit=lambda f: _without_device(jax.jit(f)),
      device=_with_device,
      device_jit=lambda f: _with_device(jax.jit(f)),
  )
  return factories


def _get_vmap_variant_factories():
  """Returns factories for variants operating on batch data."""
  factories = dict(
      jit_vmap=lambda f: _without_device(jax.jit(jax.vmap(f))),
      device_vmap=lambda f: _with_device(jax.vmap(f)),
      device_jit_vmap=lambda f: _with_device(jax.jit(jax.vmap(f))),
      iteration=lambda f: _with_iteration(_without_device(f)),
      iteration_jit=lambda f: _with_iteration(_without_device(jax.jit(f))),
      iteration_device=lambda f: _with_iteration(_with_device(f)),
      iteration_device_jit=lambda f: _with_iteration(_with_device(jax.jit(f))),
  )
  return factories


def strict_zip(*args):
  """A strict `zip()` that requires sequences with the same length."""
  expected_len = len(args[0])
  for arg in args:
    np.testing.assert_equal(len(arg), expected_len)
  return zip(*args)


def _with_iteration(fn):
  """Uses iteration to produce vmap-like output."""
  def wrapper(*args):
    outputs = []
    # Iterating over the first axis.
    for inputs in strict_zip(*args):
      outputs.append(fn(*inputs))
    return jax.tree_util.tree_multimap(lambda *x: jnp.stack(x), *outputs)
  return wrapper


def _with_device(fn):
  """Puts all inputs to a device."""
  def wrapper(*args):
    converted = jax.device_put(args)
    return fn(*converted)
  return wrapper


def _without_device(fn):
  """Moves all inputs outside of a device."""
  def wrapper(*args):
    def get(x):
      if isinstance(x, jnp.DeviceArray):
        return jax.device_get(x)
      return x
    converted = jax.tree_util.tree_map(get, args)
    return fn(*converted)
  return wrapper

