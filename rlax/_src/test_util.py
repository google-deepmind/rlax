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
import jax
import numpy as np


def check_output(fn, *args, **kwargs):
  """Wraps the function to check the function outputs from different usages.

  On each call of the wrapper, the function will be called with or without vmap,
  with or without jit and with or without device_put. The outputs will be
  compared to the output from vmap.

  Args:
    fn: Function to wrap.
    *args: Extra args to the function.
    **kwargs: Extra kwargs to the function.

  Returns:
    The wrapped function.
  """
  fn = functools.partial(fn, *args, **kwargs)
  base_variant = jax.vmap(fn)
  variants = [
      ("jit", jax.jit(base_variant)),
      ("device", _with_device(base_variant)),
      ("device_jit", _with_device(jax.jit(base_variant))),
      ("iteration", _with_iteration(fn)),
      ("iteration_jit", _with_iteration(jax.jit(fn))),
      ("iteration_device", _with_iteration(_with_device(fn))),
      ("iteration_device_jit", _with_iteration(_with_device(jax.jit(fn)))),
  ]

  def wrapper(*args):
    expected = base_variant(*args)
    for name, variant in variants:
      try:
        got = variant(*args)
      except Exception as e:
        raise ValueError("failed variant: {}, cause: {}".format(name, e)
                         ).with_traceback(e.__traceback__)
      err_msg = "variant: {}".format(name)
      np.testing.assert_allclose(expected, got, rtol=1e-3, err_msg=err_msg)
    return expected
  return wrapper


def strict_zip(*args):
  """A strict `zip()` that requires sequences with the same length."""
  expected_len = len(args[0])
  for arg in args:
    np.testing.assert_equal(len(arg), expected_len)
  return zip(*args)


def _with_iteration(fn):
  def wrapper(*args):
    outputs = []
    # Iterating over the first axis.
    for inputs in strict_zip(*args):
      outputs.append(fn(*inputs))
    return jax.tree_util.tree_multimap(lambda *x: np.stack(x), *outputs)
  return wrapper


def _with_device(fn):
  def wrapper(*args):
    converted = jax.device_put(args)
    return fn(*converted)
  return wrapper


