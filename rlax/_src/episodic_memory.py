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
"""JAX functions implementing utilities on a simple episodic memory.

Some reinforcement learning agents utilize episodic memory to calculate
exploration bonus rewards among other things. This subpackage contains utility
functions for querying episodic memory.
"""

import functools
from typing import Callable

import chex
import jax
import jax.numpy as jnp

Array = chex.Array
Scalar = chex.Scalar


@chex.dataclass
class KNNQueryResult():
  neighbors: jnp.ndarray
  neighbor_indices: jnp.ndarray
  neighbor_neg_distances: jnp.ndarray


def _sqeuclidian(x: Array, y: Array) -> Scalar:
  return jnp.sum(jnp.square(x - y))


def _cdist(
    a: Array,
    b: Array,
    metric: Callable[[Array, Array], Scalar]
) -> Array:
  """Returns the distance between each pair of the two collections of inputs."""
  return jax.vmap(jax.vmap(metric, (None, 0)), (0, None))(a, b)


def knn_query(
    data: Array,
    query_points: Array,
    num_neighbors: int,
    metric: Callable[[Array, Array], Scalar] = _sqeuclidian
) -> KNNQueryResult:
  """Finds closest neighbors in data to the query points & their neg distances.

  NOTE: For this function to be jittable, static_argnums=[2,] must be passed, as
  the internal jax.lax.top_k(neg_distances, num_neighbors) computation cannot be
  jitted with a dynamic num_neighbors that is passed as an argument.

  Args:
    data: array of existing data points (elements in database x feature size)
    query_points: array of points to find neighbors of
        (num query points x feature size).
    num_neighbors: number of neighbors to find.
    metric: Metric to use in calculating distance between two points.

  Returns:
    KNNQueryResult with (all sorted by neg distance):
      - neighbors (num query points x num neighbors x feature size)
      - neighbor_indices (num query points x num neighbors)
      - neighbor_neg_distances (num query points x num neighbors)
   * if num_neighbors is greater than number of elements in the database,
     just return KNNQueryResult with the number of elements in the database.
  """
  chex.assert_rank([data, query_points], 2)
  assert data.shape[-1] == query_points.shape[-1]
  distance_fn = jax.jit(functools.partial(_cdist, metric=metric))
  neg_distances = -distance_fn(query_points, data)
  neg_distances, indices = jax.lax.top_k(
      neg_distances, k=min(num_neighbors, data.shape[0]))
  # Batch index into data using indices shaped [num queries, num neighbors]
  neighbors = jax.vmap(lambda d, i: d[i], (None, 0))(jnp.array(data), indices)
  return KNNQueryResult(neighbors=neighbors, neighbor_indices=indices,
                        neighbor_neg_distances=neg_distances)
