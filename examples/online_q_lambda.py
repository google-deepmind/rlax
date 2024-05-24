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
"""An online Q-lambda agent trained to play BSuite's Catch env."""

import collections
from absl import app
from absl import flags
from bsuite.environments import catch
import dm_env
import haiku as hk
from haiku import nets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from rlax.examples import experiment

ActorOutput = collections.namedtuple("ActorOutput", ["actions", "q_values"])

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("train_episodes", 500, "Number of train episodes.")
flags.DEFINE_integer("num_hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_integer("sequence_length", 4,
                     "Length of (action, timestep) sequences.")
flags.DEFINE_float("epsilon", 0.01, "Epsilon-greedy exploration probability.")
flags.DEFINE_float("lambda_", 0.9, "Mixing parameter for Q(lambda).")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50,
                     "Number of episodes between evaluations.")


def build_network(num_hidden_units: int, num_actions: int) -> hk.Transformed:
  """Factory for a simple MLP network for approximating Q-values."""

  def q(obs):
    flatten = lambda x: jnp.reshape(x, (-1,))
    network = hk.Sequential(
        [flatten, nets.MLP([num_hidden_units, num_actions])])
    return network(obs)

  return hk.without_apply_rng(hk.transform(q))


class SequenceAccumulator:
  """Accumulator for gathering the latest timesteps into sequences.

  Note sequences can overlap and cross episode boundaries.
  """

  def __init__(self, length):
    self._timesteps = collections.deque(maxlen=length)

  def push(self, timestep, action):
    # Replace `None`s with zeros as these will be put into NumPy arrays.
    a_tm1 = 0 if action is None else action
    timestep_t = timestep._replace(
        step_type=int(timestep.step_type),
        reward=0. if timestep.reward is None else timestep.reward,
        discount=0. if timestep.discount is None else timestep.discount,
    )
    self._timesteps.append((a_tm1, timestep_t))

  def sample(self, batch_size):
    """Returns current sequence of accumulated timesteps."""
    if batch_size != 1:
      raise ValueError("Require batch_size == 1.")
    if len(self._timesteps) != self._timesteps.maxlen:
      raise ValueError("Not enough timesteps for a full sequence.")

    actions, timesteps = jax.tree.map(lambda *ts: np.stack(ts),
                                           *self._timesteps)
    return actions, timesteps

  def is_ready(self, batch_size):
    if batch_size != 1:
      raise ValueError("Require batch_size == 1.")
    return len(self._timesteps) == self._timesteps.maxlen


class OnlineQLambda:
  """An online Q-lambda agent."""

  def __init__(self, observation_spec, action_spec, num_hidden_units, epsilon,
               lambda_, learning_rate):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._epsilon = epsilon
    self._lambda = lambda_

    # Neural net and optimiser.
    self._network = build_network(num_hidden_units, action_spec.num_values)

    self._optimizer = optax.adam(learning_rate)
    # Jitting for speed.
    self.actor_step = jax.jit(self.actor_step)
    self.learner_step = jax.jit(self.learner_step)

  def initial_params(self, key):
    sample_input = self._observation_spec.generate_value()
    return self._network.init(key, sample_input)

  def initial_actor_state(self):
    return ()

  def initial_learner_state(self, params):
    return self._optimizer.init(params)

  def actor_step(self, params, env_output, actor_state, key, evaluation):
    q = self._network.apply(params, env_output.observation)
    train_a = rlax.epsilon_greedy(self._epsilon).sample(key, q)
    eval_a = rlax.greedy().sample(key, q)
    a = jax.lax.select(evaluation, eval_a, train_a)
    return ActorOutput(actions=a, q_values=q), actor_state

  def learner_step(self, params, data, learner_state, unused_key):
    dloss_dtheta = jax.grad(self._loss)(params, *data)
    updates, learner_state = self._optimizer.update(dloss_dtheta, learner_state)
    params = optax.apply_updates(params, updates)
    return params, learner_state

  def _loss(self, params, actions, timesteps):
    """Calculates Q-lambda loss given parameters, actions and timesteps."""
    network_apply_sequence = jax.vmap(self._network.apply, in_axes=(None, 0))
    q = network_apply_sequence(params, timesteps.observation)

    # Use a mask since the sequence could cross episode boundaries.
    mask = jnp.not_equal(timesteps.step_type, int(dm_env.StepType.LAST))
    a_tm1 = actions[1:]
    r_t = timesteps.reward[1:]
    # Discount ought to be zero on a LAST timestep, use the mask to ensure this.
    discount_t = timesteps.discount[1:] * mask[1:]
    q_tm1 = q[:-1]
    q_t = q[1:]
    mask_tm1 = mask[:-1]

    # Mask out TD errors for the last state in an episode.
    td_error_tm1 = mask_tm1 * rlax.q_lambda(
        q_tm1, a_tm1, r_t, discount_t, q_t, lambda_=self._lambda)
    return jnp.sum(rlax.l2_loss(td_error_tm1)) / jnp.sum(mask_tm1)


def main(unused_arg):
  env = catch.Catch(seed=FLAGS.seed)
  agent = OnlineQLambda(
      observation_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      num_hidden_units=FLAGS.num_hidden_units,
      epsilon=FLAGS.epsilon,
      lambda_=FLAGS.lambda_,
      learning_rate=FLAGS.learning_rate)

  accumulator = SequenceAccumulator(length=FLAGS.sequence_length)
  experiment.run_loop(
      agent=agent,
      environment=env,
      accumulator=accumulator,
      seed=FLAGS.seed,
      batch_size=1,
      train_episodes=FLAGS.train_episodes,
      evaluate_every=FLAGS.evaluate_every,
      eval_episodes=FLAGS.eval_episodes,
  )


if __name__ == "__main__":
  app.run(main)
