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
"""A simple online Q-learning agent trained to play BSuite's Catch env."""

import collections
from absl import app
from absl import flags
from bsuite.environments import catch
import haiku as hk
from haiku import nets
import jax
import jax.numpy as jnp
import optax
import rlax
from rlax.examples import experiment

ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
Transition = collections.namedtuple("Transition",
                                    "obs_tm1 a_tm1 r_t discount_t obs_t")

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("train_episodes", 500, "Number of train episodes.")
flags.DEFINE_integer("num_hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_float("epsilon", 0.01, "Epsilon-greedy exploration probability.")
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


class TransitionAccumulator:
  """Simple Python accumulator for transitions."""

  def __init__(self):
    self._prev = None
    self._action = None
    self._latest = None

  def push(self, env_output, action):
    self._prev = self._latest
    self._action = action
    self._latest = env_output

  def sample(self, batch_size):
    assert batch_size == 1
    return Transition(self._prev.observation, self._action, self._latest.reward,
                      self._latest.discount, self._latest.observation)

  def is_ready(self, batch_size):
    assert batch_size == 1
    return self._prev is not None


class OnlineQ:
  """An online Q-learning deep RL agent."""

  def __init__(self, observation_spec, action_spec, num_hidden_units, epsilon,
               learning_rate):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._epsilon = epsilon
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

  def _loss(self, params, obs_tm1, a_tm1, r_t, discount_t, obs_t):
    q_tm1 = self._network.apply(params, obs_tm1)
    q_t = self._network.apply(params, obs_t)
    td_error = rlax.q_learning(q_tm1, a_tm1, r_t, discount_t, q_t)
    return rlax.l2_loss(td_error)


def main(unused_arg):
  env = catch.Catch(seed=FLAGS.seed)
  agent = OnlineQ(
      observation_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      num_hidden_units=FLAGS.num_hidden_units,
      epsilon=FLAGS.epsilon,
      learning_rate=FLAGS.learning_rate,
  )

  accumulator = TransitionAccumulator()
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
