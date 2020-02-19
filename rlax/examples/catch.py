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
"""A simple Q-learning agent trained to play Catch."""

from absl import app
from absl import flags
from bsuite.experiments.catch import catch
import haiku as hk
from haiku import nets
import jax
from jax.experimental import optix
import jax.numpy as jnp
import rlax

FLAGS = flags.FLAGS

flags.DEFINE_integer("train_episodes", 500, "Number of train episodes.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_float("epsilon", 0.01, "eps-greedy exploration probability.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.01, "Optimizer learning rate.")
flags.DEFINE_integer("seed", 1234, "Random seed.")


def build_network(num_actions: int) -> hk.Transformed:

  def q(obs):
    flatten = lambda x: jnp.reshape(x, (-1,))
    network = hk.Sequential(
        [flatten, nets.MLP([FLAGS.hidden_units, num_actions])])
    return network(obs)

  return hk.transform(q)


def main_loop(unused_arg):
  env = catch.Catch(seed=FLAGS.seed)
  rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))

  # Build and initialize Q-network.
  num_actions = env.action_spec().num_values
  network = build_network(num_actions)
  sample_input = env.observation_spec().generate_value()
  net_params = network.init(next(rng), sample_input)

  # Build and initialize optimizer.
  optimizer = optix.adam(FLAGS.learning_rate)
  opt_state = optimizer.init(net_params)

  @jax.jit
  def policy(net_params, key, obs):
    """Sample action from epsilon-greedy policy."""
    q = network.apply(net_params, obs)
    a = rlax.epsilon_greedy(epsilon=FLAGS.epsilon).sample(key, q)
    return q, a

  @jax.jit
  def eval_policy(net_params, key, obs):
    """Sample action from greedy policy."""
    q = network.apply(net_params, obs)
    return rlax.greedy().sample(key, q)

  @jax.jit
  def update(net_params, opt_state, obs_tm1, a_tm1, r_t, discount_t, q_t):
    """Update network weights wrt Q-learning loss."""

    def q_learning_loss(net_params, obs_tm1, a_tm1, r_t, discount_t, q_t):
      q_tm1 = network.apply(net_params, obs_tm1)
      td_error = rlax.q_learning(q_tm1, a_tm1, r_t, discount_t, q_t)
      return rlax.l2_loss(td_error)

    dloss_dtheta = jax.grad(q_learning_loss)(net_params, obs_tm1, a_tm1, r_t,
                                             discount_t, q_t)
    updates, opt_state = optimizer.update(dloss_dtheta, opt_state)
    net_params = optix.apply_updates(net_params, updates)
    return net_params, opt_state

  print(f"Training agent for {FLAGS.train_episodes} episodes...")
  for _ in range(FLAGS.train_episodes):
    timestep = env.reset()
    obs_tm1 = timestep.observation

    _, a_tm1 = policy(net_params, next(rng), obs_tm1)

    while not timestep.last():
      new_timestep = env.step(int(a_tm1))
      obs_t = new_timestep.observation

      # Sample action from agent policy.
      q_t, a_t = policy(net_params, next(rng), obs_t)

      # Update Q-values.
      r_t = new_timestep.reward
      discount_t = FLAGS.discount_factor * new_timestep.discount
      net_params, opt_state = update(net_params, opt_state, obs_tm1, a_tm1, r_t,
                                     discount_t, q_t)

      timestep = new_timestep
      obs_tm1 = obs_t
      a_tm1 = a_t

  print(f"Evaluating agent for {FLAGS.eval_episodes} episodes...")
  returns = 0.
  for _ in range(FLAGS.eval_episodes):
    timestep = env.reset()
    obs = timestep.observation

    while not timestep.last():
      action = eval_policy(net_params, next(rng), obs)
      timestep = env.step(int(action))
      obs = timestep.observation
      returns += timestep.reward

  avg_returns = returns / FLAGS.eval_episodes
  print(f"Done! Average returns: {avg_returns} (range [-1.0, 1.0])")


if __name__ == "__main__":
  app.run(main_loop)
