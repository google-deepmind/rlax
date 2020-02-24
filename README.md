# RLax

RLax (pronounced "relax") is a library built on top of JAX that exposes
useful building blocks for implementing reinforcement learning agents.

## Installation

RLax can be installed with pip directly from github, with the following command:

`pip install git+git://github.com/deepmind/rlax.git`.

All RLax code may then be just in time compiled for different hardware
(e.g. CPU, GPU, TPU) using `jax.jit`.

In order to run the `examples/` you will also need to install
[haiku](https://github.com/deepmind/haiku) and
[bsuite](https://github.com/deepmind/bsuite).

## Usage

The operations and functions provided are not complete algorithms, but
implementations of reinforcement learning specific mathematical operations that
are needed when building fully-functional agents.

See `examples/catch.py` for an example of using some of the functions in RLax
to implement a Q-learning agent capable of learning to play Catch (a common
unit-test for agent learning in the reinforcement learning literature).

See file-level and function-level doc-strings for the documentation of these
functions and for references to the papers that introduced and/or used them.

## Background

Reinforcement learning studies the problem of a learning system (the *agent*),
which must learn to interact with the universe it is embedded in (the
*environment*).

Agent and environment interact on discrete steps. On each step the agent selects
an *action*, and is provided in return a (partial) snapshot of the state of the
environment (the *observation*), and a scalar feedback signal (the *reward*).

The behaviour of the agent is characterized by a probability distribution over
actions, conditioned on past observations of the environment (the *policy*). The
agents seeks a policy that, from any given step, maximises the discounted
cumulative reward that will be collected from that point onwards (the *return*).

Often the agent policy or the environment dynamics itself are stochastic. In
this case the return is a random variable, and the optimal agent's policy is
typically more precisely specified as a policy that maximises the expectation of
the return (the *value*), under the agent's and environment's stochasticity.

## Reinforcement Learning Algorithms

There are three prototypical families of reinforcement learning algorithms:

1.  those that estimate the value of states and actions, and infer a policy by
    *inspection* (e.g. by selecting the action with highest estimated value)
2.  those that learn a model of the environment (capable of predicting the
    observations and rewards) and infer a policy via *planning*.
3.  those that parameterize a policy that can be directly *executed*,

In any case, policies, values or models are just functions. In deep
reinforcement learning such functions are represented by a neural network.
In this setting, it is common to formulate reinforcement learning updates as
differentiable pseudo-loss functions (analogously to (un-)supervised learning).
Under automatic differentiation, the original update rule is recovered.

Note however, that in particular, the updates are only valid if the input data
is sampled in the correct manner. For example, a policy gradient loss is only
valid if the input trajectory is an unbiased sample from the current policy;
i.e. the data are on-policy. The library cannot check or enforce such
constraints. Links to papers describing how each operation is used are however
provided in the functions' doc-strings.

## Naming Conventions and Developer Guidelines

We define functions and operations for agents interacting with a single stream
of experience. The JAX construct `vmap` can be used to apply these same
functions to batches (e.g. to support *replay* and *parallel* data generation).

Many functions consider policies, actions, rewards, values, in consecutive
timesteps in order to compute their outputs. In this case the suffix `_t` and
`tm1` is often to clarify on which step each input was generated, e.g:

*   `q_tm1`: the action value in the `source` state of a transition.
*   `a_tm1`: the action that was selected in the `source` state.
*   `r_t`: the resulting rewards collected in the `destination` state.
*   `discount_t`: the `discount` associated with a transition.
*   `q_t`: the action values in the `destination` state.

Extensive testing is provided for each function. All tests should also verify
the output of `rlax` functions when compiled to XLA using `jax.jit` and when
performing batch operations using `jax.vmap`.

## Citing RLax

To cite this repository:

```
@software{rlax2020github,
  author = {David Budden and Matteo Hessel and John Quan and Steven Kapturowski},
  title = {{RL}ax: {R}einforcement {L}earning in {JAX}},
  url = {http://github.com/deepmind/rlax},
  version = {0.0.1a0},
  year = {2020},
}
```

In this bibtex entry, the version number is intended to be from
[rlax/\_\_init\_\_.py](https://github.com/deepmind/rlax/blob/master/rlax/__init__.py),
and the year corresponds to the project's open-source release.
