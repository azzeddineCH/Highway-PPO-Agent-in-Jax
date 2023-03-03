import pickle
from pathlib import Path
from collections import deque
from typing import Callable
import jax
from jax import numpy as jnp
from tensorboardX import SummaryWriter

from src.buffer import RolloutBuffer
from src.policies.base_policy import BasePPOPolicy
from src.utils import AgentState, Transition, EpisodeStatsBuilder


class PPOTrainer:
    """
    The PPO trainer handles the evaluation and update loop of the PPO algorithm
    """

    def __init__(
            self,
            policy_class,
            env_factory: Callable,
            agent_state_factory: Callable,
            num_iteration: int,
            num_sgd_iteration: int,
            learning_rate: float,
            policy_clip: float,
            entropy_coefficient: float,
            value_coefficient: float,
            batch_size: int,
            num_batches: int,
            discount_gamma: float,
            gae_lambda: float,
            value_clip: float,
            use_gae: bool,
            apply_value_clipping: bool,
            explore=True,
            evaluation_device: str = "cpu",
            training_device: str = "gpu",
            checkpoint_freq: int = 25,
            checkpoint_dir: str = "checkpoints",
    ):

        self._policy_cls = policy_class
        self._num_sgd_iteration = num_sgd_iteration
        self._num_iteration = num_iteration
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._discount_gamma = discount_gamma
        self._gae_lambda = gae_lambda
        self._policy_clip = policy_clip
        self._entropy_coefficient = entropy_coefficient
        self._value_coefficient = value_coefficient
        self._agent_state_factory = agent_state_factory
        self._learning_rate = learning_rate
        self._use_gae = use_gae
        self._apply_value_clipping = apply_value_clipping
        self._value_clip = value_clip
        self._explore = explore
        self._checkpoint_freq = checkpoint_freq
        self._checkpoint_dir = checkpoint_dir

        self._env = env_factory()
        self._buffer = RolloutBuffer(
            discount_gamma=self._discount_gamma,
            gae_lambda=self._gae_lambda,
            batch_size=self._batch_size,
            num_batches=self._num_batches,
            use_gae=self._use_gae
        )
        [*_, self._train_device], [*_, self._eval_device] = jax.devices(training_device), jax.devices(evaluation_device)

    def _log_results(self, iter, episodes_stats, training_stats, writer):
        episodes_stats = jax.tree_util.tree_map(lambda *t: jnp.asarray(t), *list(episodes_stats))
        episodes_stats = jax.tree_util.tree_map(lambda t: (jnp.min(t), jnp.mean(t), jnp.max(t)), episodes_stats)
        for key in episodes_stats:
            min_value, mean_value, max_value = episodes_stats[key]
            writer.add_scalar(f'ppo/eval/{key}_mean', mean_value, iter)
            writer.add_scalar(f'ppo/eval/{key}_max', max_value, iter)
            writer.add_scalar(f'ppo/eval/{key}_min', min_value, iter)

        for key in training_stats:
            writer.add_scalar(f'ppo/learner/{key}', training_stats[key], iter)

        print(f"iteration {iter + 1}...")

    def _save_checkpoint(self, key, agent_state):
        checkpoint_dir = Path(self._checkpoint_dir).joinpath(str(key))
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        with open(checkpoint_dir.joinpath("state.npy"), "wb") as f:
            for x in jax.tree_util.tree_leaves(agent_state):
                jnp.save(f, x, allow_pickle=False)

        tree_struct = jax.tree_map(lambda _: 0, agent_state)
        with open(checkpoint_dir.joinpath("struct.pkl"), "wb") as f:
            pickle.dump(tree_struct, f)

    def _load_checkpoint(self, checkpoint_dir):

        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir.joinpath("struct.pkl"), "rb") as f:
            tree_struct = pickle.load(f)

        leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
        with open(checkpoint_dir.joinpath("state.npy"), "rb") as f:
            flat_state = [jnp.load(f) for _ in leaves]

        return jax.tree_util.tree_unflatten(treedef, flat_state)

    def _run_policy_evaluation(self, policy: BasePPOPolicy, agent_state: AgentState, seed: jnp.array, device):

        with jax.default_device(device):
            agent_state = jax.device_put(agent_state, device)
            self._buffer.reset()
            episodes_stats = []
            while len(self._buffer) < self._batch_size * self._num_batches:
                seed, episode_seed = jax.random.split(seed)
                transitions, episode_stats = self.run_single_episode(policy, agent_state, episode_seed, self._explore)
                self._buffer.add(transitions)
                episodes_stats.append(episode_stats)

            return self._buffer, episodes_stats

    def _run_policy_update(self, transitions, policy: BasePPOPolicy, agent_state: AgentState, device):

        @jax.jit
        def _train(agent_state, transitions):
            agent_state, training_stats = jax.lax.scan(
                f=lambda a, _: _train_single_iteration(a, transitions),
                init=agent_state,
                xs=jnp.arange(self._num_sgd_iteration)
            )
            training_stats = jax.tree_util.tree_map(lambda t: jnp.mean(t, axis=-1), training_stats)
            return agent_state, training_stats

        def _train_single_iteration(agent_state, transitions):
            agent_state, iteration_training_stats = jax.lax.scan(_train_single_batch, init=agent_state, xs=transitions)
            iteration_training_stats = jax.tree_util.tree_map(lambda t: jnp.mean(t, axis=-1), iteration_training_stats)
            return agent_state, iteration_training_stats

        def _train_single_batch(agent_state, transitions):
            agent_state, batch_training_stats = policy.update(transitions, agent_state)
            return agent_state, batch_training_stats

        with jax.default_device(device):
            transitions = jax.device_put(transitions, device)
            agent_state = jax.device_put(agent_state, device)
            agent_state, training_stats = _train(agent_state, transitions)

        return agent_state, training_stats

    def run_single_episode(self, policy, agent_state, seed, explore, render=False):

        observation, _ = self._env.reset()
        done = False
        transitions = []
        episode_stats_builder = EpisodeStatsBuilder()
        while not done:
            seed, step_seed = jax.random.split(seed)
            action, logits, log_pi, state_value = policy.act(observation, agent_state.params, step_seed, explore=explore)

            action = jax.device_get(action)
            if action.shape[-1] == 2:
                next_observation, reward, done, truncated, info = self._env.step(action)
            else:
                next_observation, reward, done, truncated, info = self._env.step(int(action))

            episode_stats_builder.add(**info, reward=reward)

            transition = Transition(
                observation=jnp.array(observation),
                action=jnp.array(action),
                reward=jnp.array(reward),
                done=jnp.array(done),
                next_observation=jnp.array(next_observation),
                logits=jnp.array(logits),
                log_pi=log_pi,
                state_value=state_value
            )

            transitions.append(transition)
            episode_stats_builder.add(**info, reward=reward)
            observation = next_observation
            if render:
                self._env.render()

        return transitions, episode_stats_builder.get()

    def run_rollouts(self, rng_key, num_episodes, render, checkpoint_dir=None):

        rng_key, params_seed = jax.random.split(rng_key)

        apply_fun, opt_update_fun, random_agent_state = self._agent_state_factory(
            env=self._env,
            lr=self._learning_rate,
            rng_key=params_seed,
            device=self._eval_device
        )

        if checkpoint_dir:
            agent_state = self._load_checkpoint(checkpoint_dir)
        else:
            agent_state = random_agent_state

        policy = self._policy_cls(
            apply_fun=apply_fun,
            opt_update_fun=opt_update_fun,
            policy_clip=self._policy_clip,
            entropy_coefficient=self._entropy_coefficient,
            value_coefficient=self._value_coefficient,
            apply_value_clipping=self._apply_value_clipping,
            value_clip=self._value_clip
        )

        for _ in range(num_episodes):
            rng_key, episode_seed = jax.random.split(rng_key)
            self.run_single_episode(policy, agent_state, render=render, seed=episode_seed, explore=False)

    def train(self, rng_key):

        writer = SummaryWriter()

        rng_key, params_seed = jax.random.split(rng_key)
        apply_fun, opt_update_fun, agent_state = self._agent_state_factory(
            env=self._env,
            lr=self._learning_rate,
            rng_key=params_seed,
            device=self._eval_device
        )

        policy = self._policy_cls(
            apply_fun=apply_fun,
            opt_update_fun=opt_update_fun,
            policy_clip=self._policy_clip,
            entropy_coefficient=self._entropy_coefficient,
            value_coefficient=self._value_coefficient,
            apply_value_clipping=self._apply_value_clipping,
            value_clip=self._value_clip
        )

        episode_stats_queue = deque(maxlen=100)
        for iter in range(self._num_iteration):
            rng_key, iteration_seed = jax.random.split(rng_key)
            buffer, episodes_stats = self._run_policy_evaluation(
                policy,
                agent_state,
                seed=iteration_seed,
                device=self._eval_device
            )

            transitions = buffer.get(iteration_seed)

            agent_state, training_stats = self._run_policy_update(
                transitions,
                policy,
                agent_state,
                self._train_device
            )

            episode_stats_queue.extend(episodes_stats)
            self._log_results(iter, episode_stats_queue, training_stats, writer)

            if (iter + 1) % self._checkpoint_freq == 0:
                print(f"Saving checkpoint {iter + 1}...")
                self._save_checkpoint(key=iter, agent_state=agent_state)

        writer.close()
