import time
from collections import deque
from typing import Callable
import jax
from jax import numpy as jnp
from tensorboardX import SummaryWriter

from src.buffer import RolloutBuffer
from src.policy import BasePPOPolicy
from src.utils import AgentState, Transition


class PPOTrainer:
    """
    The PPO trainer handles the evaluation and update loop of the PPO algorithm
    """

    def __init__(
            self,
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
    ):

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

        self._env = env_factory()
        [*_, self._gpu_device], [*_, self._cpu_device] = jax.devices("cpu"), jax.devices("cpu")

    def evaluate(self, env, policy: BasePPOPolicy, agent_state: AgentState, seed: jnp.array, device):

        with jax.default_device(device):
            agent_state = jax.device_put(agent_state, device)
            buffer = RolloutBuffer(
                discount_gamma=self._discount_gamma,
                gae_lambda=self._gae_lambda,
                batch_size=self._batch_size,
                num_batches=self._num_batches,
                use_gae=self._use_gae
            )
            episode_rewards = []
            while len(buffer) < self._batch_size * self._num_batches:

                observation, _ = env.reset()

                episode_reward = 0
                done = False
                i = 0
                while not done:
                    seed, step_seed = jax.random.split(seed)
                    action, logits, log_pi, state_value = policy.act(observation, agent_state.params, step_seed)
                    next_observation, reward, done, truncated, info = env.step(int(jax.device_get(action)))
                    episode_reward += reward
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
                    observation = next_observation
                    i += 1

                    buffer.add(transition)
                episode_rewards.append(episode_reward)
            return buffer, episode_rewards

    def train(self, transitions, policy: BasePPOPolicy, agent_state: AgentState, device):

        @jax.jit
        def _train(agent_state, transitions):
            return jax.lax.fori_loop(
                0, self._num_sgd_iteration, lambda _, state: _train_single_iteration(state, transitions),
                init_val=agent_state)

        def _train_single_iteration(agent_state, transitions):
            agent_state, _ = jax.lax.scan(_train_single_batch, init=agent_state, xs=transitions)
            return agent_state

        def _train_single_batch(agent_state, transitions):
            agent_state = policy.update(transitions, agent_state)
            return agent_state, None

        with jax.default_device(device):
            transitions = jax.device_put(transitions, device)
            agent_state = jax.device_put(agent_state, device)
            agent_state = _train(agent_state, transitions)

        return agent_state

    def run(self, rng_key, policy_class):

        writer = SummaryWriter()

        rng_key, params_seed = jax.random.split(rng_key)
        apply_fun, opt_update_fun, agent_state = self._agent_state_factory(
            env=self._env,
            lr=self._learning_rate,
            rng_key=params_seed,
            device=self._cpu_device
        )

        policy = policy_class(
            apply_fun=apply_fun,
            opt_update_fun=opt_update_fun,
            policy_clip=self._policy_clip,
            entropy_coefficient=self._entropy_coefficient,
            value_coefficient=self._value_coefficient,
            apply_value_clipping=self._apply_value_clipping,
            value_clip=self._value_clip
        )

        episode_return_queue = deque(maxlen=100)
        for i in range(self._num_iteration):
            t0 = time.time()

            rng_key, iteration_seed = jax.random.split(rng_key)
            buffer, episode_rewards = self.evaluate(self._env, policy, agent_state, iteration_seed, self._cpu_device)
            transitions = buffer.get(iteration_seed)
            agent_state = self.train(transitions, policy, agent_state, self._gpu_device)
            episode_return_queue.extend(episode_rewards)

            print(f"iteration {i + 1} took {time.time() - t0}s")
            writer.add_scalar('ppo/episode_reward_mean', sum(episode_return_queue) / len(episode_return_queue), i)

        writer.close()
