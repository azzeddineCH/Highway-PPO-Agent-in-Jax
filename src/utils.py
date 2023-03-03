from collections import defaultdict

import chex
import haiku
import haiku as hk
import jax
import optax
from highway_env.envs import AbstractEnv


@chex.dataclass
class Transition:
    observation: chex.ArrayDevice
    action: int
    reward: float
    done: int
    next_observation: chex.ArrayDevice
    logits: chex.ArrayDevice
    log_pi: float
    state_value: chex.ArrayDevice
    advantage: float = None
    step_return: float = None


@chex.dataclass
class AgentState:
    params: hk.Params
    optimizer_state: optax.OptState


@chex.dataclass
class EpisodeStats:
    reward: int = 0
    num_steps: int = 0
    speed: float = 0
    num_crashes: int = 0
    collision_reward: float = 0
    right_lane_reward: float = 0
    high_speed_reward: float = 0
    on_road_reward: float = 0


@chex.dataclass
class TrainingStats:
    value_loss: float
    policy_loss: float
    entropy_loss: float


class EpisodeStatsBuilder:

    def __init__(self):
        self._stats = EpisodeStats()

    def add(self, **kwargs):
        self._stats.num_steps += 1
        self._stats.reward += float(kwargs["reward"])
        self._stats.speed += float(kwargs["speed"])
        self._stats.num_crashes += int(kwargs["crashed"])
        self._stats.collision_reward += float(kwargs["rewards"]["collision_reward"])
        self._stats.right_lane_reward += float(kwargs["rewards"]["right_lane_reward"])
        self._stats.high_speed_reward += float(kwargs["rewards"]["high_speed_reward"])
        self._stats.on_road_reward += float(kwargs["rewards"]["on_road_reward"])

    def get(self):
        self._stats.speed /= self._stats.num_steps
        return self._stats


def make_agent_state(env: AbstractEnv, model_factory: haiku.Module, lr: float, rng_key: chex.Array, device):
    with jax.default_device(device):
        init, apply = hk.without_apply_rng(hk.transform(model_factory))

        dummy_obs, _ = env.reset()
        params = init(rng=rng_key, x=dummy_obs[None, ...])

        optimizer = optax.adam(learning_rate=lr)
        optimizer_state = optimizer.init(params)

        return apply, optimizer.update, AgentState(params=params, optimizer_state=optimizer_state)
