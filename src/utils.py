import chex
import gym
import haiku
import haiku as hk
import jax
import optax
from highway_env.envs import AbstractEnv

from src.model import DiscreteModel


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


def make_agent_state(env: AbstractEnv, model_factory: haiku.Module, lr: float, rng_key: chex.Array, device):
    with jax.default_device(device):
        init, apply = hk.without_apply_rng(hk.transform(model_factory))

        dummy_obs, _ = env.reset()
        params = init(rng=rng_key, x=dummy_obs[None, ...])

        optimizer = optax.adam(learning_rate=lr)
        optimizer_state = optimizer.init(params)

        return apply, optimizer.update, AgentState(params=params, optimizer_state=optimizer_state)
