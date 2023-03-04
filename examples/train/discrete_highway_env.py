import chex
import jax
import gym
from highway_env.envs import AbstractEnv

from src.model import DiscreteModel
from src.policies.discrete_policy import DiscretePPOPolicy
from src.trainer import PPOTrainer
from src.utils import make_agent_state


def make_discrete_highway_env():
    observation_type = "Kinematics"
    action_type = "DiscreteMetaAction"

    env = gym.make("highway-fast-v0")
    env.config["observation"]["type"] = observation_type
    env.config["action"]["type"] = action_type
    env.config["vehicles_count"] = 25
    env.config["vehicles_density"] = 1
    env.config["duration"] = 200
    return env


def _make_agent_state(env: AbstractEnv, lr: float, rng_key: chex.Array, device):
    action_space = env.action_type.space()
    model_factory = lambda x: DiscreteModel(num_actions=action_space.n)(x)
    return make_agent_state(env, model_factory=model_factory, lr=lr, rng_key=rng_key, device=device)


if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(10)
    PPOTrainer(
        policy_class=DiscretePPOPolicy,
        env_factory=make_discrete_highway_env,
        agent_state_factory=_make_agent_state,
        num_iteration=1000,
        num_sgd_iteration=10,
        learning_rate=0.003,
        policy_clip=0.2,
        entropy_coefficient=0.01,
        value_coefficient=1.0,
        batch_size=128,
        num_batches=4,
        discount_gamma=0.99,
        gae_lambda=0.95,
        value_clip=0.1,
        use_gae=True,
        apply_value_clipping=False,
        training_device="cpu"
    ).train(rng_key)
