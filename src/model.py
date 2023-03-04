import haiku as hk
import jax
import jax.numpy as jnp


class DiscreteModel(hk.Module):

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def __call__(self, observation):
        x = hk.Flatten()(observation)
        x = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(512), jax.nn.relu,
            hk.Linear(512), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
        ])(x)

        action_logtis = hk.Linear(self.num_actions)(x)

        x = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(512), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
        ])(x)

        value = hk.Linear(1)(x)
        value = jax.nn.tanh(value)

        return action_logtis, value


class ContinuousModel(hk.Module):

    def __call__(self, observation):
        x = hk.Flatten()(observation)
        x = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(512), jax.nn.relu,
            hk.Linear(512), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
        ])(x)

        mean = hk.Linear(2)(x)
        mean = jax.nn.tanh(mean)
        logstd = hk.Linear(2)(x)

        x = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(512), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
        ])(x)

        value = hk.Linear(1)(x)
        value = jax.nn.tanh(value)

        return jnp.concatenate([mean, logstd], axis=-1), value
