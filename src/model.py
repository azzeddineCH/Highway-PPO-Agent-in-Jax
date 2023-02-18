import haiku as hk
import jax


class DiscreteModel(hk.Module):

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def __call__(self, observation):
        x = hk.Flatten()(observation)
        x = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
        ])(x)

        action_logtis = hk.Linear(self.num_actions)(x)
        value = hk.Linear(1)(x)

        return action_logtis, value


class ContinuousModel(hk.Module):

    def __call__(self, observation):
        x = hk.Flatten()(observation)
        x = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
        ])(x)

        mu_sigma = hk.Linear(2)(x)
        value = hk.Linear(1)(x)

        return mu_sigma, value
