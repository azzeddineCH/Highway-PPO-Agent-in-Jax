from functools import partial
from typing import List
import jax
from jax import numpy as jnp
import rlax

from src.utils import Transition


class RolloutBuffer:

    def __init__(self, batch_size: int, num_batches: int, discount_gamma: int, gae_lambda: int, use_gae: bool):
        self._transitions: List[Transition] = []
        self._discount_gamma = discount_gamma
        self._gae_lambda = gae_lambda
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._use_gae = use_gae

    def reset(self):
        self._transitions = []

    def add(self, transition: Transition):
        if isinstance(transition, list):
            self._transitions.extend(transition)
        else:
            self._transitions.append(transition)

    @partial(jax.jit, static_argnums=0)
    def preprocess(self, rng_key, transitions):
        transitions = jax.tree_util.tree_map(lambda *t: jnp.asarray(t), *transitions)
        transitions.state_value = jax.lax.stop_gradient(transitions.state_value)

        discount_t = (1 - transitions.done) * self._discount_gamma

        if self._use_gae:
            extended_values = jnp.append(transitions.state_value, 0)
            transitions.advantage = rlax.truncated_generalized_advantage_estimation(
                r_t=transitions.reward,
                discount_t=discount_t,
                lambda_=self._gae_lambda,
                values=extended_values,
                stop_target_gradients=True
            )
            transitions.step_return = transitions.advantage + transitions.state_value
        else:
            transitions.step_return = rlax.discounted_returns(
                r_t=transitions.reward,
                discount_t=discount_t,
                v_t=0.0,
            )

            transitions.advantage = transitions.step_return - transitions.state_value

        transitions = jax.tree_util.tree_map(lambda t: jax.random.permutation(key=rng_key, x=t), transitions)
        transitions = jax.tree_util.tree_map(
            lambda t:
            t[:self._batch_size * self._num_batches].reshape(self._num_batches, self._batch_size, -1).squeeze(),
            transitions
        )

        return transitions

    def get(self, rng_key):
        return self.preprocess(rng_key, self._transitions)

    def __len__(self):
        return len(self._transitions)
