from functools import partial

import chex
import haiku as hk
import jax
from jax import numpy as jnp
import distrax

from src.policies.base_policy import BasePPOPolicy


class DiscretePPOPolicy(BasePPOPolicy):

    def get_importance_sampling_ratios(self, transitions, params):
        new_logits, new_values = self.apply_fun(params, transitions.observation)
        ratio = distrax.importance_sampling_ratios(
            target_dist=distrax.Categorical(new_logits),
            sampling_dist=distrax.Categorical(transitions.logits),
            event=transitions.action
        )
        return ratio, new_logits, new_values

    def get_entropy_loss(self, transitions, logits):
        return -jnp.mean(distrax.Softmax(logits).entropy())

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: chex.ArrayNumpy, params: hk.Params, rng_key: chex.Array, explore=True):
        def sample(rng_key):
            return distribution.sample_and_log_prob(seed=rng_key)

        def deterministic_sample(*_):
            action = distribution.mode()
            log_pi = distribution.log_prob(action)

            return action, log_pi

        logits, state_value = self.apply_fun(params, observation[None, ...])
        logits, state_value = jnp.squeeze(logits), jnp.squeeze(state_value)

        distribution = distrax.Categorical(logits=logits)
        action, log_pi = jax.lax.cond(explore, true_fun=sample, false_fun=deterministic_sample, operand=rng_key)

        return action, logits, log_pi, state_value
