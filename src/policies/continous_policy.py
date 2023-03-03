import chex
import haiku as hk
from jax import numpy as jnp
import distrax
from src.policies.base_policy import BasePPOPolicy

MIN_LOG_NN_OUTPUT = -5
MAX_LOG_NN_OUTPUT = 2


class ContinuousPPOPolicy(BasePPOPolicy):

    def get_importance_sampling_ratios(self, transitions, params):
        mean_logstd, new_values = self.apply_fun(params, transitions.observation)

        mean, log_std = jnp.split(mean_logstd, indices_or_sections=2, axis=-1)
        log_std = jnp.clip(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)

        throttle_distribution = distrax.Transformed(
            distrax.Normal(loc=mean[:, 0, None], scale=jnp.exp(log_std[:, 0, None])),
            distrax.Tanh()
        )

        steering_distribution = distrax.Transformed(
            distrax.Normal(loc=mean[:, 1, None], scale=jnp.exp(log_std[:, 1, None])),
            distrax.Tanh()
        )

        throttle_log_pi = throttle_distribution.log_prob(transitions.action[..., 0, None])
        steering_log_pi = steering_distribution.log_prob(transitions.action[..., 1, None])

        new_log_pi = throttle_log_pi + steering_log_pi
        ratio = jnp.divide(jnp.squeeze(new_log_pi), transitions.log_pi)

        return ratio, mean_logstd, new_values

    def get_entropy_loss(self, transitions, logits):

        mean, log_std = jnp.split(logits, indices_or_sections=2, axis=-1)
        log_std = jnp.clip(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)

        throttle_entropy = jnp.sum(log_std[:, 0, None] + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1).mean()
        steering_entropy = jnp.sum(log_std[:, 1, None] + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1).mean()

        return -(throttle_entropy + steering_entropy)

    def act(self, observation: chex.ArrayNumpy, params: hk.Params, rng_key: chex.Array, explore=True):
        mean_logstd, state_value = self.apply_fun(params, observation[None, ...])
        mean_logstd, state_value = jnp.squeeze(mean_logstd), jnp.squeeze(state_value)

        mean, log_std = jnp.split(mean_logstd, indices_or_sections=2)
        log_std = jnp.clip(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)

        throttle_distribution = distrax.Transformed(
            distrax.Normal(loc=mean[0], scale=jnp.exp(log_std[0])),
            distrax.Tanh()
        )

        steering_distribution = distrax.Transformed(
            distrax.Normal(loc=mean[1], scale=jnp.exp(log_std[1])),
            distrax.Tanh()
        )
        if explore:
            throttle_action, throttle_log_pi = throttle_distribution.sample_and_log_prob(seed=rng_key)
            steering_action, steering_log_pi = steering_distribution.sample_and_log_prob(seed=rng_key)
        else:
            throttle_action = throttle_distribution.mean()
            steering_action = steering_distribution.mean()

            throttle_log_pi = throttle_distribution.log_prob(throttle_action)
            steering_log_pi = steering_distribution.log_prob(steering_action)

        log_pi = throttle_log_pi + steering_log_pi
        return jnp.array([throttle_action, steering_action]), mean_logstd, log_pi, state_value
