import abc
from typing import Callable
import chex
import haiku as hk
import jax
import optax
from jax import numpy as jnp
import rlax
import distrax

from src.utils import Transition, AgentState


class BasePPOPolicy(abc.ABC):

    def __init__(
            self,
            apply_fun: Callable,
            opt_update_fun: Callable,
            policy_clip: float,
            entropy_coefficient: float,
            value_coefficient: float,
            apply_value_clipping: bool = False,
            value_clip: int = 0.2
    ):
        self.apply_fun = apply_fun
        self._opt_update_fun = opt_update_fun
        self.policy_clip = policy_clip
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self._apply_value_clipping = apply_value_clipping
        self._value_clip = value_clip

    def get_importance_sampling_ratios(self, transitions, params):
        raise NotImplementedError()

    def get_entropy_loss(self, transitions, logits):
        raise NotImplementedError()

    def loss(self, params: hk.Params, transitions: Transition):
        ratio, new_logits, new_values = self.get_importance_sampling_ratios(transitions, params)

        advantage = jax.nn.standardize(transitions.advantage)

        policy_loss = rlax.clipped_surrogate_pg_loss(
            prob_ratios_t=ratio,
            adv_t=advantage,
            epsilon=self.policy_clip
        )

        value_loss = rlax.l2_loss(
            predictions=jnp.squeeze(new_values),
            targets=transitions.step_return
        )

        if self._apply_value_clipping:
            vf_clipped = transitions.state_value + jnp.clip(
                new_values - transitions.state_value,
                -self._value_clip,
                self._value_clip
            )

            value_loss_2 = rlax.l2_loss(
                predictions=jnp.squeeze(vf_clipped),
                targets=transitions.step_return
            )

            value_loss = jnp.min(value_loss, value_loss_2)

        value_loss = jnp.mean(value_loss)

        entropy_loss = self.get_entropy_loss(transitions, new_logits)

        loss_value = policy_loss + self.value_coefficient * value_loss + self.entropy_coefficient * entropy_loss

        return loss_value

    def update(self, transitions: Transition, agent_state: AgentState):
        loss_value, grad = jax.value_and_grad(self.loss)(agent_state.params, transitions)
        updates, new_opt_state = self._opt_update_fun(grad, agent_state.optimizer_state)
        new_params = optax.apply_updates(params=agent_state.params, updates=updates)

        return AgentState(params=new_params, optimizer_state=new_opt_state)

    @abc.abstractmethod
    def act(self, observation: chex.ArrayNumpy, params: hk.Params, rng_key: chex.Array):
        pass


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

    def act(self, observation: chex.ArrayNumpy, params: hk.Params, rng_key: chex.Array, explore=True):
        """
        sample an action following the agent policy
        """
        logits, state_value = self.apply_fun(params, observation[None, ...])
        logits, state_value = jnp.squeeze(logits), jnp.squeeze(state_value)

        distribution = distrax.Categorical(logits=logits)
        if explore:
            action, log_pi = distribution.sample_and_log_prob(seed=rng_key)
        else:
            action = distribution.sample(seed=rng_key)  # TODO: fix
            log_pi = None

        return action, logits, log_pi, state_value


class ContinuousPPOPolicy(BasePPOPolicy):

    def get_importance_sampling_ratios(self, transitions, params):
        new_mu_sigma, new_values = self.apply_fun(params, transitions.observation)

        new_mu, new_sigma = jnp.split(new_mu_sigma, indices_or_sections=[1, 1], axis=-1)
        new_distribution = distrax.Transformed(distrax.MultivariateNormalDiag(loc=new_mu, scale_diag=new_sigma),
                                               distrax.Tanh())

        new_log_pi = new_distribution.log_prob(transitions.action)
        ratio = jnp.divide(new_log_pi, transitions.log_pi)

        return ratio, new_mu_sigma, new_values

    def get_entropy_loss(self, transitions, logits):
        mu, sigma = jnp.split(logits, indices_or_sections=[1, 1], axis=-1)
        distribution = distrax.Transformed(distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma), distrax.Tanh())

        return -jnp.mean(distribution.entropy())

    def act(self, observation: chex.ArrayNumpy, params: hk.Params, rng_key: chex.Array, explore=True):
        mu_sigma, state_value = self.apply_fun(params, observation[None, ...])
        mu_sigma, state_value = jnp.squeeze(mu_sigma), jnp.squeeze(state_value)
        mu, sigma = jnp.split(mu_sigma, indices_or_sections=[1, 1])
        distribution = distrax.Normal(loc=mu, scale=sigma)

        if explore:
            action, log_pi = distribution.sample_and_log_prob(seed=rng_key)
        else:
            action = distribution.mean()
            log_pi = distribution.log_prob(action)

        return action, mu_sigma, log_pi, state_value
