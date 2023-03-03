import abc
from typing import Callable
import chex
import haiku as hk
import jax
import optax
from jax import numpy as jnp
import rlax

from src.utils import Transition, AgentState, TrainingStats


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

        targets = jax.nn.standardize(transitions.step_return)
        value_loss = rlax.l2_loss(
            predictions=jnp.squeeze(new_values),
            targets=targets
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

        return loss_value, (policy_loss, value_loss, entropy_loss)

    def update(self, transitions: Transition, agent_state: AgentState):
        (loss_value, (policy_loss, value_loss, entropy_loss)), grad = jax.value_and_grad(self.loss, has_aux=True)(
            agent_state.params, transitions
        )
        updates, new_opt_state = self._opt_update_fun(grad, agent_state.optimizer_state)
        new_params = optax.apply_updates(params=agent_state.params, updates=updates)

        training_stats = TrainingStats(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
        )

        agent_state = AgentState(params=new_params, optimizer_state=new_opt_state)
        return agent_state, training_stats

    @abc.abstractmethod
    def act(self, observation: chex.ArrayNumpy, params: hk.Params, rng_key: chex.Array):
        pass
