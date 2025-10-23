import functools
from typing import Any, Dict, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.training import train_state

from jaxrl2.agents.agent import Agent
from jaxrl2.networks.encoders.networks import Encoder
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.networks.mlp import MLP
from jaxrl2.networks.values.state_action_ensemble import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any = None


def _squeeze_pixels(pixels: jnp.ndarray) -> jnp.ndarray:
    if pixels.ndim == 5 and pixels.shape[-1] == 1:
        return pixels[..., 0]
    return pixels


class ObservationEncoder(nn.Module):
    encoder: nn.Module
    latent_dim: int
    use_bottleneck: bool = True

    @nn.compact
    def __call__(self, observations: FrozenDict, training: bool = False) -> jnp.ndarray:
        observations = FrozenDict(observations)
        pixels = observations["pixels"]
        pixels = _squeeze_pixels(pixels)
        encoded = self.encoder(pixels, training=training)
        if self.use_bottleneck:
            encoded = nn.Dense(self.latent_dim)(encoded)
            encoded = nn.LayerNorm()(encoded)
            encoded = nn.tanh(encoded)
        features = [encoded]
        if "state" in observations:
            state = observations["state"]
            state = jnp.reshape(state, (state.shape[0], -1))
            features.append(state)
        return jnp.concatenate(features, axis=-1)


class FlowMatchingPolicy(nn.Module):
    encoder: ObservationEncoder
    hidden_dims: Sequence[int]
    action_dim: int
    num_integration_steps: int = 16

    def setup(self):
        self.obs_encoder = self.encoder
        self.mlp = MLP((*self.hidden_dims, self.action_dim))

    def vector_field(
        self, observations: FrozenDict, noisy_actions: jnp.ndarray, timesteps: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        obs_features = self.obs_encoder(observations, training=training)
        inputs: Dict[str, jnp.ndarray] = {
            "features": obs_features,
            "noise": noisy_actions,
            "time": timesteps,
        }
        return self.mlp(inputs, training=training)

    def __call__(
        self, observations: FrozenDict, noisy_actions: jnp.ndarray, timesteps: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        return self.vector_field(observations, noisy_actions, timesteps, training=training)

    def sample(
        self,
        observations: FrozenDict,
        noise: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        def euler_step(carry, t):
            actions = carry
            step = (t + 1).astype(jnp.float32) / self.num_integration_steps
            t_batch = jnp.full((actions.shape[0], 1), step)
            velocity = self.vector_field(observations, actions, t_batch, training=training)
            new_actions = actions + velocity / self.num_integration_steps
            return new_actions, new_actions

        init = noise
        steps = jnp.arange(self.num_integration_steps)
        final_actions, _ = jax.lax.scan(euler_step, init, steps)
        return final_actions


class StudentPolicy(nn.Module):
    encoder: ObservationEncoder
    hidden_dims: Sequence[int]
    action_dim: int

    def setup(self):
        self.obs_encoder = self.encoder
        self.mlp = MLP((*self.hidden_dims, self.action_dim))

    def __call__(
        self, observations: FrozenDict, noise: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        obs_features = self.obs_encoder(observations, training=training)
        inputs: Dict[str, jnp.ndarray] = {
            "features": obs_features,
            "noise": noise,
        }
        return self.mlp(inputs, training=training)


class ChunkCritic(nn.Module):
    encoder: ObservationEncoder
    hidden_dims: Sequence[int]
    num_qs: int = 2

    def setup(self):
        self.obs_encoder = self.encoder
        self.critic_head = StateActionEnsemble(self.hidden_dims, num_qs=self.num_qs)

    def __call__(self, observations: FrozenDict, actions: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        obs_features = self.obs_encoder(observations, training=training)
        return self.critic_head(obs_features, actions, training=training)


def _make_encoder(encoder_type: str, encoder_norm: str, use_spatial_softmax: bool, softmax_temperature: float) -> nn.Module:
    if encoder_type == "small":
        return Encoder()
    if encoder_type == "impala":
        return ImpalaEncoder()
    if encoder_type == "impala_small":
        return SmallerImpalaEncoder()
    if encoder_type == "resnet_small":
        return ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
    if encoder_type == "resnet_18_v1":
        return ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
    if encoder_type == "resnet_34_v1":
        return ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
    if encoder_type == "resnet_small_v2":
        return ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
    if encoder_type == "resnet_18_v2":
        return ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
    if encoder_type == "resnet_34_v2":
        return ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


@functools.partial(
    jax.jit,
    static_argnames=("discount", "tau", "distill_weight", "bc_weight"),
)
def _update_jit(
    rng: PRNGKey,
    actor_teacher: TrainState,
    actor_student: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    batch: FrozenDict,
    discount: float,
    tau: float,
    distill_weight: float,
    bc_weight: float,
) -> Tuple[
    PRNGKey,
    TrainState,
    TrainState,
    TrainState,
    Params,
    Dict[str, jnp.ndarray],
]:
    rng, critic_key, teacher_key, teacher_t_key, student_key = jax.random.split(rng, 5)

    actions = jnp.reshape(jnp.asarray(batch["actions"]), (batch["actions"].shape[0], -1))
    rewards = jnp.asarray(batch["rewards"])
    masks = jnp.asarray(batch["masks"])
    discounts = jnp.asarray(batch["discount"])

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        qs = critic.apply_fn({"params": params}, batch["observations"], actions)
        next_noise = jax.random.normal(critic_key, (actions.shape[0], actions.shape[1]))
        next_actions = actor_student.apply_fn(
            {"params": actor_student.params}, batch["next_observations"], next_noise
        )
        next_actions = jax.lax.stop_gradient(next_actions)
        target_qs = critic.apply_fn({"params": target_critic_params}, batch["next_observations"], next_actions)
        target_q = jnp.min(target_qs, axis=0)
        target = rewards + discounts * masks * target_q
        td_error = qs - target[None, :]
        loss = 0.5 * jnp.mean(td_error ** 2)
        info = {
            "critic_q": jnp.mean(jnp.min(qs, axis=0)),
            "critic_target": jnp.mean(target),
        }
        return loss, info

    (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=critic_grads)
    new_target_params = soft_target_update(new_critic.params, target_critic_params, tau)

    def teacher_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        base_noise = jax.random.normal(teacher_key, (actions.shape[0], actions.shape[1]))
        times = jax.random.uniform(teacher_t_key, (actions.shape[0], 1))
        x_t = (1.0 - times) * base_noise + times * actions
        velocities = actions - base_noise
        pred_vel = actor_teacher.apply_fn({"params": params}, batch["observations"], x_t, times)
        loss = 0.5 * jnp.mean((pred_vel - velocities) ** 2)
        return loss * bc_weight, {"bc_flow_loss": loss}

    (teacher_loss, teacher_info), teacher_grads = jax.value_and_grad(teacher_loss_fn, has_aux=True)(actor_teacher.params)
    new_teacher = actor_teacher.apply_gradients(grads=teacher_grads)

    def student_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        noise = jax.random.normal(student_key, (actions.shape[0], actions.shape[1]))
        teacher_samples = actor_teacher.apply_fn(
            {"params": new_teacher.params},
            batch["observations"],
            noise,
            method=FlowMatchingPolicy.sample,
        )
        teacher_samples = jax.lax.stop_gradient(teacher_samples)
        student_samples = actor_student.apply_fn({"params": params}, batch["observations"], noise)
        distill = 0.5 * jnp.mean((student_samples - teacher_samples) ** 2)
        qs = new_critic.apply_fn({"params": new_critic.params}, batch["observations"], student_samples)
        q_val = jnp.min(qs, axis=0)
        q_loss = -jnp.mean(q_val)
        loss = distill_weight * distill + q_loss
        info = {
            "distill_loss": distill,
            "q_loss": -q_loss,
        }
        return loss, info

    (student_loss, student_info), student_grads = jax.value_and_grad(student_loss_fn, has_aux=True)(actor_student.params)
    new_student = actor_student.apply_gradients(grads=student_grads)

    info = {
        "critic_loss": critic_loss,
        "actor_teacher_loss": teacher_loss,
        "actor_student_loss": student_loss,
        **critic_info,
        **teacher_info,
        **student_info,
    }

    return rng, new_teacher, new_student, new_critic, new_target_params, info


def _repeat_observations(observations: Dict[str, np.ndarray], count: int) -> Dict[str, np.ndarray]:
    def _repeat(x: np.ndarray) -> np.ndarray:
        return np.repeat(x, count, axis=0)

    return {k: _repeat(v) if isinstance(v, np.ndarray) else _repeat_observations(v, count) for k, v in observations.items()}


class NoiseChunkFQLAgent(Agent):
    def __init__(
        self,
        seed: int,
        observations: Dict[str, np.ndarray],
        actions: np.ndarray,
        *,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        flow_lr: float = 3e-4,
        student_hidden_dims: Sequence[int] = (256, 256),
        flow_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256),
        latent_dim: int = 128,
        discount: float = 0.99,
        tau: float = 0.005,
        distill_weight: float = 10.0,
        bc_weight: float = 1.0,
        num_qs: int = 2,
        flow_num_integration_steps: int = 16,
        use_bottleneck: bool = True,
        encoder_type: str = "resnet_18_v1",
        encoder_norm: str = "group",
        use_spatial_softmax: bool = True,
        softmax_temperature: float = 1.0,
        best_of_n: int = 0,
        use_best_of_n: bool = True,
        use_student_policy: bool = True,
        num_cameras: int = 1,
    ):  
        self.num_cameras = num_cameras
        self.discount = discount
        self.tau = tau
        self.distill_weight = distill_weight
        self.bc_weight = bc_weight
        self.best_of_n = best_of_n
        self.use_best_of_n = use_best_of_n and best_of_n > 0
        self.use_student_policy = use_student_policy
        self.flow_num_integration_steps = flow_num_integration_steps

        observations = jax.tree_map(lambda x: jnp.asarray(x), observations)
        actions = jnp.asarray(actions)

        self.action_chunk_shape = actions.shape[-2:]
        self.full_action_dim = int(np.prod(self.action_chunk_shape))

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, flow_key = jax.random.split(rng, 4)

        encoder_def = _make_encoder(encoder_type, encoder_norm, use_spatial_softmax, softmax_temperature)

        obs_encoder_def = ObservationEncoder(encoder=encoder_def, latent_dim=latent_dim, use_bottleneck=use_bottleneck)

        teacher_def = FlowMatchingPolicy(
            encoder=obs_encoder_def,
            hidden_dims=flow_hidden_dims,
            action_dim=self.full_action_dim,
            num_integration_steps=flow_num_integration_steps,
        )
        teacher_params = teacher_def.init(flow_key, observations, jnp.zeros((1, self.full_action_dim)), jnp.zeros((1, 1)))
        self.actor_teacher = TrainState.create(
            apply_fn=teacher_def.apply,
            params=teacher_params["params"],
            tx=optax.adam(flow_lr),
        )

        student_def = StudentPolicy(
            encoder=obs_encoder_def,
            hidden_dims=student_hidden_dims,
            action_dim=self.full_action_dim,
        )
        student_params = student_def.init(actor_key, observations, jnp.zeros((1, self.full_action_dim)))
        self.actor_student = TrainState.create(
            apply_fn=student_def.apply,
            params=student_params["params"],
            tx=optax.adam(actor_lr),
        )

        critic_def = ChunkCritic(
            encoder=obs_encoder_def,
            hidden_dims=critic_hidden_dims,
            num_qs=num_qs,
        )
        critic_params = critic_def.init(critic_key, observations, jnp.zeros((1, self.full_action_dim)))
        self._critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params["params"],
            tx=optax.adam(critic_lr),
        )

        self._target_critic_params = critic_params["params"].copy()
        self._rng = rng
        self._actor = self.actor_student
        self.actor_teacher_def = teacher_def
        self.actor_student_def = student_def
        self.critic_def = critic_def

    def sample_actions(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        obs = jax.tree_map(lambda x: jnp.asarray(x), observations)
        batch_size = next(iter(obs.values())).shape[0]
        self._rng, rng = jax.random.split(self._rng)
        if self.use_best_of_n:
            rng, noise_key = jax.random.split(rng)
            noise = jax.random.normal(noise_key, (self.best_of_n * batch_size, self.full_action_dim))
            obs_rep = _repeat_observations(jax.tree_map(np.asarray, observations), self.best_of_n)
            obs_rep = jax.tree_map(lambda x: jnp.asarray(x), obs_rep)
            teacher_samples = self.actor_teacher.apply_fn(
                {"params": self.actor_teacher.params},
                obs_rep,
                noise,
                method=FlowMatchingPolicy.sample,
            )
            qs = self._critic.apply_fn({"params": self._critic.params}, obs_rep, teacher_samples)
            q_vals = jnp.min(qs, axis=0)
            q_vals = q_vals.reshape((batch_size, self.best_of_n))
            best_indices = jnp.argmax(q_vals, axis=1)
            teacher_samples = teacher_samples.reshape((batch_size, self.best_of_n, self.full_action_dim))
            selected = teacher_samples[jnp.arange(batch_size), best_indices]
            actions = selected
        elif self.use_student_policy:
            rng, noise_key = jax.random.split(rng)
            noise = jax.random.normal(noise_key, (batch_size, self.full_action_dim))
            actions = self.actor_student.apply_fn({"params": self.actor_student.params}, obs, noise)
        else:
            rng, noise_key = jax.random.split(rng)
            noise = jax.random.normal(noise_key, (batch_size, self.full_action_dim))
            actions = self.actor_teacher.apply_fn(
                {"params": self.actor_teacher.params},
                obs,
                noise,
                method=FlowMatchingPolicy.sample,
            )
        actions = jax.device_get(actions)
        actions = actions.reshape((batch_size, *self.action_chunk_shape))
        return np.asarray(actions)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        self._rng, new_teacher, new_student, new_critic, new_target_params, info = _update_jit(
            self._rng,
            self.actor_teacher,
            self.actor_student,
            self._critic,
            self._target_critic_params,
            batch,
            self.discount,
            self.tau,
            self.distill_weight,
            self.bc_weight,
        )
        self.actor_teacher = new_teacher
        self.actor_student = new_student
        self._critic = new_critic
        self._actor = new_student
        self._target_critic_params = new_target_params
        return jax.tree_map(lambda x: np.asarray(x), info)

    @property
    def _save_dict(self):
        return {
            "actor_teacher": self.actor_teacher,
            "actor_student": self.actor_student,
            "critic": self._critic,
            "target_critic_params": self._target_critic_params,
            "rng": self._rng,
        }

