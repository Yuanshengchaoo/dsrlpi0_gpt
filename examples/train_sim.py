#! /usr/bin/env python
import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents import NoiseChunkFQLAgent, PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        num_cameras = getattr(
            variant,
            'num_cameras',
            getattr(variant, 'train_kwargs', {}).get('num_cameras', 1),
        )
        # Ensure the attribute exists on the variant so downstream code can rely on it.
        variant.num_cameras = num_cameras
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        chunk_size = getattr(variant, 'noise_chunk_size', max(1, variant.query_freq))
        self.action_space = Box(low=-1, high=1, shape=(chunk_size, 32,), dtype=np.float32) # 32 is the noise action space of pi 0


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = dict(variant['train_kwargs'])
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_spatial"]()
        task_id = 0
        task = task_suite.get_task(task_id)
        # >>> 在这里追加 <<<  （确定了 task_suite / task_id / task 之后）
        variant.libero_init_states = task_suite.get_task_init_states(task_id)  # 固定初始状态列表
        variant.num_steps_wait = getattr(variant, "num_steps_wait", 10)        # 开局空转等待步数（与 openpi 评测一致）
        variant.libero_dummy_action = getattr(variant, "libero_dummy_action", [0.0]*6 + [-1.0])  # 等待时用的占位动作
        variant.episode_idx = 0  # 可选：给 collect_traj 用来索引 init_state
        # <<< 在这里追加 >>>
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    chunk_reward_mode = kwargs.pop('chunk_reward_mode', None)
    if chunk_reward_mode is not None:
        variant.chunk_reward_mode = chunk_reward_mode
    elif not hasattr(variant, 'chunk_reward_mode'):
        variant.chunk_reward_mode = 'sparse'

    if variant.algorithm == 'noise_chunk_fql':
        noise_chunk_size = kwargs.pop('noise_chunk_size', variant.query_freq if variant.query_freq > 0 else 1)
        variant.query_freq = noise_chunk_size
    else:
        noise_chunk_size = max(1, variant.query_freq if variant.query_freq > 0 else 1)
        if variant.query_freq <= 0:
            variant.query_freq = noise_chunk_size
    variant.noise_chunk_size = noise_chunk_size

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    

    if variant.env == 'libero':
        # config = openpi_config.get_config("pi0_libero")
        # checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
        config = openpi_config.get_config("pi0_piper_libero")
        checkpoint_dir = "/mnt/afs/vva/all_checkponit/openpi-assets/checkpoints/libero_task03"
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    algorithm = variant.algorithm
    if algorithm == 'noise_chunk_fql':
        agent = NoiseChunkFQLAgent(variant.seed, sample_obs, sample_action, **kwargs)
    elif algorithm == 'pixel_sac':
        agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
 