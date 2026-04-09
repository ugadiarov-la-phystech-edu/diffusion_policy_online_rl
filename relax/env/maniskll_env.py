import gymnasium.spaces
import mani_skill.envs
import mani_skill.vector.wrappers.gymnasium
from mani_skill.utils.wrappers import CPUGymWrapper
import numpy as np
import torch
import torch.nn.functional as F

from relax.env.vector.base import VectorEnv


def _get_rgb(observation):
    return observation['sensor_data']['base_camera']['rgb']


class ManiSkillVectorEnv(VectorEnv):
    def __init__(self, env, num_envs: int, seed: int, image_size: int = 84, num_stack: int = 3, auto_reset: bool = True,
                 ignore_terminations: bool = False, record_metrics: bool = False,
                 env_kwargs=dict(obs_mode='rgb', control_mode='pd_joint_delta_pos', render_mode='rgb_array',
                                 sensor_configs=dict(width=224, height=224), max_episode_steps=100)):
        self._env = mani_skill.vector.wrappers.gymnasium.ManiSkillVectorEnv(env, num_envs, auto_reset,
                                                                            ignore_terminations, record_metrics,
                                                                            **env_kwargs)
        self.seed = seed
        self.num_envs = num_envs
        self.image_size = image_size
        self.num_stack = num_stack
        self.spec = self._env.spec
        self.observation_space = gymnasium.spaces.Box(low=0, high=255,
                                                      shape=(self.num_envs, self.image_size, self.image_size,
                                                             3 * self.num_stack), dtype=np.uint8, seed=self.seed)

        # frame_stack.shape -> (num_envs, h, w, 3, num_stack)
        self.frame_stack = np.zeros((*self.observation_space.shape[:-1], 3, self.num_stack),
                                    dtype=self.observation_space.dtype)
        self._fill_frames(np.ones((self.num_envs,), dtype=np.bool_),
                          self._get_images(self._env.reset(seed=self.seed)[0]).cpu().numpy())

    def _add_frames(self, mask, frames):
        self.frame_stack[mask] = np.roll(self.frame_stack[mask], shift=-1, axis=-1)
        self.frame_stack[mask, ..., -1] = frames[mask]

    def _fill_frames(self, mask, frames):
        self.frame_stack[mask] = frames[mask][..., np.newaxis]

    def _get_obs(self):
        return self.frame_stack.reshape(*self.frame_stack.shape[:-2], -1)

    def _get_images(self, observation):
        rgb = _get_rgb(observation)
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.
        resized = F.interpolate(rgb, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).clamp_(0, 1) * 255
        resized = resized.byte().permute(0, 2, 3, 1)
        return resized

    @staticmethod
    def _to_numpy(items):
        results = []
        for item in items:
            if torch.is_tensor(item):
                results.append(item.cpu().numpy())
            elif isinstance(item, dict):
                results.append({})
                if 'success' in item:
                    results[-1]['success'] = item['success'].cpu().numpy()
            else:
                raise ValueError(f'Unsupported type: {type(item)}')

        return results

    def __getattr__(self, name):
        attr = getattr(self._env, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)

            return wrapper

        return attr

    def step(self, actions):
        obs, rew, terminations, truncations, infos = self._env.step(actions)
        done = terminations | truncations
        done = done.cpu().numpy()
        has_done = done.any().item()
        obs = self._get_images(obs).cpu().numpy()
        self._add_frames(~done, obs)
        if has_done:
            self._add_frames(done, self._get_images(infos['final_observation']).cpu().numpy())

        result_obs = self._get_obs().copy()
        if has_done:
            self._fill_frames(done, obs)

        return result_obs, *self._to_numpy([rew, terminations, truncations, infos])

    def reset(self, *, seed=None, options=dict()):
        assert seed is None, f'seed is ignored'
        return self._get_obs().copy(), {}


class ManiSkillEnv(gymnasium.ObservationWrapper):
    def __init__(self, env, env_kwargs=dict(obs_mode='rgb', control_mode='pd_joint_delta_pos', render_mode='rgb_array',
                                 sensor_configs=dict(width=224, height=224), max_episode_steps=100)):
        env = CPUGymWrapper(gymnasium.make(env, num_envs=1, **env_kwargs))
        super().__init__(env)
        sensor_config = env_kwargs['sensor_configs']
        shape = (sensor_config['width'], sensor_config['height'], 3)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        return _get_rgb(observation)
