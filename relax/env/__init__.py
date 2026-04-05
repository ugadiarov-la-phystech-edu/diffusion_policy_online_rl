from collections import deque
from typing import Union

import numpy as np
import gymnasium
from gymnasium import Env, Wrapper, make
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box

from relax.env.vector import VectorEnv, SerialVectorEnv, GymProcessVectorEnv, PipeProcessVectorEnv, SpinlockProcessVectorEnv, FutexProcessVectorEnv

class RelaxWrapper(Wrapper):
    def __init__(self, env: Env, action_seed: int = 0):
        super().__init__(env)
        self.env: Env[np.ndarray, np.ndarray]

        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
        if isinstance(env, VectorEnv):
            self.obs_shape = env.observation_space.shape[1:]
            _, self.act_dim = env.action_space.shape
            single_action_space = env.single_action_space
        else:
            self.obs_shape = env.observation_space.shape
            self.act_dim, = env.action_space.shape
            single_action_space = env.action_space

        self.obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else None
        self._is_image_obs = len(self.obs_shape) == 3

        if np.any(single_action_space.low != -1.0) or np.any(single_action_space.high != 1.0):
            print(f"NOTE: The action space is not normalized, but {single_action_space.low} to {single_action_space.high}, will be rescaled.")
            self.needs_rescale = True
            self.original_action_center = (single_action_space.low + single_action_space.high) * 0.5
            self.original_action_half_range = (single_action_space.high - single_action_space.low) * 0.5
        else:
            self.needs_rescale = False
        self.original_action_dtype = env.action_space.dtype

        self._action_space = Box(
            low=-1,
            high=1,
            shape=env.action_space.shape,
            dtype=np.float32,
            seed=action_seed
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if not self._is_image_obs:
            obs = obs.astype(np.float32, copy=False)
        return obs, info

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._is_image_obs:
            obs = obs.astype(np.float32, copy=False)
        return obs, reward, terminated, truncated, info

def _apply_image_wrappers(env, image_size: int = 84, num_stack: int = 1):
    from gymnasium.wrappers import AddRenderObservation, ResizeObservation
    env = AddRenderObservation(env, render_only=True)
    env = ResizeObservation(env, shape=(image_size, image_size))
    env = FrameStack(env, num_stack=num_stack)
    return env

def create_env(name: str, seed: int, action_seed: int = 0, image_obs: bool = False, image_size: int = 84, num_stack: int = 1,):
    render_mode = "rgb_array" if image_obs else None
    env = make(name, render_mode=render_mode)
    if image_obs:
        env = _apply_image_wrappers(env, image_size, num_stack)
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_shape, env.act_dim


def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0, mode: str = "serial",
                      image_obs: bool = False, image_size: int = 84, num_stack: int = 1, **kwargs):
    Impl = {
        "serial": SerialVectorEnv,
        "gym": GymProcessVectorEnv,
        "pipe": PipeProcessVectorEnv,
        "spinlock": SpinlockProcessVectorEnv,
        "futex": FutexProcessVectorEnv,
    }[mode]
    env = Impl(name, num_envs, seed, image_obs=image_obs, image_size=image_size, num_stack=num_stack, **kwargs)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_shape, env.act_dim


class LazyFrames:
    """Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.

    Note:
        This object should only be converted to numpy array just before forward pass.
    """

    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames: list, lz4_compress: bool = False):
        """Lazyframe for a set of frames and if to apply lz4.

        Args:
            frames (list): The frames to convert to lazy frames
            lz4_compress (bool): Use lz4 to compress the frames internally

        Raises:
            DependencyNotInstalled: lz4 is not installed
        """
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            try:
                from lz4.block import compress
            except ImportError as e:
                raise DependencyNotInstalled(
                    "lz4 is not installed, run `pip install gymnasium[other]`"
                ) from e

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        """Gets a numpy array of stacked frames with specific dtype.

        Args:
            dtype: The dtype of the stacked frames

        Returns:
            The array of stacked frames with dtype
        """
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        """Returns the number of frame stacks.

        Returns:
            The number of frame stacks
        """
        return self.shape[0]

    def __getitem__(self, int_or_slice: Union[int, slice]):
        """Gets the stacked frames for a particular index or slice.

        Args:
            int_or_slice: Index or slice to get items for

        Returns:
            np.stacked frames for the int or slice

        """
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        """Checks that the current frames are equal to the other object."""
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame


class FrameStack(gymnasium.ObservationWrapper, gymnasium.utils.RecordConstructorArgs):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [3 * 4].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStack
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(0, 255, (96, 96, 3 * 4), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (96, 96, 3 * 4)
    """

    def __init__(
        self,
        env: gymnasium.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, lz4_compress=lz4_compress
        )
        gymnasium.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        shape = self.observation_space.shape
        shape = (*shape[:-1], shape[-1] * self.num_stack)

        low = np.stack([self.observation_space.low] * self.num_stack, axis=-1).reshape(shape)
        high = np.stack([self.observation_space.high] * self.num_stack, axis=-1).reshape(shape)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.stack(self.frames).reshape(self.observation_space.shape)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)

        [self.frames.append(obs) for _ in range(self.num_stack)]

        return self.observation(None), info
