import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import jax

from relax.buffer.base import Buffer
from relax.utils.experience import Experience


class FrameStackBuffer(Buffer[Experience]):
    """Memory-efficient replay buffer for frame-stacked image observations.

    Stores single frames instead of stacked observations, reconstructing
    stacked obs/next_obs on sample. For num_stack=4, this gives ~8x memory
    reduction compared to storing full stacked obs + next_obs.

    Uses per-environment circular sub-buffers to handle vectorized envs
    where consecutive buffer entries come from different environments.
    """

    def __init__(self, frame_shape: Tuple[int, ...], act_dim: int, num_stack: int,
                 num_envs: int, capacity: int, seed: int = 0, frame_dtype=np.uint8):
        self.frame_shape = frame_shape  # (H, W, C) single frame
        self.C = frame_shape[-1]
        self.act_dim = act_dim
        self.num_stack = num_stack
        self.num_envs = num_envs
        self.env_capacity = capacity // num_envs

        self.frames = np.zeros((num_envs, self.env_capacity, *frame_shape), dtype=frame_dtype)
        self.actions = np.zeros((num_envs, self.env_capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((num_envs, self.env_capacity), dtype=np.float32)
        self.dones = np.zeros((num_envs, self.env_capacity), dtype=np.bool_)

        self.ptrs = np.zeros(num_envs, dtype=np.int64)
        self.lens = np.zeros(num_envs, dtype=np.int64)

        self.rng = np.random.default_rng(seed)

    def add(self, sample: Experience, *, from_jax: bool = False) -> None:
        if from_jax:
            sample = jax.device_get(sample)
        e = 0
        p = self.ptrs[e]
        self.frames[e, p] = np.asarray(sample.obs[..., -self.C:])
        self.actions[e, p] = sample.action
        self.rewards[e, p] = sample.reward
        self.dones[e, p] = sample.done
        self.ptrs[e] = (p + 1) % self.env_capacity
        self.lens[e] = min(self.lens[e] + 1, self.env_capacity)

    def add_batch(self, samples: Experience, *, from_jax: bool = False) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        obs_frames = np.asarray(samples.obs[..., -self.C:])
        for e in range(self.num_envs):
            p = self.ptrs[e]
            self.frames[e, p] = obs_frames[e]
            self.actions[e, p] = samples.action[e]
            self.rewards[e, p] = samples.reward[e]
            self.dones[e, p] = samples.done[e]
            self.ptrs[e] = (p + 1) % self.env_capacity
            self.lens[e] = min(self.lens[e] + 1, self.env_capacity)

    def sample(self, size: int, *, to_jax: bool = False) -> Experience:
        return self.sample_with_indices(size, to_jax=to_jax)[0]

    def sample_with_indices(self, size: int, *, to_jax: bool = False) -> Tuple[Experience, np.ndarray]:
        env_indices, time_indices = self._sample_indices(size)

        obs_batch = self._reconstruct_stack_batch(env_indices, time_indices)
        next_obs_batch = self._reconstruct_next_stack_batch(env_indices, time_indices)

        action_batch = self.actions[env_indices, time_indices]
        reward_batch = self.rewards[env_indices, time_indices]
        done_batch = self.dones[env_indices, time_indices]

        experience = Experience(
            obs=obs_batch,
            action=action_batch,
            reward=reward_batch,
            done=done_batch,
            next_obs=next_obs_batch,
        )

        virtual_indices = env_indices * self.env_capacity + time_indices

        if to_jax:
            experience = jax.device_put(experience)

        return experience, virtual_indices

    def replace(self, indices: np.ndarray, samples: Experience, *, from_jax: bool = False) -> None:
        raise NotImplementedError(
            "FrameStackBuffer does not support replace(). "
            "Use TreeBuffer for buffers that need replace (e.g. DIPO's diffusion_buffer)."
        )

    def save(self, path: Path) -> None:
        data = {
            "frames": self.frames,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "ptrs": self.ptrs,
            "lens": self.lens,
            "num_envs": self.num_envs,
            "num_stack": self.num_stack,
            "frame_shape": self.frame_shape,
            "act_dim": self.act_dim,
        }
        with path.open("wb") as f:
            pickle.dump(data, f)

    def __len__(self):
        return int(self.lens.sum())

    def __repr__(self):
        return (f"FrameStackBuffer(num_envs={self.num_envs}, env_capacity={self.env_capacity}, "
                f"num_stack={self.num_stack}, frame_shape={self.frame_shape}, len={len(self)})")

    # ---- Internal helpers ----

    def _sample_indices(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        # Exclude the most recent entry per env (next frame not yet stored)
        valid_lens = np.maximum(self.lens - 1, 0)
        total_valid = int(valid_lens.sum())
        assert total_valid > 0, "Buffer has no valid samples"

        flat_indices = self.rng.integers(0, total_valid, size=size)

        cum_lens = np.cumsum(valid_lens)
        env_indices = np.searchsorted(cum_lens, flat_indices, side="right")

        offsets = flat_indices - np.concatenate([[0], cum_lens[:-1]])[env_indices]

        time_indices = np.empty(size, dtype=np.int64)
        for i in range(size):
            e = env_indices[i]
            if self.lens[e] < self.env_capacity:
                # Buffer not full: valid range is [0, ptr-1)
                time_indices[i] = offsets[i]
            else:
                # Buffer full: oldest at ptr, newest valid at ptr-2
                time_indices[i] = (self.ptrs[e] + offsets[i]) % self.env_capacity

        return env_indices, time_indices

    def _is_boundary(self, env_idx: int, prev_idx: int) -> bool:
        """Check if prev_idx is an episode boundary (done=True) or invalid."""
        if self.dones[env_idx, prev_idx]:
            return True
        # Check if prev_idx is outside valid data
        if self.lens[env_idx] < self.env_capacity:
            if prev_idx < 0 or prev_idx >= self.ptrs[env_idx]:
                return True
        return False

    def _reconstruct_stack_batch(self, env_indices: np.ndarray, time_indices: np.ndarray) -> np.ndarray:
        """Vectorized reconstruction of stacked observations for a batch."""
        size = len(env_indices)
        ns = self.num_stack

        # Build lookback index array: (size, num_stack)
        # Column 0 = oldest frame, column ns-1 = newest (current) frame
        lookback_offsets = np.arange(ns - 1, -1, -1)  # [ns-1, ns-2, ..., 0]
        candidates = (time_indices[:, None] - lookback_offsets[None, :]) % self.env_capacity
        # candidates shape: (size, num_stack)

        # Check episode boundaries in lookback positions (exclude current frame)
        # done at candidates[:, j] means frames at j and earlier are from a previous episode
        done_flags = self.dones[env_indices[:, None], candidates].copy()
        done_flags[:, -1] = False  # current frame is always valid

        # Also check for invalid indices (buffer not yet full)
        for i in range(size):
            e = env_indices[i]
            if self.lens[e] < self.env_capacity:
                for j in range(ns - 1):
                    if candidates[i, j] >= self.ptrs[e]:
                        done_flags[i, j] = True

        # Find the latest boundary per sample (rightmost True in done_flags)
        # Positions 0..latest_boundary should be padded with frame at latest_boundary+1
        boundary_positions = np.where(done_flags, np.arange(ns)[None, :], -1)
        latest_boundary = boundary_positions.max(axis=1)  # (size,) -1 if no boundary

        # Gather all frames: (size, num_stack, H, W, C)
        all_frames = self.frames[env_indices[:, None], candidates]

        # Apply padding for samples with boundaries
        has_boundary = latest_boundary >= 0
        if np.any(has_boundary):
            boundary_mask = np.arange(ns)[None, :] <= latest_boundary[:, None]  # (size, ns)
            boundary_mask &= has_boundary[:, None]
            # For each sample with boundary, get the padding frame (frame just after boundary)
            pad_col = np.minimum(latest_boundary + 1, ns - 1)  # (size,)
            pad_frames = all_frames[np.arange(size), pad_col]  # (size, H, W, C)
            # Broadcast pad frame into boundary positions
            all_frames[boundary_mask] = np.broadcast_to(
                pad_frames[:, None], (size, ns, *self.frame_shape)
            )[boundary_mask]

        # Reshape: (size, num_stack, H, W, C) -> (size, H, W, C * num_stack)
        # Reorder axes: (size, H, W, num_stack, C) then reshape
        all_frames = np.moveaxis(all_frames, 1, -2)  # (size, H, W, num_stack, C)
        stacked = all_frames.reshape(size, *self.frame_shape[:-1], self.C * ns)

        return stacked

    def _reconstruct_next_stack_batch(self, env_indices: np.ndarray, time_indices: np.ndarray) -> np.ndarray:
        """Reconstruct next_obs stacked observations for a batch."""
        done_at_t = self.dones[env_indices, time_indices]

        # For non-done transitions: next_obs stack is centered at t+1
        next_time_indices = (time_indices + 1) % self.env_capacity
        next_stacks = self._reconstruct_stack_batch(env_indices, next_time_indices)

        if np.any(done_at_t):
            # For done transitions: return current obs stack as safe placeholder
            # (Bellman target is multiplied by (1-done)=0, so exact value doesn't matter)
            current_stacks = self._reconstruct_stack_batch(
                env_indices[done_at_t], time_indices[done_at_t]
            )
            next_stacks[done_at_t] = current_stacks

        return next_stacks

    @staticmethod
    def create(frame_shape: Tuple[int, ...], act_dim: int, num_stack: int,
               num_envs: int, capacity: int, seed: int = 0,
               frame_dtype=np.uint8) -> "FrameStackBuffer":
        return FrameStackBuffer(frame_shape, act_dim, num_stack, num_envs, capacity, seed, frame_dtype)
