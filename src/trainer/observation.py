from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import AvgPool2D
from gymnasium import spaces

from utils import Factory


class ObservationFunction:

    def __call__(self, state):
        return self.observe(state)

    def observe(self, state):
        raise NotImplementedError()

    def get_observation_space(self, state):
        raise NotImplementedError()

    def to_experience(self, obs):
        raise NotImplementedError()

    def to_experience_batch(self, obs):
        raise NotImplementedError()

    def from_experience_batch(self, batch):
        raise NotImplementedError()

    def from_experience_batch_tf(self, batch):
        raise NotImplementedError()

    def experience_length(self):
        raise NotImplementedError()


class GlobLocObservation(ObservationFunction):
    @dataclass
    class Params:
        global_map_scaling: int = 3
        local_map_size: int = 17

    def __init__(self, params: Params, max_budget):
        self.params = params
        self.max_budget = max_budget

    def observe(self, state):
        map_layers = tf.convert_to_tensor(state["map"])
        budget = tf.reshape(tf.convert_to_tensor(state["budget"], dtype=tf.float32), (-1, 1))
        landed = tf.reshape(state["landed"], (-1, 1))
        mask = tf.convert_to_tensor(state["mask"])
        heuristic_action = tf.convert_to_tensor(state["heuristic_action"]) if "heuristic_action" in state else None

        global_map, local_map, scalars, mask = self._prepare_observations_tf(map_layers, budget, landed, mask,
                                                                             heuristic_action)
        return {"global_map": global_map, "local_map": local_map, "scalars": scalars, "mask": mask}

    @tf.function
    def _prepare_observations_tf(self, map_layers, budget, landed, mask, heuristic_action):
        map_layers = tf.cast(map_layers, dtype=tf.float32)
        global_map = AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(map_layers)

        crop_frac = float(self.params.local_map_size) / float(map_layers.shape[1])
        local_map = tf.image.central_crop(map_layers, crop_frac)

        max_budget = self.max_budget
        scalars = tf.concat((budget / max_budget, tf.cast(landed, tf.float32)), axis=-1)

        return global_map, local_map, scalars, mask

    def get_observation_space(self, state):
        obs = self.observe(state)
        return spaces.Dict(
            {
                "global_map": spaces.Box(low=0, high=1, shape=obs["global_map"].shape, dtype=np.float32),
                "local_map": spaces.Box(low=0, high=1, shape=obs["local_map"].shape, dtype=np.float32),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=np.float32),
                "mask": spaces.Box(low=0, high=1, shape=obs["mask"].shape, dtype=bool)
            }
        )

    def to_experience(self, obs):
        return [
            obs["global_map"].numpy()[0].astype(np.float32),
            obs["local_map"].numpy()[0].astype(np.float32),
            obs["scalars"].numpy()[0].astype(np.float32),
            obs["mask"].numpy()[0].astype(bool)
        ]

    def to_experience_batch(self, obs):
        return [
            obs["global_map"].numpy().astype(np.float32),
            obs["local_map"].numpy().astype(np.float32),
            obs["scalars"].numpy().astype(np.float32),
            obs["mask"].numpy().astype(bool)
        ]

    def from_experience_batch(self, batch):
        return {
            "global_map": tf.convert_to_tensor(batch[0]),
            "local_map": tf.convert_to_tensor(batch[1].astype(np.float32)),
            "scalars": tf.convert_to_tensor(batch[2]),
            "mask": tf.convert_to_tensor(batch[3])
        }

    def from_experience_batch_tf(self, batch):
        return {
            "global_map": batch[0],
            "local_map": tf.cast(batch[1], dtype=tf.float32),
            "scalars": batch[2],
            "mask": batch[3]
        }

    def experience_length(self):
        return 4


class CenteredMapObservation(ObservationFunction):
    @dataclass
    class Params:
        observe_mask: bool = False
        observe_heuristic: bool = False
        bool_map: bool = True

    def __init__(self, params: Params, max_budget):
        self.params = params
        self.max_budget = max_budget

    def observe(self, obs):
        map_layers = tf.convert_to_tensor(obs["map"])
        budget = tf.reshape(tf.convert_to_tensor(obs["budget"], dtype=tf.float32), (-1, 1))
        landed = tf.reshape(obs["landed"], (-1, 1))
        mask = tf.convert_to_tensor(obs["mask"])
        heuristic_action = tf.convert_to_tensor(obs["heuristic_action"]) if "heuristic_action" in obs else None

        centered_map, scalars = self._prepare_observations_tf(map_layers, budget, landed, mask, heuristic_action)
        return {"centered_map": centered_map, "scalars": scalars, "mask": mask}

    @tf.function
    def _prepare_observations_tf(self, map_layers, budget, landed, mask, heuristic_action):
        centered_map = tf.cast(map_layers, dtype=tf.float32)
        max_budget = self.max_budget
        scalars = tf.concat((budget / max_budget, tf.cast(landed, tf.float32)), axis=-1)

        if self.params.observe_mask:
            scalars = tf.concat((scalars, tf.cast(mask, tf.float32)), axis=-1)

        if self.params.observe_heuristic and heuristic_action is not None:
            heuristic_hot = tf.one_hot(heuristic_action, depth=mask.shape[-1] - 1,
                                       dtype=tf.float32, axis=-1)
            scalars = tf.concat((scalars, heuristic_hot), axis=-1)

        return centered_map, scalars

    def get_observation_space(self, state):
        obs = self.observe(state)
        return spaces.Dict(
            {
                "centered_map": spaces.Box(low=0, high=1, shape=obs["centered_map"].shape, dtype=float),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=float),
                "mask": spaces.Box(low=0, high=1, shape=obs["mask"].shape, dtype=bool)
            }
        )

    def to_experience(self, obs):
        return [
            obs["centered_map"].numpy()[0].astype(bool if self.params.bool_map else np.float32),
            obs["scalars"].numpy()[0].astype(np.float32),
            obs["mask"].numpy()[0].astype(bool)
        ]

    def from_experience_batch(self, batch):
        return {
            "centered_map": tf.cast(tf.convert_to_tensor(batch[0]), tf.float32),
            "scalars": tf.cast(tf.convert_to_tensor(batch[1]), tf.float32),
            "mask": tf.convert_to_tensor(batch[2])
        }

    def experience_length(self):
        return 3

    def minimum_map_size(self, min_sizes):
        return min_sizes


class ObservationFunctionFactory(Factory):
    @classmethod
    def registry(cls):
        return {
            "glob_loc": GlobLocObservation,
            "centered": CenteredMapObservation,
        }

    @classmethod
    def defaults(cls):
        return "glob_loc", GlobLocObservation
