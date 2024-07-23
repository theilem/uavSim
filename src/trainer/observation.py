from dataclasses import dataclass
from typing import Tuple

import numpy as np
from gymnasium import spaces

from utils import Factory

import skimage.measure


class ObservationFunction:
    @dataclass
    class Params:
        position_history: bool = True
        random_layer: bool = False

    def __call__(self, state):
        return self.observe(state)

    def observe(self, state):
        raise NotImplementedError()

    def observe_multi(self, states):
        observes = [self.observe(state) for state in states]
        obs = {key: np.concatenate([observe[key] for observe in observes], axis=0) for key in observes[0].keys()}
        return obs

    def get_observation_space(self, state):
        raise NotImplementedError()


class PlainMapObservation(ObservationFunction):
    @dataclass
    class Params(ObservationFunction.Params):
        padding_values: Tuple[int] = (0, 1, 1, 0, 0, 0)
        pad_frame: bool = False

    def __init__(self, params: Params, max_budget):
        self.params = params
        self.max_budget = max_budget
        self.padded_map = None

    def observe(self, state):
        map_layers = state.map
        position_layer = np.zeros_like(state.map[..., 0])
        position_layer[state.position[0], state.position[1]] = 1
        map_layers = np.concatenate((map_layers, np.expand_dims(position_layer, -1)), axis=-1)
        if self.params.position_history:
            map_layers = np.concatenate((map_layers, np.expand_dims(state.position_history, -1)), axis=-1)
        if self.params.pad_frame:
            if self.padded_map is None:
                m = map_layers.shape[0] + 2
                self.padded_map = np.repeat(
                    np.repeat(np.reshape(self.params.padding_values, (1, 1, -1)), repeats=m, axis=0), repeats=m,
                    axis=1).astype(float)
            pm = self.padded_map.copy()
            pm[1:-1, 1:-1] = map_layers
            map_layers = pm

        map_layers = np.expand_dims(map_layers, axis=0)
        scalars = np.expand_dims(
            np.stack((state.budget / self.max_budget, state.landed), axis=-1), axis=0)
        mask = np.expand_dims(state.action_mask, axis=0)

        obs = {"map": map_layers, "scalars": scalars, "mask": mask}
        return obs

    def get_observation_space(self, state):
        obs = self.observe(state)
        return spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=obs["map"].shape, dtype=float),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=float),
                "mask": spaces.Box(low=0, high=1, shape=obs["mask"].shape, dtype=bool)
            }
        )


class CenteredMapObservation(ObservationFunction):
    @dataclass
    class Params(ObservationFunction.Params):
        padding_values: Tuple[int] = (0, 1, 1, 0, 0)

    def __init__(self, params: Params, max_budget):
        self.params = params
        self.max_budget = max_budget

        self.centered_map = None

    def pad_centered(self, map_layers, position):

        x, y = np.array(map_layers.shape[:2]) - position - 1
        m = map_layers.shape[0]

        if self.centered_map is None:
            m_c = m * 2 - 1
            self.centered_map = np.repeat(
                np.repeat(np.reshape(self.params.padding_values, (1, 1, -1)), repeats=m_c, axis=0), repeats=m_c,
                axis=1).astype(float)

        centered_map = self.centered_map.copy()
        centered_map[x:x + m, y:y + m] = map_layers

        return centered_map

    def observe(self, state):
        map_layers = state.map
        if self.params.position_history:
            map_layers = np.concatenate((map_layers, np.expand_dims(state.position_history, -1)), axis=-1)

        centered_map = np.expand_dims(self.pad_centered(map_layers, state.position), axis=0)
        scalars = np.expand_dims(
            np.stack((state.budget / self.max_budget, state.landed), axis=-1), axis=0)
        mask = np.expand_dims(state.action_mask, axis=0)

        return {"map": centered_map, "scalars": scalars, "mask": mask}

    def observe_multi(self, states):
        map_layers = [state.map for state in states]

        if self.params.position_history:
            position_histories = [state.position_history for state in states]
            map_layers = [np.concatenate((maps, np.expand_dims(position_history, -1)), axis=-1) for
                          maps, position_history in zip(map_layers, position_histories)]

        centered_map = np.stack([self.pad_centered(maps, state.position) for maps, state in zip(map_layers, states)], 0)
        scalars = np.stack([np.stack((state.budget / self.max_budget, state.landed), axis=-1) for state in states],
                           axis=0)
        mask = np.stack([state.action_mask for state in states], axis=0)
        return {"map": centered_map, "scalars": scalars, "mask": mask}

    def get_observation_space(self, state):
        obs = self.observe(state)
        return spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=obs["map"].shape, dtype=float),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=float),
                "mask": spaces.Box(low=0, high=1, shape=obs["mask"].shape, dtype=bool)
            }
        )


class GlobLocObservation(CenteredMapObservation):
    @dataclass
    class Params(CenteredMapObservation.Params):
        global_map_scaling: int = 3
        local_map_size: int = 17

    def __init__(self, params: Params, max_budget):
        super().__init__(params, max_budget)
        self.params = params

    def observe(self, state):
        obs = super().observe(state)
        obs = self._observe(obs)

        return obs

    def _observe(self, obs):
        centered = obs.pop("map")
        g = self.params.global_map_scaling
        l = self.params.local_map_size
        global_map = skimage.measure.block_reduce(centered, (1, g, g, 1), np.mean)
        x, y = centered.shape[1:3]
        local_map = centered[:, x // 2 - l // 2: x // 2 + l // 2 + 1, x // 2 - l // 2: x // 2 + l // 2 + 1, :]
        obs.update({"global_map": global_map, "local_map": local_map})

        return obs

    def observe_multi(self, states):
        obs = super().observe_multi(states)
        obs = self._observe(obs)

        return obs

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


class ObservationFunctionFactory(Factory):
    @classmethod
    def registry(cls):
        return {
            "glob_loc": GlobLocObservation,
            "centered": CenteredMapObservation,
            "plain": PlainMapObservation
        }

    @classmethod
    def defaults(cls):
        return "glob_loc", GlobLocObservation
