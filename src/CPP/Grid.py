import numpy as np

import src.Map.Map as ImageLoader
from src.CPP.State import CPPState
from src.CPP.RandomTargetGenerator import RandomTargetGenerator, RandomTargetGeneratorParams
from src.base.BaseGrid import BaseGrid, BaseGridParams


class CPPGridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.generator_params = RandomTargetGeneratorParams()


class CPPGrid(BaseGrid):

    def __init__(self, params: CPPGridParams, stats):
        super().__init__(params, stats)
        self.params = params

        self.generator = RandomTargetGenerator(params.generator_params, self.map_image.get_size())
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

    def init_episode(self):
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

        state = CPPState(self.map_image)
        state.reset_target(self.target_zone)

        idx = np.random.randint(0, len(self.starting_vector))
        state.position = self.starting_vector[idx]

        state.movement_budget = np.random.randint(low=self.params.movement_range[0],
                                                  high=self.params.movement_range[1] + 1)

        state.initial_movement_budget = state.movement_budget
        state.landed = False
        state.terminal = False

        return state

    def init_scenario(self, state: CPPState):
        self.target_zone = state.target

        return state

    def get_example_state(self):
        state = CPPState(self.map_image)
        state.reset_target(self.target_zone)
        state.position = [0, 0]
        state.movement_budget = 0
        state.initial_movement_budget = 0
        return state

    def get_target_zone(self):
        return self.target_zone
