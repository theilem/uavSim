import numpy as np

from src.DH.DeviceManager import DeviceManagerParams, DeviceManager
import src.Map.Map as Map
from src.DH.State import DHState
from src.base.BaseGrid import BaseGrid, BaseGridParams


class DHGridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.device_manager = DeviceManagerParams()


class DHGrid(BaseGrid):

    def __init__(self, params: DHGridParams, stats):
        super().__init__(params, stats)
        self.params = params

        self.device_list = None
        self.device_manager = DeviceManager(self.params.device_manager)

        free_space = np.logical_not(
            np.logical_or(self.map_image.obstacles, self.map_image.start_land_zone))
        free_idcs = np.where(free_space)
        self.device_positions = list(zip(free_idcs[1], free_idcs[0]))

    def get_comm_obstacles(self):
        return self.map_image.obstacles

    def get_data_map(self):
        return self.device_list.get_data_map(self.shape)

    def get_collected_map(self):
        return self.device_list.get_collected_map(self.shape)

    def get_device_list(self):
        return self.device_list

    def get_grid_params(self):
        return self.params

    def init_episode(self):
        self.device_list = self.device_manager.generate_device_list(self.device_positions)

        state = DHState(self.map_image)
        state.reset_devices(self.device_list)

        # Replace False insures that starting positions of the agents are different
        idx = np.random.randint(len(self.starting_vector))
        state.position = self.starting_vector[idx]

        state.movement_budget = np.random.randint(low=self.params.movement_range[0],
                                                  high=self.params.movement_range[1] + 1, size=1)

        state.initial_movement_budget = state.movement_budget.copy()

        return state

    def init_scenario(self, scenario):
        self.device_list = scenario.device_list

        return scenario.init_state

    def get_example_state(self):
        state = DHState(self.map_image)
        state.device_map = np.zeros(self.shape, dtype=float)
        state.collected = np.zeros(self.shape, dtype=float)
        return state
