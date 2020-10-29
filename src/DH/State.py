import numpy as np
from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.base.BaseState import BaseState


class DHScenario:
    def __init__(self):
        self.device_idcs = []
        self.device_data = []
        self.position_idx = 0
        self.movement_budget = 100


class DHState(BaseState):
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.device_list = None
        self.device_map = None  # Floating point sparse matrix showing devices and their data to be collected

        self.position = [0, 0]
        self.movement_budget = 0
        self.landed = False
        self.terminal = False
        self.device_com = -1

        self.initial_movement_budget = 0
        self.initial_total_data = 0
        self.collected = None

    def set_landed(self, landed):
        self.landed = landed

    def set_position(self, position):
        self.position = position

    def decrement_movement_budget(self):
        self.movement_budget -= 1

    def set_terminal(self, terminal):
        self.terminal = terminal

    def set_device_com(self, device_com):
        self.device_com = device_com

    def get_remaining_data(self):
        return np.sum(self.device_map)

    def get_total_data(self):
        return self.initial_total_data

    def get_scalars(self):
        """
        Return the scalars without position, as it is treated individually
        """
        return np.array([self.movement_budget])

    def get_num_scalars(self):
        return 1

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)

        padded_rest = pad_centered(self, np.expand_dims(self.landing_zone, -1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_float_map(self):
        return pad_centered(self, np.expand_dims(self.device_map, -1), 0)

    def get_float_map_shape(self):
        return self.get_float_map().shape

    def is_in_landing_zone(self):
        return self.landing_zone[self.position[1]][self.position[0]]

    def is_in_no_fly_zone(self):
        # Out of bounds is implicitly nfz
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            return self.no_fly_zone[self.position[1], self.position[0]]
        return True

    def get_collection_ratio(self):
        return np.sum(self.collected) / self.initial_total_data

    def get_collected_data(self):
        return np.sum(self.collected)

    def reset_devices(self, device_list):
        self.device_map = device_list.get_data_map(self.no_fly_zone.shape)
        self.collected = np.zeros(self.no_fly_zone.shape, dtype=float)
        self.initial_total_data = device_list.get_total_data()
        self.device_list = device_list

    def is_terminal(self):
        return self.terminal

