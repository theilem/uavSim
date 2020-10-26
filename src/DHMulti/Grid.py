import numpy as np

from src.DH.Grid import DHGrid, DHGridParams
from src.DHMulti.State import DHMultiState


class DHMultiGridParams(DHGridParams):
    def __init__(self):
        super().__init__()
        self.num_agents_range = [1, 3]


class DHMultiGrid(DHGrid):

    def __init__(self, params: DHMultiGridParams, stats):
        super().__init__(params, stats)
        self.params = params
        self.num_agents = params.num_agents_range[0]

    def init_episode(self):
        self.device_list = self.device_manager.generate_device_list(self.device_positions)

        self.num_agents = int(np.random.randint(low=self.params.num_agents_range[0],
                                                high=self.params.num_agents_range[1] + 1, size=1))
        state = DHMultiState(self.map_image, self.num_agents)
        state.reset_devices(self.device_list)

        # Replace False insures that starting positions of the agents are different
        idx = np.random.choice(len(self.starting_vector), size=self.num_agents, replace=False)
        state.positions = [self.starting_vector[i] for i in idx]

        state.movement_budgets = np.random.randint(low=self.params.movement_range[0],
                                                   high=self.params.movement_range[1] + 1, size=self.num_agents)

        state.initial_movement_budgets = state.movement_budgets.copy()

        return state

    def init_scenario(self, scenario):
        self.device_list = scenario.device_list
        self.num_agents = scenario.init_state.num_agents

        return scenario.init_state

    def get_example_state(self):
        num_agents = self.params.num_agents_range[0]
        state = DHMultiState(self.map_image, num_agents)
        state.device_map = np.zeros(self.shape, dtype=float)
        state.collected = np.zeros(self.shape, dtype=float)
        return state
