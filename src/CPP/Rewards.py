from src.CPP.State import CPPState
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewardParams, GridRewards


class CPPRewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.cell_multiplier = 0.4


# Class used to track rewards
class CPPRewards(GridRewards):

    def __init__(self, reward_params: CPPRewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

    def calculate_reward(self, state: CPPState, action: GridActions, next_state: CPPState):
        reward = self.calculate_motion_rewards(state, action, next_state)

        # Reward the collected data
        reward += self.params.cell_multiplier * (state.get_remaining_cells() - next_state.get_remaining_cells())

        # Cumulative reward
        self.cumulative_reward += reward

        return reward
