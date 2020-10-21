import copy

from src.DDQN.Agent import DDQNAgentParams, DDQNAgent
from src.CPP.Display import CPPDisplay
from src.CPP.Grid import CPPGrid, CPPGridParams
from src.CPP.Physics import CPPPhysics, CPPPhysicsParams
from src.CPP.State import CPPState
from src.CPP.Rewards import CPPRewardParams, CPPRewards

from src.DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class CPPEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = CPPGridParams()
        self.reward_params = CPPRewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = CPPPhysicsParams()


class CPPEnvironment(BaseEnvironment):
    def __init__(self, params: CPPEnvironmentParams):
        self.display = CPPDisplay()
        super().__init__(params, self.display)

        self.grid = CPPGrid(params.grid_params, self.stats)
        self.rewards = CPPRewards(params.reward_params, stats=self.stats)
        self.physics = CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent = DDQNAgent(params.agent_params, self.grid.get_example_state(), self.physics.get_example_action(),
                               stats=self.stats)
        self.trainer = DDQNTrainer(params=params.trainer_params, agent=self.agent)

    def test_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.terminal:
            action = self.agent.get_exploitation_action_target(state)
            next_state = self.physics.step(GridActions(action))
            reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
            self.stats.add_experience((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
            state = copy.deepcopy(next_state)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def test_scenario(self, scenario):
        state = copy.deepcopy(self.init_episode(scenario))
        while not state.terminal:
            action = self.agent.get_exploitation_action_target(state)
            state = self.physics.step(GridActions(action))

    def step(self, state: CPPState, random=False):
        if random:
            action = self.agent.get_random_action()
        else:
            action = self.agent.act(state)
        next_state = self.physics.step(GridActions(action))
        reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
        self.trainer.add_experience(state, action, reward, next_state)
        self.stats.add_experience((state, action, reward, next_state))
        self.step_count += 1
        return next_state
