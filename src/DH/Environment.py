import copy
import distutils.util

import tqdm

from src.DDQN.Agent import DDQNAgentParams, DDQNAgent
from src.DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from src.DH.Display import DHDisplay
from src.DH.Grid import DHGridParams, DHGrid
from src.DH.Physics import DHPhysicsParams, DHPhysics
from src.DH.Rewards import DHRewardParams, DHRewards
from src.DH.State import DHState
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class DHEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = DHGridParams()
        self.reward_params = DHRewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = DHPhysicsParams()


class DHEnvironment(BaseEnvironment):
    def __init__(self, params: DHEnvironmentParams):
        self.display = DHDisplay()
        super().__init__(params, self.display)

        self.grid = DHGrid(params.grid_params, stats=self.stats)
        self.rewards = DHRewards(params.reward_params, stats=self.stats)
        self.physics = DHPhysics(params=params.physics_params, stats=self.stats)
        self.agent = DDQNAgent(params.agent_params, self.grid.get_example_state(), self.physics.get_example_action(),
                               stats=self.stats)
        self.trainer = DDQNTrainer(params.trainer_params, agent=self.agent)

        self.display.set_channel(self.physics.channel)

        self.first_action = True
        self.last_actions = []
        self.last_rewards = []
        self.last_states = []

    def test_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.all_terminal:
            for self.physics.state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue
                action = self.agent.get_exploitation_action_target(state)
                next_state = self.physics.step(GridActions(action))
                self.stats.add_experience((copy.deepcopy(state), action, 0.0, copy.deepcopy(next_state)))
                state = copy.deepcopy(next_state)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def test_scenario(self, scenario):
        state = copy.deepcopy(self.init_episode(scenario))
        while not state.all_terminal:
            for state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue
                action = self.agent.get_exploitation_action_target(state)
                state = self.physics.step(GridActions(action))

    def step(self, state: DHState, random=False):
        for state.active_agent in range(state.num_agents):
            if state.terminal:
                continue
            if random:
                action = self.agent.get_random_action()
            else:
                action = self.agent.act(state)
            if not self.first_action:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward, state))

            self.last_states[state.active_agent] = copy.deepcopy(state)
            self.last_actions[state.active_agent] = action
            state = self.physics.step(GridActions(action))
            if state.terminal:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward, state))

        self.step_count += 1
        self.first_action = False
        return state

    def init_episode(self, init_state=None):
        state = super().init_episode(init_state)
        self.last_states = [None] * state.num_agents
        self.last_actions = [None] * state.num_agents
        self.first_action = True
        return state
