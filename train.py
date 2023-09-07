import argparse
from dataclasses import dataclass

from src.gym import PathPlanningGymFactory
from src.trainer.agent import AgentFactory
from src.trainer.observation import ObservationFunctionFactory
from src.base.evaluator import Evaluator
from src.base.logger import Logger
from src.trainer.trainer import TrainerFactory
from utils import AbstractParams


@dataclass
class PathPlanningParams(AbstractParams):
    trainer: TrainerFactory.default_param_type() = TrainerFactory.default_params()
    gym: PathPlanningGymFactory.default_param_type() = PathPlanningGymFactory.default_params()
    logger: Logger.Params = Logger.Params()
    evaluator: Evaluator.Params = Evaluator.Params()
    agent: AgentFactory.default_param_type() = AgentFactory.default_params()
    observation: ObservationFunctionFactory.default_param_type() = ObservationFunctionFactory.default_params()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = PathPlanningParams.add_args_to_parser(parser)
    args = parser.parse_args()

    params, args = PathPlanningParams.from_parsed_args(args)
    log_dir = params.create_folders(args)

    gym = PathPlanningGymFactory.create(params.gym)

    observation_function = ObservationFunctionFactory.create(params.observation,
                                                             max_budget=gym.params.budget_range[-1])
    action_space = gym.action_space

    obs_space = observation_function.get_observation_space(gym.observation_space.sample())
    agent = AgentFactory.create(params.agent, obs_space=obs_space, act_space=action_space)

    if args.verbose:
        agent.summary()

    logger = Logger(params.logger, log_dir, agent)
    trainer = TrainerFactory.create(params.trainer, gym=gym, logger=logger, agent=agent,
                                    observation_function=observation_function, action_space=action_space)
    evaluator = Evaluator(params.evaluator, trainer, gym)
    logger.evaluator = evaluator

    params.save_to(params.log_dir + "config.json")

    trainer.train()
    agent.save_keras(params.log_dir + 'models/')
