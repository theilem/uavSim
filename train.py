import argparse
from dataclasses import dataclass

from src.gym import PathPlanningGymFactory
from src.trainer.agent import AgentFactory
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = PathPlanningParams.add_args_to_parser(parser)
    args = parser.parse_args()

    params, args = PathPlanningParams.from_parsed_args(args)
    log_dir = params.create_folders(args)

    gym = PathPlanningGymFactory.create(params.gym)

    action_space = gym.action_space
    obs_space = gym.observation_space

    agent = AgentFactory.create(params.agent, obs_space=obs_space, act_space=action_space)

    if args.verbose:
        agent.summary()

    logger = Logger(params.logger, log_dir, agent)
    trainer = TrainerFactory.create(params.trainer, gym=gym, logger=logger, agent=agent)
    evaluator = Evaluator(params.evaluator, trainer, gym)
    logger.evaluator = evaluator

    params.save_to(params.log_dir + "config.json")

    if not args.gpu and args.gpu_id is None:
        print("Running on CPU")
    else:
        print(f"Running on GPU {args.gpu_id}" if args.gpu_id is not None else "Running on GPU")

    trainer.train()
    agent.save_keras(params.log_dir + 'models/')

    gym.close()
