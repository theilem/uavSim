import argparse
import os
import pickle

import numpy as np
import pandas as pd
import pygame

from src.gym import PathPlanningGymFactory
from src.trainer.agent import AgentFactory
from src.base.evaluator import PyGameHuman, InteractiveEvaluator
from src.trainer.trainer import TrainerFactory

from train import PathPlanningParams

from utils import find_map, find_scenario, find_config_model, override_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', nargs='*', default=None, help='Add maps')
    parser.add_argument('-d', action='store_true', help='remove all other maps')
    parser.add_argument('-r', nargs='*', default=None, help='Record episode only, potentially override render params')
    parser.add_argument('-n', default=20, help='Parallel gyms for evaluate')
    parser.add_argument('--scenario', default=None, help='Load specific scenario')
    parser.add_argument('--scenarios', default=None, help='Load file with multiple scenarios')
    parser.add_argument('--all_maps', action='store_true', help='Load all maps')
    parser.add_argument('--heuristic', action='store_true', help='Use Heuristic Only')
    parser.add_argument('--stochastic', action='store_true', help='Set agent to stochastic')
    parser.add_argument('--maps_only', action='store_true', help='Draws maps only')
    parser.add_argument('--gym_only', action='store_true', help='Only evaluates gym. Specify full config path.')

    parser = PathPlanningParams.add_args_to_parser(parser)
    args = parser.parse_args()

    agent_name = args.config

    if not args.gym_only:
        args.config = find_config_model(args.config)

    params, args = PathPlanningParams.from_parsed_args(args)
    log_dir = args.config.rsplit('/', maxsplit=1)[0]

    if args.d:
        params.gym["params"]["map_path"] = []

    if args.all_maps:
        maps = [file.replace(".png", "") for file in os.listdir("res") if file.endswith(".png")]
        for m in maps:
            map_path = find_map(m)
            if map_path in params.gym["params"]["map_path"]:
                continue
            params.gym["params"]["map_path"].append(map_path)
    elif args.a is not None:
        for m in args.a:
            params.gym["params"]["map_path"].append(find_map(m))

    init = None
    if args.scenario is not None:
        with open(find_scenario(args.scenario), 'rb') as f:
            init = pickle.load(f)

    gym = PathPlanningGymFactory.create(params.gym)

    if not args.gym_only:
        obs_space = gym.observation_space

        action_space = gym.action_space
        agent = AgentFactory.create(params.agent, obs_space=obs_space, act_space=action_space)

        if args.verbose:
            agent.summary()

        trainer = TrainerFactory.create(params.trainer, gym=gym, logger=None, agent=agent)
        model_dir = log_dir + "/models"
        try:
            agent.load_keras(model_dir)
            print("Loaded Keras Model")
        except OSError as e:
            print("Could not load Keras Model")
            print(e)
            agent.load_network(model_dir)
            agent.load_weights(model_dir)
            print("Loaded network and weights")
            agent.save_keras(model_dir)
    else:
        trainer = None

    human = PyGameHuman([(pygame.K_RIGHT, 0),
                         (pygame.K_DOWN, 1),
                         (pygame.K_LEFT, 2),
                         (pygame.K_UP, 3),
                         (pygame.K_SPACE, 4),
                         (pygame.K_m, 5),
                         (pygame.K_n, 6),
                         (pygame.K_s, -1)])
    evaluator = InteractiveEvaluator(params.evaluator, trainer, gym, human)
    if args.maps_only:
        evaluator.draw_maps()
        return

    if args.heuristic:
        evaluator.use_heuristic = True
    if args.stochastic:
        evaluator.stochastic = True

    if args.scenarios is not None:
        with open(args.scenarios, "rb") as f:
            scenarios = pickle.load(f)

        inits = [scenario["init"] for scenario in scenarios]
        # Check that maps are available
        available = True
        missing = []
        for init in inits:
            if init.map_name not in gym.map_names:
                if init.map_name in missing:
                    continue
                print(f"Missing map {init.map_name}")
                available = False
                missing.append(init.map_name)

        if not available:
            print("Cannot run. Add missing maps.")
            exit(1)

        infos = evaluator.evaluate_episodes(inits, int(args.n))

        total_steps = np.array([info["total_steps"] for info in infos])
        task_solved = np.array([info["task_solved"] for info in infos])
        total_steps_heuristic = np.array([scenario["total_steps"] for scenario in scenarios])
        rpd = np.where(task_solved, (total_steps - total_steps_heuristic) / total_steps_heuristic, np.nan)

        df = pd.DataFrame()
        df["map_name"] = [scenario["map_name"] for scenario in scenarios]
        df["unique_id"] = [scenario["unique_id"] for scenario in scenarios]
        df["task_solved"] = task_solved
        df["rpd"] = rpd
        if not evaluator.stochastic:
            agent_name = f"{agent_name}_deterministic"
        data = {"data": df, "agent": agent_name, "scenarios": args.scenarios}
        filename = f"{args.scenarios.replace('.pickle', f'_{agent_name}.pickle')}"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        return

    if args.r is not None:
        if init is None:
            print("Init required to record episode. Pass with --scenario")
            exit(1)

        if len(args.r) > 0:
            gym.params.rendering = override_params(gym.params.rendering, args.r)
            evaluator.render_params = gym.params.rendering

        if args.heuristic:
            name = "heuristic"
        else:
            name = log_dir if "/" not in log_dir else log_dir.split("/")[-1]
        scenario = args.scenario if "/" not in args.scenario else args.scenario.split("/")[-1]

        evaluator.record_episode(init, name=f"{name}_{scenario}")
        print(f"Finished recording {name}_{scenario}")
        return

    evaluator.evaluate_interactive(init)
    gym.close()


if __name__ == "__main__":
    main()
