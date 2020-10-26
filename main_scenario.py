import argparse
import os

import numpy as np

from src.CPP.Environment import CPPEnvironment
from src.DH.Environment import DHEnvironment
from utils import override_params, read_config


def scenario_cpp(args, params):
    env = CPPEnvironment(params)
    env.agent.load_weights(args.weights)
    env.test_episode()
    env.display.display_episode(env.grid.map_image, env.stats.trajectory, plot=True)


def scenario_dh(args, params):
    env = DHEnvironment(params)
    env.agent.load_weights(args.weights)
    env.test_episode()
    env.display.display_episode(env.grid.map_image, env.stats.trajectory, plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to weights')
    parser.add_argument('--config', required=True, help='Config file for agent shaping')
    parser.add_argument('--seed', default=None, help="Seed for repeatability")
    parser.add_argument('--params', nargs='*', default=None)

    # DH Params
    parser.add_argument('--dh', action='store_true', help='Run Path Planning for Data Harvesting')

    # CPP Params
    parser.add_argument('--cpp', action='store_true', help='Run Coverage Path Planning')

    args = parser.parse_args()

    if args.seed:
        np.random.seed(int(args.seed))

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    params.model_stats_params.save_model = "models/save"
    if args.seed:
        params.model_stats_params.log_file_name = "scenario_" + str(args.seed)
    else:
        params.model_stats_params.log_file_name = "scenario_random"

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.cpp:
        scenario_cpp(args, params)
    elif args.dh:
        scenario_dh(args, params)

