import argparse
import os

import numpy as np

from src.CPP.Environment import CPPEnvironment
from src.DH.Environment import DHEnvironment
from utils import override_params, read_config, get_bool_user


def save_video(env):
    if get_bool_user('Save run?', False):
        save_as_default = "video.mp4"
        save_as = input('Save as: [{}]\n'.format(save_as_default))
        if save_as == '':
            save_as = save_as_default
        frame_rate_default = 4
        frame_rate = input('Frame rate: [{}]\n'.format(frame_rate_default))
        if frame_rate == '':
            frame_rate = frame_rate_default
        frame_rate = int(frame_rate)

        draw_path = get_bool_user('Show path?', False)
        env.display.create_video(env.grid.map_image, env.stats.trajectory, save_as, frame_rate, draw_path=draw_path)


def scenario(args, params):
    env = None
    if args.cpp:
        env = CPPEnvironment(params)
    elif args.dh:
        env = DHEnvironment(params)
    else:
        print("Need --cpp or --dh")
        exit(1)

    env.agent.load_weights(args.weights)

    init_state = None
    if args.scenario:
        scenario = read_config(args.scenario)
        init_state = env.grid.create_scenario(scenario)

    env.test_episode(init_state)
    env.display.display_episode(env.grid.map_image, env.stats.trajectory, plot=True)
    if args.video:
        save_video(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to weights')
    parser.add_argument('--config', required=True, help='Config file for agent shaping')
    parser.add_argument('--scenario', default=None, help='Config file for scenario')
    parser.add_argument('--seed', default=None, help="Seed for repeatability")
    parser.add_argument('--video', action='store_true', help="Will ask to create video after plotting")
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

    scenario(args, params)
