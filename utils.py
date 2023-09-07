import argparse
import os
import distutils.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, Tuple

import numpy as np
from dataclasses_json import DataClassJsonMixin

import tensorflow as tf

from src.trainer.utils import ParamType


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')

    try:
        return getattr_recursive_(obj, split)
    except KeyError:
        split.insert(0, 'params')
        return getattr_recursive_(obj, split)


def getattr_recursive_(obj, split):
    if isinstance(obj, dict):
        if len(split) > 1:
            return getattr_recursive(obj[split[0]], split[1:])
        else:
            return obj[split[0]]
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if not isinstance(s, list):
        s = s.split('/')

    if isinstance(obj, dict):
        if not s[0] in obj:
            s.insert(0, 'params')
        if len(s) > 1:
            return setattr_recursive(obj[s[0]], s[1:], val)
        else:
            obj[s[0]] = val
            return None
    if not hasattr(obj, s[0]):
        s.insert(0, 'params')
    return setattr_recursive(getattr(obj, s[0]), s[1:], val) if len(s) > 1 else setattr(obj, s[0],
                                                                                        val)


def get_bool_user(message, default: bool):
    if default:
        default_string = '[Y/n]'
    else:
        default_string = '[y/N]'
    resp = input('{} {}\n'.format(message, default_string))
    try:
        return distutils.util.strtobool(resp)
    except ValueError:
        return default


def get_value_user(message, default):
    resp = input(f'{message} [{default}]\n')
    if len(resp) == 0:
        return default
    return type(default)(resp)


def dict_mean(dict_list):
    mean_dict = {}
    elem1 = dict_list[0]
    for key in elem1.keys():
        if isinstance(elem1[key], str):
            continue
        try:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        except KeyError:
            continue
    return mean_dict


class AbstractParams(DataClassJsonMixin):

    def __init__(self):
        self.log_dir = None

    def save_to(self, config_path):
        js = self.to_dict()
        with open(config_path, 'w') as f:
            json.dump(js, f, indent=4)

    @classmethod
    def read_from(cls, config_path):
        with open(config_path, 'r') as f:
            js = json.load(f)
            params = cls.from_dict(js)
            return params

    def override_params(self, overrides):
        return override_params(self, overrides)

    @classmethod
    def from_parsed_args(cls, args):
        if args.generate or not os.path.isfile(args.config):
            cls().save_to(args.config)
            print(f"Saved config to {args.config}")
            exit(0)
        params = cls.read_from(args.config)
        if args.params is not None:
            params.override_params(args.params)

        if not args.gpu and args.gpu_id is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                gpu_id = int(args.gpu_id) if args.gpu_id is not None else 0
                gpu_used = physical_devices[gpu_id]
                tf.config.set_visible_devices(gpu_used, 'GPU')
                tf.config.experimental.set_memory_growth(gpu_used, True)
                print('Using following GPU: ', gpu_used.name)
            except:
                print("Invalid device or cannot modify virtual devices once initialized. Not too good probably")
                exit(0)
                pass
        return params, args

    @classmethod
    def from_args(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser = cls.add_args_to_parser(parser)
        args = parser.parse_args()

        return cls.from_parsed_args(args)

    @classmethod
    def add_args_to_parser(cls, parser):
        parser.add_argument('--gpu', action='store_true', help='Activates usage of GPU')
        parser.add_argument('--gpu_id', default=None, help='Activates usage of GPU on specific GPU id')
        parser.add_argument('--id', default=None, help='Gives the log files a specific name, else config name')
        parser.add_argument('--generate', action='store_true', help='Generate config file for parameter class')
        parser.add_argument('--verbose', action='store_true', help='Prints the network summary at the start')
        parser.add_argument('--params', nargs='*', default=None,
                            help='Override parameters as: path/to/param1 value1 path/to/param2 value2 ...')
        parser.add_argument('config', help='Path to config file')
        return parser

    def create_folders(self, args, config_name="config.json"):
        run_id = args.id
        if run_id is None:
            run_id = args.config.split("/")[-1].split(".json")[0]

        self.log_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}/"
        os.makedirs(self.log_dir, exist_ok=True)
        model_dir = self.log_dir + 'models/'
        os.makedirs(model_dir, exist_ok=True)

        self.save_to(self.log_dir + config_name)

        return self.log_dir


def print_nn_summary(network_path):
    model = tf.keras.models.load_model(network_path)
    model.summary()


@dataclass
class FactoryParams(Generic[ParamType]):
    type: str
    params: ParamType


class Factory:

    @classmethod
    def registry(cls):
        raise NotImplementedError()

    @staticmethod
    def resolve_recursive(default, params):
        for key, value in params.items():
            if isinstance(value, dict):
                t = type(getattr(default, key))
                try:
                    Factory.resolve_recursive(t(), value)
                except TypeError:
                    continue
                params[key] = t(**value)
        # return params

    @classmethod
    def create(cls, params: dict, **kwargs):
        type_id = params["type"] if "type" in params.keys() else cls.defaults()[0]
        obj_type = cls.registry()[type_id]
        p = params["params"] if "params" in params.keys() else params
        if isinstance(p, dict):
            cls.resolve_recursive(obj_type.Params(), p)
            obj_params = obj_type.Params(**p)
        else:
            obj_params = p
        if "params" in params.keys():
            params["params"] = obj_params
        return obj_type(obj_params, **kwargs)

    @classmethod
    def default_params(cls):
        return FactoryParams[cls.defaults()[1].Params](type=cls.defaults()[0], params=cls.defaults()[1].Params())

    @classmethod
    def defaults(cls) -> Tuple[str, type]:
        default = list(cls.registry().items())[0]
        return default[0], default[1]

    @classmethod
    def default_param_type(cls):
        return FactoryParams[cls.defaults()[1].Params]

    @classmethod
    def type_ids(cls):
        return list(cls.registry().keys())

    @classmethod
    def param_types(cls):
        return [obj_type.Params for obj_type in cls.registry().values()]

    @classmethod
    def obj_types(cls):
        return [obj_type for obj_type, _ in cls.registry().values()]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def find_map(m):
    files = [m, f"{m}.png", f"res/{m}.png"]
    return find_file(m, files)


def find_scenario(scenario, exit_on_not_found=True):
    files = [scenario, f"{scenario}.pickle", f"{scenario}_init.pickle", f"example/scenarios/{scenario}_init.pickle"]
    return find_file(scenario, files, exit_on_not_found)


def find_config_model(model, exit_on_not_found=True):
    files = [f"{model}/config.json", f"logs/{model}/config.json", f"example/models/{model}/config.json"]
    return find_file(model, files, exit_on_not_found)


def find_file(name, files, exit_on_not_found=True):
    for file in files:
        if Path(file).is_file():
            return file
    print(f"Could not find {name}, tried all of {files}")
    if exit_on_not_found:
        exit(1)
    return None


def override_params(params, overrides):
    assert (len(overrides) % 2 == 0)
    for k in range(0, len(overrides), 2):
        try:
            oldval = getattr_recursive(params, overrides[k])
            if type(oldval) == bool:
                to_val = bool(distutils.util.strtobool(overrides[k + 1]))
            else:
                to_val = type(oldval)(overrides[k + 1])
            setattr_recursive(params, overrides[k],
                              to_val)
            print("Overriding param", overrides[k], "from", oldval, "to", to_val)
        except (KeyError, AttributeError):
            print("Could not override", overrides[k], "as it does not exist. Aborting.")
            exit(1)

    return params
