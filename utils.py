import distutils
import json

from types import SimpleNamespace as Namespace


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return setattr_recursive(getattr(obj, split[0]), split[1:], val) if len(split) > 1 else setattr(obj, split[0], val)


def generate_config(params, file_path):
    print("Saving Configs")
    f = open(file_path, "w")
    json_data = json.dumps(params.__dict__, default=lambda o: o.__dict__, indent=4)
    f.write(json_data)
    f.close()


def read_config(config_path):
    print('Parse Params file here from ', config_path, ' and pass into main')
    json_data = open(config_path, "r").read()
    return json.loads(json_data, object_hook=lambda d: Namespace(**d))


def override_params(params, overrides):
    assert (len(overrides) % 2 == 0)
    for k in range(0, len(overrides), 2):
        oldval = getattr_recursive(params, overrides[k])
        if type(oldval) == bool:
            to_val = bool(distutils.util.strtobool(overrides[k + 1]))
        else:
            to_val = type(oldval)(overrides[k + 1])
        setattr_recursive(params, overrides[k],
                          to_val)
        print("Overriding param", overrides[k], "from", oldval, "to", to_val)

    return params


def get_bool_user(message, default: bool):
    if default:
        default_string = '[Y/n]'
    else:
        default_string = '[y/N]'
    resp = input('{} {}\n'.format(message, default_string))
    try:
        if distutils.util.strtobool(resp):
            return True
        else:
            return False
    except ValueError:
        return default
