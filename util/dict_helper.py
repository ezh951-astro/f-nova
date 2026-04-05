import yaml
import copy

def unflatten(d):

    result = {}

    for key, value in d.items():

        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    return result

def merge_config_dict(base, override):

    #result = base    # Note mutation in place
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_config_dict(result[key], value)
        else:
            result[key] = value

    return result

def get_dotted(d, key):
    for part in key.split("."):
        d = d[part]
    return d

def extract_dotted_keys(d, keys):
    return {k: get_dotted(d, k) for k in keys}



def load_yaml_required(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping (dict) and cannot be empty")

    return data