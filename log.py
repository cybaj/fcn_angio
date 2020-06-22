import os

_base_log_dir = os.path.join("logs")
_target_log_file_name = "log.csv"

_log_start_size = 1

_key_index = {
    "train": {
        "start_index": 7,
        "end_index": 11,
        "length": 5,
        "indice": [
            {
                "index": 2,
                "key": "train/loss"
            },
            {
                "index": 3,
                "key": "train/acc"
            },
            {
                "index": 4,
                "key": "train/acc_cls"
            },
            {
                "index": 5,
                "key": "train/mean_iu"
            },
            {
                "index": 6,
                "key": "train/fwavacc"
            },
        ]
    },
    "validation": {
        "start_index": 2,
        "end_index": 6,
        "length": 5,
        "indice": [
            {
                "index": 7,
                "key": "valid/loss"
            },
            {
                "index": 8,
                "key": "valid/acc"
            },
            {
                "index": 9,
                "key": "valid/acc_cls"
            },
            {
                "index": 10,
                "key": "valid/mean_iu"
            },
            {
                "index": 11,
                "key": "valid/fwavacc"
            },
            ]
    },
    "indice": {
        "iteration": {
            "index": 1
        },
        "epoch": {
            "index": 0
        },
        "elapsed_time": {
            "index": 12
        }
    }
}

_linetype = {
                "validation": {"empty_indice": [item["index"] for item in _key_index["train"]["indice"]]}, 
                "train": {"empty_indice": [item["index"] for item in _key_index["validation"]["indice"]]}
            }

def has_empty(splitted, counter, policy="only"):
    if policy == 'only':
        for _target_index in counter:
            if not splitted[_target_index]:
                return True 

def check_iter_right(line, iter_num):
    splitted = []
    if type(line) == str:
        splitted = line.split(',')
    else:
        splitted = line
    return iter_num == int(splitted[_key_index['indice']['iteration']['index']])

def get_linetype(line):
    _policy = 'only'
    splitted = []
    if type(line) == str:
        splitted = line.split(',')
    else:
        splitted = line
    must_empty_indice_in_validation = _linetype["validation"]["empty_indice"]
    
    if has_empty(splitted, must_empty_indice_in_validation, policy=_policy):
        return "validation"
    else:
        return "train"

def parse_line(line, target="validation"):
    splitted = []
    if type(line) == str:
        splitted = line.split(',')
    else:
        splitted = line
    assert target in ("validation", "train")
    return {item["key"] : {"value": splitted[item["index"]], "iter_num": int(splitted[_key_index['indice']['iteration']['index']]), "index": item["index"]} for item in _key_index[target]["indice"]}

def parse_iter(path, iter_num, target="validation"):
    with open(path, 'r') as fp:
        total_lines = fp.readlines()[_log_start_size + iter_num:]
        for line in total_lines:
            if target == get_linetype(line) and check_iter_right(line, iter_num):
                return parse_line(line, target=target)
            else:
                continue
        raise

def make_log_path(logdir):
    return os.path.join(_base_log_dir, logdir, _target_log_file_name)

