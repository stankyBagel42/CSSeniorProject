from collections import deque
from os import PathLike
from pathlib import Path
import random
from queue import Queue

import numpy as np
import torch
import yaml

repo_root = Path(__file__).absolute().parents[2]


class RunningAvg:
    def __init__(self, n):
        """Keeps a running average of all keys in dictionaries passed to the log() method"""
        self.n = n
        self.mem = None

    def log(self, log_dict: dict) -> dict:
        if self.mem is None:
            self.mem = {key: deque([val], maxlen=self.n) for key, val in log_dict.items() if
                        isinstance(val, int | float | np.number)}

        avg = {}
        for key, val in log_dict.items():
            if isinstance(val, int | float | np.number):
                self.mem[key].append(val)
            avg[key] = sum(self.mem[key]) / len(self.mem[key])

        return avg


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_packed_team(team_name: str = None):
    """Get a packed team, returns a random one if the given team name is none."""
    team_dir = repo_root / 'packed_teams'

    assert team_dir.exists(), "Before getting a packed team, you must run 'split_teams.py'"

    team_file = f"{team_name}.txt" if team_name is not None else random.choice(list(team_dir.iterdir()))

    with open(team_file, 'r') as team_inp:
        team_str = team_inp.read()

    return team_str


def read_yaml(yaml_filepath: str | Path) -> dict:
    """Read a yaml formatted file"""
    with open(yaml_filepath, 'r') as inp:
        data = yaml.load(inp, yaml.FullLoader)

    # convert path strings to path objects
    for key, val in data.items():
        if isinstance(val, str) and Path(val).exists():
            data[key] = Path(val)
    return data


def write_yaml(data: dict, yaml_filepath: str | Path):
    """Write a dictionary to a yaml formatted file"""
    # copy the data so we don't modify the original
    to_write = data.copy()

    # replace paths with strings
    for key, val in data.items():
        if isinstance(val, Path):
            to_write[key] = str(val)

    with open(yaml_filepath, 'w') as outp:
        yaml.dump(to_write, outp)


def get_packed_teams(packed_teams_dir: Path) -> list[str]:
    """Get a list of all packed team strings from the given directory"""
    packed_teams = []
    for txt in packed_teams_dir.iterdir():
        with open(txt, 'r') as team_inp:
            team = team_inp.read()
            packed_teams.append(team)
    return packed_teams


def latest_ckpt_file(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    # find maximum number in the filename
    nums = []
    for file in checkpoint_dir.iterdir():
        checkpoint_num = file.stem.split('_')[-1].split('.')[0]
        nums.append(int(checkpoint_num))

    # get that file path
    stem = f'PokeNet_{max(nums)}*'
    max_file = list(checkpoint_dir.glob(stem))[0]
    return max_file


def load_pytorch(filepath: PathLike, device: str = None, queue: Queue = None):
    """
    Loads the data in the given filepath with pytorch, used when reading multiple files at once in different
    threads
    """
    data = torch.load(filepath, map_location=device)
    if queue is not None:
        queue.put(data)
    else:
        return data


def batch_iter(iterable, n=1):
    """Batch the given iterable, so it returns constant sized chunks"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
