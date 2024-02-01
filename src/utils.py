import os
from pathlib import Path
import random

import yaml

repo_root = Path(__file__).absolute().parents[1]


def run_showdown_cmd(cmd:str, args:str) -> str:
    """Run a command using pokemon showdown CLI and node.JS. THIS PREPENDS 'node pokemon-showdown' TO YOUR COMMAND"""
    showdown_root = repo_root / 'pokemon-showdown'
    cwd = os.getcwd()
    os.chdir(showdown_root)
    ret = os.system(f"node pokemon-showdown {cmd} {args}")
    # process = subprocess.run(["node", "pokemon-showdown", cmd, args], capture_output=True, encoding='utf-8', cwd=str(showdown_root))

    os.chdir(cwd)
    return ret

def get_packed_team(team_name:str = None):
    """Get a packed team, returns a random one if the given team name is none."""
    team_dir = repo_root / 'packed_teams'

    assert team_dir.exists(), "Before getting a packed team, you must run 'split_teams.py'"

    team_file = f"{team_name}.txt" if team_name is not None else random.choice(list(team_dir.iterdir()))

    with open(team_file, 'r') as team_inp:
        team_str = team_inp.read()

    return team_str


def read_yaml(yaml_filepath:str | Path) -> dict:
    """Read a yaml formatted file"""
    with open(yaml_filepath, 'r') as inp:
        data = yaml.load(inp, yaml.FullLoader)
    return data


