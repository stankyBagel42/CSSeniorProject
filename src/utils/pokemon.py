import asyncio
import os
import random
from enum import Enum
from pathlib import Path

import poke_env.exceptions
from poke_env import AccountConfiguration
from poke_env.data import GenData
from poke_env.environment import Pokemon
from poke_env.player import RandomPlayer
from poke_env.teambuilder import ConstantTeambuilder

from src.utils.general import repo_root

POKEMON_IDX_MAP = {}

GEN_DATA = GenData(4)


class STAT_IDX(Enum):
    """Stat index mapping for game state vector"""
    ATK = 0
    DEF = 1
    SPA = 2
    SPD = 3
    SPE = 4
    EVASION = 5
    ACCURACY = 6


class SideCondition(Enum):
    """Pared down version of poke-env side condition since we don't need every possible value."""
    LIGHT_SCREEN = 0
    REFLECT = 1
    SAFEGUARD = 2
    SPIKES = 3
    STEALTH_ROCK = 4
    TAILWIND = 5
    TOXIC_SPIKES = 6

    def is_stackable(self) -> bool:
        """Poke-env has a weird thing where the value of the side condition dictionary is the turn a condition was
        played if it isn't stackable. I just need if it is present, and the stacks otherwise."""
        return self in [SideCondition.SPIKES, SideCondition.TOXIC_SPIKES]


class Field(Enum):
    """Pared down version of poke-env side fields since we don't need every possible value."""
    GRAVITY = 0
    MUD_SPORT = 1
    TRICK_ROOM = 2
    WATER_SPORT = 3


class Weather(Enum):
    HAIL = 0
    RAINDANCE = 1
    SANDSTORM = 2
    SUNNYDAY = 3


class NotableAbility(Enum):
    """Abilities that change the way the pokemon takes damage or otherwise significantly impact gameplay"""
    LEVITATE = 0
    VOLTABSORB = 1
    WATERABSORB = 2
    FLASHFIRE = 3
    MAGICGUARD = 4


def test_vs_bot(player, teams: list[str], n_challenges: int = 100, baseline_player=None,
                format: str = "gen4anythinggoes") -> float:
    """Battle the given player against a baseline player for n_challenges times. Useful for benchmarking performance
    over time when training"""

    original_wins = baseline_player.n_won_battles
    original_battles = baseline_player.n_finished_battles
    if baseline_player is None:
        baseline_player = RandomPlayer(battle_format=format)

    for i in range(n_challenges):
        baseline_player._team = ConstantTeambuilder(random.choice(teams))
        asyncio.get_event_loop().run_until_complete(baseline_player.battle_against(player, n_battles=1))

    new_wins = baseline_player.n_won_battles - original_wins
    new_battles = baseline_player.n_finished_battles - original_battles
    return 1 - (new_wins / new_battles)


def pokemon_to_index(pokemon_obj: Pokemon) -> int:
    global POKEMON_IDX_MAP
    if len(POKEMON_IDX_MAP.keys()) == 0:
        with open(repo_root / 'all_pokemon.csv', 'r') as inp:
            for row in inp.readlines()[1:]:
                pokemon, idx = row.split(',')
                idx = int(idx)
                POKEMON_IDX_MAP[pokemon.lower()] = idx

    return POKEMON_IDX_MAP[pokemon_obj.species.lower()]


def run_showdown_cmd(cmd: str, args: str) -> str:
    """Run a command using pokemon showdown CLI and Node.js. THIS PREPENDS 'node pokemon-showdown' TO YOUR COMMAND"""
    showdown_root = repo_root / 'pokemon-showdown'
    cwd = os.getcwd()
    os.chdir(showdown_root)
    ret = os.system(f"node pokemon-showdown {cmd} {args}")

    os.chdir(cwd)
    return ret


def create_player(player_class, username: str = None, **kwargs):
    """Creates a player with the given username, increments (or adds) the last number if the user is taken. it also
    limits the username size to -18"""
    count = 0
    if username is None:
        name = player_class.__name__
    else:
        name = username
    while True:
        # create a new player object with the given configuration
        try:
            if count == 0:
                account = AccountConfiguration(name[:18], None)
            else:
                for i in name:
                    if i.isnumeric():
                        break
                count_len = len(str(count))
                account = AccountConfiguration(name[:name.index(i)][:18 - count_len] + str(count), None)
                assert len(account.username) <= 18, f"{account.username} is too long!"

            player = player_class(account_configuration=account, **kwargs)
            asyncio.get_event_loop().run_until_complete(player.ps_client.wait_for_login(wait_for=10))

            if player.ps_client.logged_in.is_set():
                return player
        except poke_env.exceptions.ShowdownException | NameError:
            print(f"IGNORED AN EXCEPTION")
        if not name[-1].isnumeric():
            # max size is -18
            name = name[:18] + "0"
        count += 1