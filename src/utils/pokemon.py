import asyncio
import os
import random
from enum import Enum
from typing import TypeVar, Type

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


class SideCondition(Enum):
    """Pared down version of poke-env side condition since we don't need every possible value."""
    LIGHT_SCREEN = 0
    REFLECT = 1
    STEALTH_ROCK = 2
    TAILWIND = 3
    TOXIC_SPIKES = 4
    SPIKES = 2 # spikes = stealth rock, since they don't appear in our games

    def is_stackable(self) -> bool:
        """Poke-env has a weird thing where the value of the side condition dictionary is the turn a condition was
        played if it isn't stackable. I just need if it is present, and the stacks otherwise."""
        return self == SideCondition.TOXIC_SPIKES


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
    """Abilities that change the way the Pokemon takes damage or otherwise significantly impact gameplay"""
    LEVITATE = 0
    VOLTABSORB = 1
    WATERABSORB = 2
    FLASHFIRE = 3
    MAGICGUARD = 4


class Status(Enum):
    """Status Effects on Pokemon"""
    BRN = 0
    FRZ = 1
    FNT = 6
    PAR = 2
    PSN = 3
    SLP = 4
    TOX = 5


def test_vs_bot(player, teams: list[str], n_challenges: int = 100, baseline_player=None,
                format: str = "gen4anythinggoes") -> float:
    """Battle the given player against a baseline player for n_challenges times. Useful for benchmarking performance
    over time when training. The players each play teams once before new teams are selected to help remove any team
    strength bias."""
    orig_team = player._team
    original_wins = baseline_player.n_won_battles
    original_battles = baseline_player.n_finished_battles
    if baseline_player is None:
        baseline_player = RandomPlayer(battle_format=format)

    # random formats don't need the team setup, so they can just play
    if 'random' in format.lower():
        asyncio.get_event_loop().run_until_complete(baseline_player.battle_against(player, n_battles=n_challenges))
    else:
        for i in range(n_challenges // 2):
            team_1 = random.choice(teams)
            team_2 = random.choice(teams)

            # run the first battle
            player._team = ConstantTeambuilder(team_1)
            baseline_player._team = ConstantTeambuilder(team_2)
            asyncio.get_event_loop().run_until_complete(baseline_player.battle_against(player, n_battles=1))

            # swap teams and try again
            player._team = ConstantTeambuilder(team_2)
            baseline_player._team = ConstantTeambuilder(team_1)
            asyncio.get_event_loop().run_until_complete(baseline_player.battle_against(player, n_battles=1))

    new_wins = baseline_player.n_won_battles - original_wins
    new_battles = baseline_player.n_finished_battles - original_battles
    player._team = orig_team
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

# template type
T = TypeVar('T')
def create_player(player_class: Type[T], username: str = None, **kwargs) -> T:
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
            asyncio.get_event_loop().run_until_complete(
                player.ps_client.wait_for_login(checking_interval=0.05, wait_for=15))

            if player.ps_client.logged_in.is_set():
                return player
        except poke_env.exceptions.ShowdownException | NameError | AssertionError:
            print(f"IGNORED AN EXCEPTION WHEN LOGGING IN")
        if not name[-1].isnumeric():
            # max size is -18
            name = name[:18] + "0"
        count += 1
