import asyncio
import os
from enum import Enum

from poke_env.data import GenData
from poke_env.environment import Pokemon
from poke_env.player import RandomPlayer

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



def test_vs_random(player, n_challenges: int = 100, random_player: RandomPlayer = None,
                   format: str = "gen4anythinggoes", teambuilder=None) -> float:
    """Battle the given player against a random player for n_challenges times. Useful for benchmarking performance
    over time when training"""
    if random_player is None:
        random_player = RandomPlayer(battle_format=format, team=teambuilder)
    asyncio.get_event_loop().run_until_complete(random_player.battle_against(player, n_battles=n_challenges))

    return 1 - random_player.win_rate


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
