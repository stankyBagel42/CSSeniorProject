import asyncio
import random

from poke_env import AccountConfiguration
from poke_env.data import GenData
from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder
from tabulate import tabulate

from src.poke_env_classes import MultiTeambuilder, TrainedRLPlayer
from src.utils.general import repo_root, get_packed_teams

GEN_DATA = GenData(4)

class MaxDamagePlayer(Player):

    # Same method as in previous examples
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            max_dmg = -1
            max_move = None
            for move in battle.available_moves:
                type_multiplier = move.type.damage_multiplier(battle.opponent_active_pokemon.type_1,
                                                         battle.opponent_active_pokemon.type_2,
                                                         type_chart=GEN_DATA.type_chart)
                if move.base_power * type_multiplier > max_dmg:
                    max_dmg = move.base_power * type_multiplier
                    max_move = move
            return self.create_order(max_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

def single_battle(player:TrainedRLPlayer, opponent: Player, teams:list[str]) -> bool:
    """Returns true if the player won, false if they didn't"""
    player._team = ConstantTeambuilder(random.choice(teams))
    opponent._team = ConstantTeambuilder(random.choice(teams))
    wins_before = player.n_won_battles
    asyncio.get_event_loop().run_until_complete(player.battle_against(opponent))
    wins_after = player.n_won_battles
    return wins_after > wins_before




def benchmark_player(player:TrainedRLPlayer, teams:list[str],n_challenges:int=100) -> dict[str,float]:
    """Each player in players plays against the others n_challenges times, the resulting win rates are stored in a
    dictionary mapping opponents to the winrate against that opponent."""
    battle_format = player.format
    players = [
        MaxDamagePlayer(battle_format=battle_format, team=MultiTeambuilder(teams)),
        RandomPlayer(battle_format=battle_format, team=MultiTeambuilder(teams),
                     account_configuration=AccountConfiguration("Random Benchmark", None)),
        MaxBasePowerPlayer(battle_format=battle_format, team=MultiTeambuilder(teams)),
        SimpleHeuristicsPlayer(battle_format=battle_format, team=MultiTeambuilder(teams)),
    ]

    winrates = {}

    for opponent in players:
        wins = 0
        for i in range(n_challenges):
            player_won = single_battle(player, opponent, teams=teams)
            if player_won:
                wins += 1
        winrates[opponent.username] = wins/n_challenges

    return winrates


if __name__ == '__main__':
    MODEL_PATH = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\2402002_BigState_1mil\player_1\PokeNet_26.pt"
    NUM_CHALLENGES = 100

    packed_teams = get_packed_teams(repo_root / 'packed_teams')
    teambuilder = MultiTeambuilder(packed_teams)
    p1 = AccountConfiguration("RL 1",None)

    format = 'gen4anythinggoes'
    rl_player = TrainedRLPlayer(model=MODEL_PATH,battle_format=format, team=teambuilder, account_configuration=p1)

    cross_eval_results = benchmark_player(rl_player, packed_teams, NUM_CHALLENGES)


    table = [["-"] + [p.username for p in cross_eval_results.keys()]]
    for p_1, results in cross_eval_results.items():
        table.append([p_1] + [cross_eval_results[p_1][p_2] for p_2 in results])
    print("Cross evaluation of 2 DDQNs with baselines:")
    print(tabulate(table))