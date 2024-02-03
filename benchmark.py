import asyncio

from poke_env import AccountConfiguration
from poke_env.data import GenData
from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer, cross_evaluate
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


def benchmark_player(player:TrainedRLPlayer, n_challenges:int=100, teambuilder=None) -> dict[str,dict[str,float]]:
    """Each player in players plays against the others n_challenges times, the resulting win rates are stored in a
    dictionary of dictionaries."""
    if teambuilder is None:
        teambuilder = MultiTeambuilder(get_packed_teams(repo_root / "packed_teams"))
    battle_format = player.format
    players = [
        player,
        MaxDamagePlayer(battle_format=battle_format, team=MultiTeambuilder(get_packed_teams(repo_root / "packed_teams"))),
        RandomPlayer(battle_format=battle_format, team=MultiTeambuilder(get_packed_teams(repo_root / "packed_teams")),
                     account_configuration=AccountConfiguration("Random Benchmark", None)),
        MaxBasePowerPlayer(battle_format=battle_format, team=MultiTeambuilder(get_packed_teams(repo_root / "packed_teams"))),
        SimpleHeuristicsPlayer(battle_format=battle_format, team=MultiTeambuilder(get_packed_teams(repo_root / "packed_teams"))),
    ]

    cross_eval_results = asyncio.get_event_loop().run_until_complete(cross_evaluate(players, n_challenges=n_challenges))

    return cross_eval_results


if __name__ == '__main__':
    MODEL_PATH = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\2402002_BigState_1mil\player_1\PokeNet_26.pt"
    NUM_CHALLENGES = 100

    packed_teams = get_packed_teams(repo_root / 'packed_teams')
    teambuilder = MultiTeambuilder(packed_teams)
    p1 = AccountConfiguration("RL 1",None)

    format = 'gen4anythinggoes'
    rl_player = TrainedRLPlayer(model=MODEL_PATH,battle_format=format, team=teambuilder, account_configuration=p1)

    cross_eval_results = benchmark_player(rl_player, NUM_CHALLENGES, teambuilder)


    table = [["-"] + [p.username for p in cross_eval_results.keys()]]
    for p_1, results in cross_eval_results.items():
        table.append([p_1] + [cross_eval_results[p_1][p_2] for p_2 in results])
    print("Cross evaluation of 2 DDQNs with baselines:")
    print(tabulate(table))