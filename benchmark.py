import asyncio

from poke_env import AccountConfiguration
from poke_env.data import GenData
from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer, cross_evaluate
from tabulate import tabulate

from src.poke_env_classes import MultiTeambuilder, TrainedRLPlayer
from src.utils import repo_root, get_packed_teams

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

async def main():
    packed_teams = get_packed_teams(repo_root / 'packed_teams')
    teambuilder = MultiTeambuilder(packed_teams)
    p1 = AccountConfiguration("RL 1",None)
    p2 = AccountConfiguration("RL 2",None)

    format = 'gen4anythinggoes'
    rl_player = TrainedRLPlayer(model=MODEL_PATH,battle_format=format, team=teambuilder, account_configuration=p1)
    rl_player2 = TrainedRLPlayer(model=MODEL2_PATH,battle_format=format, team=teambuilder, account_configuration=p2)

    players = [
        rl_player,
        MaxDamagePlayer(battle_format=format, team=teambuilder),
        RandomPlayer(battle_format=format, team=teambuilder, account_configuration=AccountConfiguration("Random Benchmark",None)),
        MaxBasePowerPlayer(battle_format=format, team=teambuilder),
        SimpleHeuristicsPlayer(battle_format=format, team=teambuilder),
        rl_player2,
    ]
    cross_eval_results = await cross_evaluate(players, n_challenges=NUM_CHALLENGES)
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_eval_results.items():
        table.append([p_1] + [cross_eval_results[p_1][p_2] for p_2 in results])
    print("Cross evaluation of 2 DDQNs with baselines:")
    print(tabulate(table))

if __name__ == '__main__':
    MODEL_PATH = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\2402001_1mil_Resumed\player_1\PokeNet_149.pt"
    MODEL2_PATH = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\2402001_1mil_Resumed\player_2\PokeNet_149.pt"
    NUM_CHALLENGES = 100
    asyncio.get_event_loop().run_until_complete(main())