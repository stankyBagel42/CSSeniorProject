import asyncio
import pprint
import random

from poke_env import AccountConfiguration
from poke_env.data import GenData
from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder
from tqdm import tqdm

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


def two_battles(player: TrainedRLPlayer, opponent: Player, teams: list[str]) -> int:
    """Battles the two opponents with randomly chosen teams, where they play once and then swap teams and play again.
    Returns # of times the 'player' won."""
    loop = asyncio.get_event_loop()
    if 'random' in player.format.lower():
        wins_before = player.n_won_battles
        loop.run_until_complete(player.battle_against(opponent))
        loop.run_until_complete(player.battle_against(opponent))
        wins_after = player.n_won_battles
        wins = wins_after-wins_before
    else:
        # choose 2 random teams
        team_1 = random.choice(teams)
        team_2 = random.choice(teams)

        # run the first game and record the result
        player._team = ConstantTeambuilder(team_1)
        opponent._team = ConstantTeambuilder(team_2)
        wins_before = player.n_won_battles
        loop.run_until_complete(player.battle_against(opponent))
        wins_after = player.n_won_battles
        player_won = 1 if wins_after > wins_before else 0

        # swap teams and run the second game, recording the result
        player._team = ConstantTeambuilder(team_2)
        opponent._team = ConstantTeambuilder(team_1)
        wins_before = player.n_won_battles
        loop.run_until_complete(player.battle_against(opponent))
        wins_after = player.n_won_battles
        player_won2 = 1 if wins_after > wins_before else 0
        wins = player_won + player_won2
    return wins


def benchmark_player(player: TrainedRLPlayer, teams: list[str], n_challenges: int = 100, tqdm_pos:int=0, players:list[Player]=None) -> dict[str, float]:
    """Each player in players plays against the others n_challenges times, the resulting win rates are stored in a
    dictionary mapping opponents to the winrate against that opponent."""
    battle_format = player.format
    if players is None:
        players = [
            RandomPlayer(account_configuration=AccountConfiguration(f"Random{tqdm_pos}", None), battle_format=battle_format,
                         team=MultiTeambuilder(teams)),
            MaxDamagePlayer(account_configuration=AccountConfiguration(f"MaxDmg{tqdm_pos}", None), battle_format=battle_format,
                            team=MultiTeambuilder(teams)),
            MaxBasePowerPlayer(account_configuration=AccountConfiguration(f"MaxPow{tqdm_pos}", None), battle_format=battle_format,
                               team=MultiTeambuilder(teams)),
            SimpleHeuristicsPlayer(account_configuration=AccountConfiguration(f"Heuristics{tqdm_pos}", None),
                                   battle_format=battle_format,
                                   team=MultiTeambuilder(teams))
        ]

    winrates = {}

    for opponent in players:
        wins = 0
        pbar = tqdm(total=n_challenges, desc=f"{opponent.username} Battles:", position=tqdm_pos)
        for i in range(n_challenges // 2):
            num_wins = two_battles(player, opponent, teams=teams)
            wins += num_wins
            pbar.update(2)

        winrates[opponent.username] = wins / n_challenges

    return winrates


if __name__ == '__main__':
    MODEL_PATH = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\240323_NoArgMax_2milLargeNet\player_1\PokeNet_1355000.pt"
    NUM_CHALLENGES = 1000

    packed_teams = get_packed_teams(repo_root / 'packed_teams')
    teambuilder = MultiTeambuilder(packed_teams)
    p1 = AccountConfiguration("RL 1", None)

    format = 'gen4anythinggoes'
    rl_player = TrainedRLPlayer(model=MODEL_PATH, battle_format=format, team=teambuilder, account_configuration=p1)

    cross_eval_results = benchmark_player(rl_player, packed_teams, NUM_CHALLENGES)
    print(f"RESULTS FOR {MODEL_PATH}:")
    pprint.pp(cross_eval_results)
