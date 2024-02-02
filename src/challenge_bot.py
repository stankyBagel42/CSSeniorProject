import asyncio

from poke_env import AccountConfiguration

from src.poke_env_classes import TrainedRLPlayer, MultiTeambuilder
from src.utils.general import repo_root, get_packed_teams


async def challenge_human(player: TrainedRLPlayer, opponent: str, n_battles: int = 1):
    await player.send_challenges(opponent, n_battles)


if __name__ == '__main__':
    model_path = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\2402001_1mil_Resumed\player_1\PokeNet_149.pt"
    opponent_username = "stankyBagel42"
    loop = asyncio.get_event_loop()

    bot_account = AccountConfiguration("RL Trained Bot", None)

    teams = MultiTeambuilder(get_packed_teams(repo_root / 'packed_teams'))
    pokemon_bot = TrainedRLPlayer(
        model=model_path,
        account_configuration=bot_account,
        battle_format='gen4anythinggoes',
        team=teams
    )

    loop.run_until_complete(challenge_human(pokemon_bot, opponent_username))
