import asyncio

from poke_env import AccountConfiguration

from src.poke_env_classes import TrainedRLPlayer, MultiTeambuilder
from src.utils.general import repo_root, get_packed_teams
from src.utils.pokemon import create_player


async def challenge_human(player: TrainedRLPlayer, opponent: str, n_battles: int = 1):
    await player.send_challenges(opponent, n_battles)


if __name__ == '__main__':
    # set the model path, can use `latest_ckpt_file(PLAYER_CHECKPOINT_DIR)` for the latest saved checkpoint
    # model_path = latest_ckpt_file(r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\2402005_2mil_HighBatchHighMem\player_1")
    model_path = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\2402016_EstimatedMatchups_2mil\player_1\PokeNet_1215000.pt"

    # the username of the account to challenge
    opponent_username = "stankyBagel42"


    loop = asyncio.get_event_loop()

    bot_account = AccountConfiguration("RL Trained Bot", None)

    teams = MultiTeambuilder(get_packed_teams(repo_root / 'packed_teams'))
    pokemon_bot = create_player(
        TrainedRLPlayer,
        "RL Trained Bot",
        model=model_path,
        battle_format='gen4anythinggoes',
        team=teams,
        use_argmax=True
    )

    loop.run_until_complete(challenge_human(pokemon_bot, opponent_username))
