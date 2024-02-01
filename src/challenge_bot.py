import asyncio

from poke_env import AccountConfiguration

from src.poke_env_classes import TrainedRLPlayer


async def challenge_human(player: TrainedRLPlayer, opponent:str, n_battles:int=1):
    await player.send_challenges(opponent, n_battles)


if __name__ == '__main__':
    model_path = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\240131_TestRun\player_1\PokeNet_33.chkpt"
    opponent_username = "stankyBagel42"
    loop = asyncio.get_event_loop()

    bot_account = AccountConfiguration("RL Trained Bot", None)
    pokemon_bot = TrainedRLPlayer(
        model=model_path,
        account_configuration=bot_account,
    )

    loop.run_until_complete(challenge_human(pokemon_bot, opponent_username))
