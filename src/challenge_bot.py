import csv

from poke_env import AccountConfiguration
from poke_env.environment import Battle

from src.poke_env_classes import TrainedRLPlayer, MultiTeambuilder
from src.utils.general import repo_root, get_packed_teams

import asyncio


async def challenge_human(player: TrainedRLPlayer, opponent: str, n_battles: int = 1):
    await player.send_challenges(opponent, n_battles)


if __name__ == '__main__':
    # set the model path, can use `latest_ckpt_file(PLAYER_CHECKPOINT_DIR)` for the latest saved checkpoint
    model_path = r"C:\Users\Eric\Desktop\proj\CSSeniorProject\checkpoints\confused-sweep-20\player_1\PokeNet_245000.pt"
    # save battle results to repo/data/bot_challenge_log.csv
    save_log = True
    # the username of the account to challenge
    opponent_username = "stankyBagel42"
    NUM_CHALLENGES = 1

    # get loop so we can run the async code
    loop = asyncio.get_event_loop()
    bot_account = AccountConfiguration("RL Trained Bot", None)

    teams = MultiTeambuilder(get_packed_teams(repo_root / 'packed_teams'))
    pokemon_bot = TrainedRLPlayer(account_configuration=AccountConfiguration("Challenge Bot", None), model=model_path,
                                  battle_format='gen4anythinggoes', team=teams, use_argmax=True)

    pokemon_bot.reset_battles()
    loop.run_until_complete(challenge_human(pokemon_bot, opponent_username, NUM_CHALLENGES))
    if save_log:
        logfile = repo_root / 'data' / 'bot_challenge_log.csv'
        existed = logfile.exists()
        cols = ['checkpoint_path', 'won', 'bot_team', 'player_team']
        with open(logfile, 'a', newline='') as outp:
            writer = csv.DictWriter(outp, cols)
            if not existed:
                writer.writeheader()
            for battle_id, battle in pokemon_bot.battles.items():
                row = {'checkpoint_path': model_path,
                       'won': battle.won,
                       'bot_team': str(battle.team),
                       'player_team': str(battle.opponent_team)}
                writer.writerow(row)
