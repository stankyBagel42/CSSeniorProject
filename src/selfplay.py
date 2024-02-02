import asyncio
import copy
import random
import time
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm
from poke_env import AccountConfiguration

from src.poke_env_classes import SimpleRLPlayer, MultiTeambuilder
from src.rl.agent import PokemonAgent

from src.utils.general import repo_root, read_yaml, get_packed_teams, write_yaml, latest_ckpt_file, seed_all
from src.utils.pokemon import test_vs_random

async def battle_handler(player1:SimpleRLPlayer, player2:SimpleRLPlayer, num_challenges, teams:list[str]=None):
    player2_team = None
    if teams is not None:
        player1.agent.next_team = random.choice(teams)
        player2_team = random.choice(teams)
    await asyncio.gather(
        player1.agent.send_challenges(player2.username, num_challenges),
        player2.agent.accept_challenges(player1.username, num_challenges, packed_team=player2_team),
    )


def learn_loop(player: SimpleRLPlayer, opponent: SimpleRLPlayer, num_steps: int, position: int = 0):
    """Learning loop for the agents"""
    train_start = time.time()
    # wait until the bot starts a battle
    while len(player.battles) == 0:
        time.sleep(0.001)

    current_battle = player.battles[list(player.battles.keys())[0]]
    state = player.embed_battle(current_battle)
    logs = []

    prev_q = -1
    prev_loss = -1

    # for each step in training
    pbar = tqdm(total=num_steps, desc=f"{player.username} Steps", position=position)
    for i in range(num_steps):
        # if the current battle is done (can happen before any moves because the opponent could end it)
        if current_battle.finished:
            state = player.reset()
            current_battle = player.battles[sorted(list(player.battles.keys()))[-1]]
        if isinstance(state,tuple):
            state = state[0]
        # run the state through the model
        action = player.model.act(state)

        # send action to environment and get results
        try:
            next_state, reward, done, truncated, _ = player.step(action)

        # if it fails here we just continue on, as the server is still running (usually a message parsing error)
        except Exception as e:
            print(f"EXCEPTION RAISED {e}")
            next_state = player1.embed_battle(current_battle)
            reward = player1.calc_reward(copy.deepcopy(current_battle), player1.current_battle)
            done = current_battle.finished
        # remember this action state transition
        player.model.cache(state, next_state, action, reward, done)

        # learn from past observations
        q, loss = player.model.learn()
        prev_q = q if q else prev_q
        prev_loss = loss if loss else prev_loss
        win_rate = player.win_rate if len(player.battles) > 1 else -1
        # log dictionary
        log = {
            'q': f"{prev_q:+.2f}",
            'loss': f"{prev_loss:+.2f}",
            'reward': f"{reward:+06.2f}",
            'win_rate': f"{win_rate:0.2f}",
            'exploration_rate': player.model.exploration_rate
        }
        pbar.set_postfix(log)
        pbar.update()
        logs.append(log)

        state = next_state
        if done or current_battle.finished:
            state = player.reset()
            current_battle = player.battles[sorted(list(player.battles.keys()))[-1]]
    train_end = time.time()
    # log total time
    print(f"{player.username} finished {len(player.battles)} battles in {train_end-train_start}s with "
          f"a {player.win_rate:0.2%} win rate!")
    # we are done with training
    player.done_training = True
    # Play out the remaining battles so both fit() functions complete
    # We use 99 to give the agent an invalid option so it's forced
    # to take a random legal action
    while not opponent.done_training:
        _, _, done, _, _ = player.step(99)
        if done and not opponent.done_training:
            _ = player.reset()
            done = False

    # Forfeit any ongoing battles
    while player.current_battle and not player.current_battle.finished:
        _ = player.step(-1)




if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = read_yaml(repo_root / 'train_config.yaml')
    p1 = AccountConfiguration('RL Bot 1', None)
    p2 = AccountConfiguration('RL Bot 2', None)

    BATTLE_FORMAT = "gen4anythinggoes"

    # STATE:
    # Move power for active moves (4)
    # Move multiplier for active moves (4)
    # # Pokemon fainted allies (1)
    # # Pokemon fainted opponent (1)
    # # Status on allies (6)
    # # Status on enemies (6)
    # Allies HP Fraction (6)
    # Opponent HP Fraction (6)
    # Ally Active Stat Changes (7)
    # Opponent Active Stat Changes (7)
    # One hot encode ally pokemon, active first (66 * 6, 396)
    # One hot encode opponent pokemon active pokemon (66)
    STATE_DIM = 510


    checkpoint_dir = Path(cfg['checkpoint_dir']) / cfg['run_name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # only get teams if we arent in a random format
    if 'random' not in BATTLE_FORMAT.lower():
        teams = get_packed_teams(repo_root / 'packed_teams')
        teambuilder = MultiTeambuilder(teams)
    else:
        teams = None

        teambuilder = None

    AGENT_KWARGS = {
        'state_dim': STATE_DIM,
        'save_dir': checkpoint_dir / 'player_1',
        'action_dim': 9,
    }
    num_steps = cfg.pop('num_steps')

    # non-training parameter config values
    cfg.pop('checkpoint_dir')

    cfg.pop('run_name')
    AGENT_KWARGS.update(cfg)

    player2 = SimpleRLPlayer(
        battle_format=BATTLE_FORMAT,
        opponent="placeholder",
        start_challenging=False,
        account_configuration=p2,
        agent_kwargs=AGENT_KWARGS,
        team=teambuilder
    )
    AGENT_KWARGS['save_dir'] = checkpoint_dir / 'player_2'

    player1 = SimpleRLPlayer(
        battle_format=BATTLE_FORMAT,
        opponent=player2,
        start_challenging=False,
        account_configuration=p1,
        agent_kwargs=AGENT_KWARGS,
        team=teambuilder
    )

    # Setup arguments to pass to the training function
    p1_env_kwargs = {"num_steps": num_steps}
    p2_env_kwargs = {"num_steps": num_steps}

    player1.set_opponent(player2)
    player2.set_opponent(player1)
    # Self-Play bits
    player1.done_training = False
    player2.done_training = False

    loop = asyncio.get_event_loop()
    start = time.time()

    # Make Two Threads; one per player and train
    t1 = Thread(target=learn_loop,args=(player1, player2, num_steps, 0),daemon=True)
    # t1 = Thread(target=lambda: learn_loop(player1, player2, num_steps, position=0),daemon=True)
    t1.start()

    t2 = Thread(target=learn_loop, args=(player2, player1, num_steps, 1),daemon=True,)
    t2.start()

    # On the network side, keep sending & accepting battles
    while not player1.done_training or not player2.done_training:
        loop.run_until_complete(battle_handler(player1, player2, 1))

    # Wait for thread completion
    t1.join()
    t2.join()

    player1.close(purge=False)
    player2.close(purge=False)
    end = time.time()
    print(f"Finished with {num_steps} steps ({player1.n_finished_battles} battles) in {end - start:0.2f}s "
          f"({player1.n_finished_battles / (end - start) :0.2f} battles/s)")
    print(f"Player 1 WR: {player1.win_rate}")
    print(f"Player 2 WR: {player2.win_rate}")
