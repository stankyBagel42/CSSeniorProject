import asyncio
import copy
import random
import time
from threading import Thread

from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer
from tqdm import tqdm

from src.poke_env_classes import PlayerMemoryWrapper, MultiTeambuilder
from src.utils.general import get_packed_teams, repo_root
from src.utils.pokemon import create_player


async def battle_handler(player1: PlayerMemoryWrapper, player2: PlayerMemoryWrapper, num_challenges,
                         teams: list[str] = None):
    player2_team = None
    if teams is not None:
        player1.agent.next_team = random.choice(teams)
        player2_team = random.choice(teams)
    await asyncio.gather(
        player1.agent.send_challenges(player2.username, num_challenges),
        player2.agent.accept_challenges(player1.username, num_challenges, packed_team=player2_team),
    )


def learn_loop(player: PlayerMemoryWrapper, opponent: PlayerMemoryWrapper, num_steps: int, position: int = 0):
    """Learning loop for the agents"""
    train_start = time.time()
    # wait until the bot starts a battle
    while len(player.battles) == 0:
        time.sleep(0.001)

    current_battle = player.battles[list(player.battles.keys())[0]]
    state = player.embed_battle(current_battle)

    # for each step in training
    pbar = tqdm(total=num_steps, desc=f"{player.username} Steps", position=position)
    for i in range(num_steps):
        # if the current battle is done (can happen before any moves because the opponent could end it)
        if current_battle.finished:
            player.reset_env()
            current_battle._finish_battle()
            state = player.reset()
            current_battle = player.battles[sorted(list(player.battles.keys()))[-1]]
        if isinstance(state, tuple):
            state = state[0]
        # run the state through the model
        move = player.choose_move(current_battle)

        action = player.move_to_action(move, current_battle)

        # send action to environment and get results
        try:
            next_state, reward, done, truncated, _ = player.step(action)

        # if it fails here we just continue on, as the server is still running (usually a message parsing error)
        except Exception as e:
            print(f"EXCEPTION RAISED {e}")
            next_state = player.embed_battle(current_battle)
            reward = player.calc_reward(copy.deepcopy(current_battle), player.current_battle)
            done = current_battle.finished
        # remember this action state transition
        player.cache(state, next_state, action, reward, done)

        pbar.update()

        # prepare next state and loop
        state = next_state
        if done or current_battle.finished:
            state = player.reset()
            current_battle = player.battles[sorted(list(player.battles.keys()))[-1]]
    train_end = time.time()
    # log total time
    print(f"{player.username} finished {len(player.battles)} battles in {train_end - train_start}s with "
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


if __name__ == '__main__':
    BATTLE_FORMAT = 'gen4anythinggoes'
    REPLAY_NAME = "240212_AbilityState_50k"
    MEM_SIZE = 50_000
    NUM_STEPS = 50_000

    out_dir = repo_root / 'data' / 'replay_buffers'
    (out_dir/ REPLAY_NAME).mkdir(parents=True, exist_ok=True)

    teams = get_packed_teams(repo_root / 'packed_teams')

    p1 = SimpleHeuristicsPlayer()
    p2 = SimpleHeuristicsPlayer()
    player1 = create_player(
        PlayerMemoryWrapper,
        player=p1,
        opponent="placeholder",
        start_challenging=False,
        mem_size=MEM_SIZE,
        battle_format=BATTLE_FORMAT,
        team=MultiTeambuilder(teams)
    )
    player2 = create_player(
        PlayerMemoryWrapper,
        player=p2,
        opponent="placeholder",
        start_challenging=False,
        mem_size=MEM_SIZE,
        battle_format=BATTLE_FORMAT,
        team=MultiTeambuilder(teams)
    )


    # set the 2 players to face each other
    player1.set_opponent(player2)
    player2.set_opponent(player1)
    # these flags are used to kill training
    player1.done_training = False
    player2.done_training = False

    # get  async loop and start the training timer
    loop = asyncio.get_event_loop()
    start = time.time()

    # 2 threads, one for each player
    t1 = Thread(target=learn_loop, args=(player1, player2, NUM_STEPS, 0), daemon=True)
    t1.start()

    t2 = Thread(target=learn_loop, args=(player2, player1, NUM_STEPS, 1), daemon=True)
    t2.start()

    # On the network side, keep sending & accepting battles
    while not player1.done_training or not player2.done_training:
        loop.run_until_complete(battle_handler(player1, player2, 1))

    # Wait for thread completion (training to finish)
    t1.join()
    t2.join()
    player1.close(purge=False)
    player2.close(purge=False)
    end = time.time()

    print(f"Finished in {end - start:0.2f}s")
    player1.save_memory(out_dir / REPLAY_NAME / 'player1_memory.pt')
    player2.save_memory(out_dir / REPLAY_NAME / 'player2_memory.pt')
