import asyncio
import copy
import random
import time
from pathlib import Path
from threading import Thread

import wandb
from matplotlib import pyplot as plt
from poke_env.player import RandomPlayer
from tqdm import tqdm
from poke_env import AccountConfiguration

from benchmark import benchmark_player
from src.poke_env_classes import SimpleRLPlayer, MultiTeambuilder, TrainedRLPlayer
from src.rl.agent import PokemonAgent
from src.rl.game_state import GameState
from src.utils.general import repo_root, read_yaml, get_packed_teams, write_yaml, latest_ckpt_file, seed_all
from src.utils.pokemon import test_vs_random


async def battle_handler(player1: SimpleRLPlayer, player2: SimpleRLPlayer, num_challenges, teams: list[str] = None):
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

    prev_q = -1
    prev_loss = -1

    # for each step in training
    pbar = tqdm(total=num_steps, desc=f"{player.username} Steps", position=position)
    for i in range(num_steps):
        # if the current battle is done (can happen before any moves because the opponent could end it)
        if current_battle.finished:
            state = player.reset()
            current_battle = player.battles[sorted(list(player.battles.keys()))[-1]]
        if isinstance(state, tuple):
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
        # log dictionary
        log = {
            'step': player.model.curr_step,
            'q': f"{prev_q:+.2f}",
            'loss': f"{prev_loss:+.2f}",
            'reward': f"{reward:+06.2f}",
            'exploration_rate': player.model.exploration_rate
        }
        if loss == -1:
            log.pop('loss')
        if q == -1:
            log.pop('q')
        pbar.set_postfix(log)
        pbar.update()
        step = log.pop('step')
        wandb_log = {
            f"{player.username}/{k}": float(v) for k, v in log.items()
        }
        wandb_log['step'] = step
        # log player values under separate folders
        wandb.log(wandb_log)

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


def load_latest(checkpoint_directory: str | Path, **kwargs) -> PokemonAgent:
    latest_checkpoint = latest_ckpt_file(checkpoint_directory)
    agent = PokemonAgent(checkpoint=latest_checkpoint, **kwargs)
    return agent


def validate_player(player: SimpleRLPlayer, baseline_bot, trained_bot: TrainedRLPlayer, target_bot: TrainedRLPlayer):
    """Validate the given player's current model against the given baseline"""
    val_start = time.time()
    trained_bot.model = copy.deepcopy(player.model.online_net).cpu()
    target_bot.model = copy.deepcopy(player.model.target_net).cpu()

    wr = test_vs_random(trained_bot, num_validate_battles, random_player=baseline_bot)
    target_wr = test_vs_random(target_bot, num_validate_battles, random_player=baseline_bot)

    val_end = time.time()
    wandb.log({f"{player.username}/val/wr": wr, f"{player.username}/val/target_wr": target_wr, "step": player.model.curr_step})
    print("\n\n" + "-" * 30)
    print(f"RL Bot at {player.model.curr_step} Steps")
    print(f"Online net vs Random WR: {wr:0.2%}")
    print(f"Target net vs Random WR: {target_wr:0.2%}")
    print(f"Validation took {val_end - val_start:0.2f}s")
    print("-" * 30 + "\n\n")


if __name__ == "__main__":
    # Set random seed
    seed_all(42)

    cfg = read_yaml(repo_root / 'train_config.yaml')

    # initialize weights and biases to track training
    wandb.init(project='pokemon_reinforcement_learning',
               config=cfg)
    p1 = AccountConfiguration('RL Bot 1', None)
    p2 = AccountConfiguration('RL Bot 2', None)

    BATTLE_FORMAT = "gen4anythinggoes"

    # description of game state
    game_state = GameState()

    STATE_DIM = game_state.length

    checkpoint_dir = Path(cfg['checkpoint_dir']) / cfg['run_name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(cfg, checkpoint_dir / 'train_config.yaml')
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
    validate_freq = cfg.pop('validate_freq')
    num_validate_battles = cfg.pop('num_validate_battles')

    # non-training parameter config values
    cfg.pop('checkpoint_dir')
    cfg.pop('run_name')
    resume = cfg.pop('resume')
    resume_from = cfg.pop('resume_from')
    AGENT_KWARGS.update(cfg)

    if resume:
        save_dir = resume_from / 'player_1'
        player1 = SimpleRLPlayer(
            battle_format=BATTLE_FORMAT,
            opponent="placeholder",
            start_challenging=False,
            account_configuration=p1,
            model=load_latest(save_dir, **AGENT_KWARGS),
            team=teambuilder
        )

        save_dir = resume_from / 'player_2'

        AGENT_KWARGS['save_dir'] = checkpoint_dir / 'player_2'
        player2 = SimpleRLPlayer(
            battle_format=BATTLE_FORMAT,
            opponent="placeholder",
            start_challenging=False,
            account_configuration=p2,
            model=load_latest(save_dir, **AGENT_KWARGS),
            team=teambuilder
        )
    else:
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

    # create validation players once
    trained_player = TrainedRLPlayer(None, battle_format=BATTLE_FORMAT, team=MultiTeambuilder(teams))
    target_player = TrainedRLPlayer(None, battle_format=BATTLE_FORMAT, team=MultiTeambuilder(teams))

    validation_baseline = RandomPlayer(battle_format=BATTLE_FORMAT, team=MultiTeambuilder(teams))

    player1.set_opponent(player2)
    player2.set_opponent(player1)
    # Self-Play bits
    player1.done_training = False
    player2.done_training = False

    loop = asyncio.get_event_loop()
    start = time.time()

    # Make Two Threads; one per player and train
    t1 = Thread(target=learn_loop, args=(player1, player2, num_steps, 0), daemon=True)
    # t1 = Thread(target=lambda: learn_loop(player1, player2, num_steps, position=0),daemon=True)
    t1.start()

    t2 = Thread(target=learn_loop, args=(player2, player1, num_steps, 1), daemon=True, )
    t2.start()

    val_steps = []
    val_wrs = []
    val_target_wrs = []
    num_battles = 0
    # On the network side, keep sending & accepting battles
    while not player1.done_training or not player2.done_training:
        loop.run_until_complete(battle_handler(player1, player2, 1))
        num_battles += 1
        # validate against random (once the model is out of warmup)
        if num_battles % validate_freq == 0 and player1.model.curr_step > cfg['warmup_steps']:
            for p in [player1, player2]:
                validate_player(p, validation_baseline, trained_player, target_player)

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

    plt.plot(val_steps, val_wrs, label="Online Network WR")
    plt.plot(val_steps, val_target_wrs, label="Target Network WR")
    plt.title("Validation Winrate vs Random")
    plt.savefig(checkpoint_dir / 'validate_winrate_plot.png')

    # copy player 1's model to benchmark (player 2 should have similar performance)
    trained_player.model = copy.deepcopy(player1.model.online_net).cpu()

    print(f"Benchmarking...")
    # run the player benchmark vs the standard deterministic players
    benchmark_results = benchmark_player(trained_player, n_challenges=500, teambuilder=teambuilder)

    # get the average win rate in the benchmark, and log results
    total_wr = 0
    count = 0
    print(benchmark_results.keys())
    for opponent, win_rate in benchmark_results[trained_player.username].items():
        if win_rate is not None:
            total_wr += 1
            count += 1
            print(f"Bot WR vs {opponent.lower()}: {win_rate}")
            wandb.log({f'benchmark/{opponent.lower()}': win_rate})

    wandb.log({"benchmark/average_winrate": total_wr / count})

    wandb.finish()
