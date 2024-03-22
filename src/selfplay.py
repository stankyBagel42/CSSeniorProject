import asyncio
import copy
import random
import time
from dataclasses import asdict
from pathlib import Path
from threading import Thread

from poke_env.environment import AbstractBattle
from poke_env.player import SimpleHeuristicsPlayer

import wandb
from tqdm import tqdm

from src.benchmark import benchmark_player
from src.poke_env_classes import SimpleRLPlayer, MultiTeambuilder, TrainedRLPlayer
from src.rl.agent import PokemonAgent, AgentConfig
from src.rl.game_state import GameState
from src.utils.general import repo_root, read_yaml, get_packed_teams, write_yaml, latest_ckpt_file, seed_all, RunningAvg
from src.utils.pokemon import test_vs_bot, create_player


async def battle_handler(player1: SimpleRLPlayer, player2: SimpleRLPlayer, num_challenges, teams: list[str] = None):
    player2_team = None
    if teams is not None:
        player1.agent.next_team = random.choice(teams)
        player2_team = random.choice(teams)
    await asyncio.gather(
        player1.agent.send_challenges(player2.username, num_challenges),
        player2.agent.accept_challenges(player1.username, num_challenges, packed_team=player2_team),
    )


def is_battle_finished(battle: AbstractBattle) -> bool:
    """Poke env hangs annoyingly, this might help with that?"""
    all_fainted_team = True
    all_fainted_opponent = True
    for mon in battle.team.values():
        if not (mon.fainted or mon.current_hp_fraction == 0):
            all_fainted_team = False
            break
    for mon in battle.opponent_team.values():
        if not (mon.fainted or mon.current_hp_fraction == 0):
            all_fainted_opponent = False
            break
    return all_fainted_team or all_fainted_opponent


def learn_loop(player: SimpleRLPlayer, opponent: SimpleRLPlayer, num_steps: int, position: int = 0, is_p1: bool = True):
    """Learning loop for the agents"""
    train_start = time.time()
    # wait until the bot starts a battle
    while len(player.battles) == 0:
        time.sleep(0.001)

    current_battle = player.battles[list(player.battles.keys())[0]]
    state = player.embed_battle(current_battle)

    prev_q = -1
    prev_loss = -1
    # wandb suggests limiting to 100k per attribute logged, so we will average over iterations until we log 100k for
    # the total steps
    log_freq = max(num_steps // 100_000, 1)
    log_avg = RunningAvg(log_freq)

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
            next_state, reward, done, _, _ = player.step(action)

        # if it fails here we just continue on, as the server is still running (usually a message parsing error)
        except Exception as e:
            print(f"EXCEPTION RAISED {e}")
            next_state = player.embed_battle(current_battle)
            reward = player.calc_reward(None, player.current_battle)
            done = current_battle.finished
        # remember this action state transition
        player.model.cache(state, next_state, action, reward, done)

        # learn from past observations
        q, loss = player.model.learn()
        prev_q = q if q else prev_q
        prev_loss = loss if loss else prev_loss

        # setup logs for console and wandb
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
        user = 'RL Bot 1' if is_p1 else 'RL Bot 2'
        wandb_log = log_avg.log({
            f"{user}/{k}": float(v) for k, v in log.items()
        })
        wandb_log['step'] = step

        if step % log_freq == 0:
            # log player values under separate folders
            wandb.log(wandb_log)

        # prepare next state and loop
        state = next_state
        if done or current_battle.finished:
            state = player.reset()
            current_battle = player.battles[sorted(list(player.battles.keys()))[-1]]
    train_end = time.time()
    # log total time
    print(f"{player.username} finished {player.n_finished_battles} battles in {train_end - train_start}s with "
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


def load_latest(checkpoint_directory: str | Path, cfg: AgentConfig) -> tuple[PokemonAgent, GameState]:
    latest_checkpoint = latest_ckpt_file(checkpoint_directory)
    agent = PokemonAgent(cfg, checkpoint=latest_checkpoint)
    # read the train config for the run we are resuming to get the game state
    run_cfg = read_yaml(Path(checkpoint_directory).parent / 'train_config.yaml')
    game_state = GameState.from_component_list(run_cfg['state_components'])
    return agent, game_state


def validate_player(player: SimpleRLPlayer, baseline_player, trained_bot: TrainedRLPlayer, target_bot: TrainedRLPlayer,
                    num_challenges: int = 100, teams: list[str] = None, is_p1: bool = True):
    """Validate the given player's current model against the given baseline"""
    if teams is None:
        teams = get_packed_teams(repo_root / 'packed_teams')
    val_start = time.time()
    trained_bot.model = copy.deepcopy(player.model.online_net).cpu()
    target_bot.model = copy.deepcopy(player.model.target_net).cpu()

    wr = test_vs_bot(trained_bot, n_challenges=num_challenges, baseline_player=baseline_player, teams=teams)
    target_wr = test_vs_bot(target_bot, n_challenges=num_challenges, baseline_player=baseline_player,
                            teams=teams)

    val_end = time.time()
    user = 'RL Bot 1' if is_p1 else 'RL Bot 2'
    wandb.log({f"{user}/val/wr": wr, f"{user}/val/target_wr": target_wr,
               "step": player.model.curr_step})
    print("\n\n" + "-" * 30)
    print(f"RL Bot at {player.model.curr_step} Steps")
    print(f"Online net vs Baseline WR: {wr:0.2%}")
    print(f"Target net vs Baseline WR: {target_wr:0.2%}")
    print(f"Validation took {val_end - val_start:0.2f}s")
    print("-" * 30 + "\n\n")


def main():
    # Set random seed
    seed_all(42)

    cfg = read_yaml(repo_root / 'train_config.yaml')

    # initialize weights and biases to track training
    if cfg.pop('debug'):
        wandb.init(mode="disabled")
    else:
        wandb.init(project='pokemon_reinforcement_learning',
                   config=cfg)

    BATTLE_FORMAT = "gen4anythinggoes"

    # description of game state
    game_state = GameState()

    checkpoint_dir = Path(cfg['checkpoint_dir']) / cfg['run_name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(cfg, checkpoint_dir / 'train_config.yaml')
    # only get teams if we aren't in a random format
    if 'random' not in BATTLE_FORMAT.lower():
        teams = get_packed_teams(repo_root / 'packed_teams')
        teambuilder = MultiTeambuilder(teams)
    else:
        teams = None
        teambuilder = None

    num_steps = cfg.pop('num_steps')
    validate_freq = cfg.pop('validate_freq')
    num_validate_battles = cfg.pop('num_validate_battles')

    # non-training parameter config values
    cfg.pop('checkpoint_dir')
    cfg.pop('run_name')
    resume = cfg.pop('resume')
    resume_from = cfg.pop('resume_from')
    resume_from = Path(resume_from) if resume_from else None
    use_replays = cfg.pop('use_existing_buffer')
    pretrain_steps = cfg.pop('pre_train_steps')
    replay_path = cfg.pop('replay_buffer_path')
    state_components = cfg.pop('state_components')

    agent_config = AgentConfig(state_dim=game_state.length, action_dim=9, save_dir=checkpoint_dir / 'player_1',
                               game_state=game_state, **cfg)

    # save agent config, has some repeated keys, but it also has some extra info like the state dim
    write_yaml(asdict(agent_config), checkpoint_dir / 'agent_config.yaml')

    # change params based on if we are loading a model or not
    if resume:
        model1, game_state = load_latest(resume_from / 'player_1', agent_config)
        agent_config.save_dir = checkpoint_dir / 'player_2'
        model2, _ = load_latest(resume_from / 'player_2', agent_config)
        models = [model1, model2]
        agent_configs = [None, None]
    else:
        models = [None, None]
        agent_configs = [agent_config]
        cfg2 = copy.deepcopy(agent_config)
        cfg2.save_dir = checkpoint_dir / 'player_2'
        agent_configs.append(cfg2)
        game_state = GameState.from_component_list(state_components)

    # create 2 players
    player1 = create_player(
        SimpleRLPlayer,
        username="RL Bot 1",
        battle_format=BATTLE_FORMAT,
        opponent="placeholder",
        start_challenging=False,
        model=models[0],
        agent_config=agent_configs[0],
        team=teambuilder,
        game_state=game_state
    )
    player2 = create_player(
        SimpleRLPlayer,
        username="RL Bot 2",
        battle_format=BATTLE_FORMAT,
        opponent="placeholder",
        start_challenging=False,
        model=models[1],
        agent_config=agent_configs[1],
        team=teambuilder,
        game_state=game_state
    )

    # load memories if needed
    if use_replays:
        player1.model.set_memory(replay_path)
        # only load the memories once, we can copy them here to speed it up
        player2.model.memory = copy.deepcopy(player1.model.memory)

        if pretrain_steps > 0:
            print(f"Pretraining player1")
            player1.model.pretrain(pretrain_steps, True)
            print(f"Pretraining player2")
            player2.model.pretrain(pretrain_steps, False)

    # create validation players once
    trained_player = create_player(TrainedRLPlayer, model=None, battle_format=BATTLE_FORMAT,
                                   team=MultiTeambuilder(teams))
    target_player = create_player(TrainedRLPlayer, model=None, battle_format=BATTLE_FORMAT,
                                  team=MultiTeambuilder(teams))
    baseline_player = create_player(SimpleHeuristicsPlayer, 'HeuristicPlayer', battle_format=BATTLE_FORMAT,
                                    team=MultiTeambuilder(teams))

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
    t1 = Thread(target=learn_loop, args=(player1, player2, num_steps, 0, True), daemon=True)
    t2 = Thread(target=learn_loop, args=(player2, player1, num_steps, 1, False), daemon=True)

    t1.start()
    t2.start()

    num_battles = 0
    # On the network side, keep sending & accepting battles
    while not player1.done_training or not player2.done_training:
        loop.run_until_complete(battle_handler(player1, player2, 1))
        num_battles += 1
        # validate against random (once the model is out of warmup)
        if num_battles % validate_freq == 0 and player1.model.curr_step > cfg['warmup_steps']:
            validate_player(player1, baseline_player, trained_player, target_player,
                            num_challenges=num_validate_battles,
                            teams=teams,
                            is_p1=True)
            validate_player(player2, baseline_player, trained_player, target_player,
                            num_challenges=num_validate_battles,
                            teams=teams,
                            is_p1=False
                            )

    # Wait for thread completion (training to finish)
    t1.join()
    t2.join()
    player1.close(purge=False)
    player2.close(purge=False)
    end = time.time()

    # log basic stats, helps w/ sanity checks in case one player got messed up somewhere
    print(f"Finished with {num_steps} steps ({player1.n_finished_battles} battles) in {end - start:0.2f}s "
          f"({player1.n_finished_battles / (end - start) :0.2f} battles/s)")
    print(f"Player 1 WR: {player1.win_rate}")
    print(f"Player 2 WR: {player2.win_rate}")

    # run the player benchmark
    print(f"Benchmarking...")
    # copy player 1's model to benchmark (player 2 should have similar performance)
    trained_player.model = copy.deepcopy(player1.model.online_net).cpu()
    benchmark_results = benchmark_player(trained_player, n_challenges=500, teams=teams)

    # get the average win rate in the benchmark, and log results
    total_wr = 0
    count = 0
    for opponent, win_rate in benchmark_results.items():
        if win_rate is not None:
            total_wr += win_rate
            count += 1
            print(f"Bot WR vs {opponent.lower()}: {win_rate}")
            # log based off of the name, not the number of the bot
            wandb.log({f'benchmark/{opponent.lower()}': win_rate})

    wandb.log({"benchmark/average_winrate": total_wr / count})

    wandb.finish()

    player1.model.save(save_name="final")
    player2.model.save(save_name="final")


if __name__ == '__main__':
    main()
