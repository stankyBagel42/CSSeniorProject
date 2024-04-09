from collections import deque
from pathlib import Path

import numpy as np
import torch
from gym.core import ObsType
from gym.spaces import Space
from poke_env.environment import AbstractBattle, Battle, Status
from poke_env.player import Gen4EnvSinglePlayer, Player, ForfeitBattleOrder, BattleOrder
from poke_env.teambuilder import Teambuilder

from src.rl.agent import PokemonAgent, AgentConfig
from src.rl.game_state import GameState, get_game_state
from src.rl.network import PokeNet, DuelingPokeNet
from src.utils.pokemon import GEN_DATA


class MultiTeambuilder(Teambuilder):
    """Allows agents to select from multiple given teams"""

    def __init__(self, packed_teams: list[str]):
        # self.teams = [self.join_team(self.parse_showdown_team(team)) for team in packed_teams]
        self.teams = packed_teams

    def yield_team(self):
        return np.random.choice(self.teams)


class SimpleRLPlayer(Gen4EnvSinglePlayer):
    def __init__(self, model=None, agent_config: AgentConfig = None, game_state: GameState = None, *args, **kwargs):
        assert (model is not None) or (agent_config is not None), "Either a model or an agent config must be provided."
        self.done_training = False
        self.num_battles = 0

        # set the game state to avoid hard-coding embedding descriptions
        if game_state is None:
            game_state = GameState()
        self.game_state: GameState = game_state

        if model is None:
            model = PokemonAgent(agent_config)
        self.model: PokemonAgent = model

        # initialize base object
        super().__init__(*args, **kwargs)

    def calc_reward(self, last_battle:Battle, current_battle:Battle) -> float:
        status_val = 0.15
        reward = self.reward_computing_helper(current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0,
                                              status_value=status_val)

        # we don't want enemy Pokémon to take advantage of statuses, but ours can
        mult = [1, -1]
        active_pokemon = [current_battle.active_pokemon, current_battle.opponent_active_pokemon]
        for active_mon, mult in zip(active_pokemon, mult):
            if active_mon.ability == 'guts' and active_mon.status == Status.BRN:
                reward += status_val * 2 * mult
            elif active_mon.ability == 'poisonheal' and active_mon.status in [Status.TOX, Status.PSN]:
                reward += status_val * 2 * mult

            # stat boosts
            for boost, val in active_mon.boosts.items():
                reward += val * mult * status_val
        return reward

    def describe_embedding(self) -> Space[ObsType]:
        return self.game_state.description

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # pass to game state object
        return self.game_state.embed_state(battle)


class TrainedRLPlayer(Player):
    def __init__(self, model: PokeNet | str | Path, game_state: GameState = None, use_argmax: bool = True, *args,
                 **kwargs):
        """model is the pytorch model used to make actions given a battle state (can be a model or a PathLike object
        pointing to the saved weights. Args and Kwargs are sent to Player init"""
        super().__init__(*args, **kwargs)
        self.MESSAGES_TO_IGNORE.add('medal-msg')
        # get game state info from the model if it exists
        if isinstance(model, str | Path) and game_state is None:
            if torch.cuda.is_available():
                model_info = torch.load(model, map_location=torch.device('cuda:0'))
            else:
                model_info = torch.load(model, map_location=torch.device('cpu'))
            game_state = get_game_state(model, model_info['cfg'])
        # default game state
        if game_state is None:
            game_state = GameState()
        self.game_state = game_state
        self.use_argmax = use_argmax
        # load the model
        if isinstance(model, str | Path):
            if model_info['cfg']['dueling_dqn']:
                model = DuelingPokeNet(num_inputs=self.game_state.length, num_outputs=9,
                                layers_per_side=model_info['cfg']['num_layers_per_side'],
                                base_nodes=model_info['cfg']['base_nodes_layer']).float()
            else:
                model = PokeNet(num_inputs=self.game_state.length, num_outputs=9,
                                layers_per_side=model_info['cfg']['num_layers_per_side'],
                                base_nodes=model_info['cfg']['base_nodes_layer']).float()
            model.load_state_dict(model_info['online_model'])

        self.model = model

        #
        self.last_switch = float('-inf')
    def choose_move(self, battle):
        state = torch.tensor(self.game_state.embed_state(battle))
        if next(self.model.parameters()).is_cuda:
            state = state.cuda()
        with torch.no_grad():
            predictions = self.model(state)

        if self.use_argmax:
            # choose moves from highest likelihood to lowest, only doing random if nothing else works
            cur_action = len(predictions) - 1
            actions = np.argsort(predictions)
            while cur_action > 0:
                action = actions[cur_action]
                # from action_to_move
                if action == -1:
                    return ForfeitBattleOrder()
                elif (
                        action < 4
                        and action < len(battle.available_moves)
                        and not battle.force_switch
                ):
                    return Player.create_order(battle.available_moves[action])
                elif 0 <= action - 4 < len(battle.available_switches) and battle.turn - self.last_switch > 1:
                    self.last_switch = battle.turn
                    return Player.create_order(battle.available_switches[action - 4])
                cur_action -= 1
            return self.choose_random_move(battle)
        else:
            action = torch.distributions.Categorical(predictions).sample().item()
            if action == -1:
                return ForfeitBattleOrder()
            elif (
                    action < 4
                    and action < len(battle.available_moves)
                    and not battle.force_switch
            ):
                return self.create_order(battle.available_moves[action])
            elif 0 <= action - 4 < len(battle.available_switches) and battle.turn - self.last_switch > 1:
                self.last_switch = battle.turn
                return self.create_order(battle.available_switches[action - 4])
            else:
                return self.choose_random_move(battle)


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
                # STAB
                if move.type in battle.active_pokemon.types:
                    type_multiplier *= 1.5
                if move.base_power * type_multiplier > max_dmg:
                    max_dmg = move.base_power * type_multiplier
                    max_move = move

            return self.create_order(max_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class PlayerMemoryWrapper(Gen4EnvSinglePlayer):
    def __init__(self, player: Player, mem_size: int = 100_000, *args, **kwargs):
        """Wraps the given player object so we save all state/action pairs"""
        self._player = player
        self.mem_size = mem_size
        self.memory = deque(maxlen=self.mem_size)
        self.game_state = kwargs.pop('game_state', GameState())
        self.last_action = None
        self.last_state = None
        super().__init__(*args, **kwargs)

    def calc_reward(self, last_battle, current_battle) -> float:
        status_val = 0.15
        reward = self.reward_computing_helper(current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0,
                                              status_value=status_val)

        # we don't want enemy Pokémon to take advantage of statuses, but ours can
        mult = [1, -1]
        active_pokemon = [current_battle.active_pokemon, current_battle.opponent_active_pokemon]
        for active_mon, mult in zip(active_pokemon, mult):
            if active_mon.ability == 'guts' and active_mon.status == Status.BRN:
                reward += status_val * 2 * mult
            elif active_mon.ability == 'poisonheal' and active_mon.status in [Status.TOX, Status.PSN]:
                reward += status_val * 2 * mult

            # stat boosts
            for boost, val in active_mon.boosts.items():
                reward += val * mult * status_val
        return reward

    def describe_embedding(self) -> Space[ObsType]:
        return self.game_state.description

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # pass to game state object
        return self.game_state.embed_state(battle)


    def move_to_action(self, move: BattleOrder, battle: AbstractBattle) -> int:
        """Translates a battle order (move) to an action integer for the memory"""
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )

        return available_orders.index(move)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        move = self._player.choose_move(battle)
        return move

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        """
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    def save_memory(self, output_path:Path):
        """Saves the memories stored in this object to the given path"""
        torch.save(self.memory, output_path)