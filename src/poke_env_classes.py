from pathlib import Path

import numpy as np
import torch
from gym.spaces import Box, Space
from poke_env.data import GenData
from poke_env.player import Gen4EnvSinglePlayer, Player, ForfeitBattleOrder
from poke_env.teambuilder import Teambuilder

from src.rl.agent import PokemonAgent
from src.rl.network import PokeNet

GEN_DATA = GenData(4)


class MultiTeambuilder(Teambuilder):
    """Allows agents to select from multiple given teams"""
    def __init__(self, packed_teams:list[str]):
        # self.teams = [self.join_team(self.parse_showdown_team(team)) for team in packed_teams]
        self.teams =packed_teams

    def yield_team(self):
        return np.random.choice(self.teams)




class SimpleRLPlayer(Gen4EnvSinglePlayer):
    def __init__(self, model=None, agent_kwargs: dict = None, *args, **kwargs):
        super(SimpleRLPlayer, self).__init__(*args, **kwargs)
        self.done_training = False
        self.num_battles = 0
        if agent_kwargs is not None and 'action_dim' not in agent_kwargs.keys():
            agent_kwargs['action_dim'] = self.action_space_size()
        if model is None:
            model = PokemonAgent(**agent_kwargs)
        self.model = model

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0, status_value=1
        )

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_DATA.type_chart
                )

        status_vals = np.zeros(6)
        opponent_status_vals = np.zeros(6)
        hp_vals = np.zeros(6)
        opponent_hp_vals = np.zeros(6)

        fainted_mon_team = 0
        fainted_mon_opponent = 0
        for i, mon in enumerate(battle.team.values()):
            if mon.fainted:
                fainted_mon_team += 1
                continue
            status_vals[i] = 1 if mon.status else 0
            hp_vals[i] = mon.current_hp_fraction

        for i, mon in enumerate(battle.opponent_team.values()):
            if mon.fainted:
                fainted_mon_opponent += 1
                continue
            opponent_status_vals[i] = 1 if mon.status else 0
            opponent_hp_vals[i] = mon.current_hp_fraction

        fainted_mon_team /= 6
        fainted_mon_opponent /= 6

        # Final vector with 34 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                status_vals,
                opponent_status_vals,
                hp_vals,
                opponent_hp_vals
            ]
        )
        return np.float32(final_vector)


class TrainedRLPlayer(Player):
    def __init__(self, model: PokeNet | str | Path, *args, **kwargs):
        """model is the pytorch model used to make actions given a battle state (can be a model or a PathLike object
        pointing to the saved weights. Args and Kwargs are sent to Player init"""
        super().__init__(*args, **kwargs)
        if not isinstance(model, PokeNet):
            model_weights = torch.load(model)['online_model']
            model = PokeNet(num_inputs=34, num_outputs=9, layers_per_side=3)
            model.load_state_dict(model_weights)
        self.model = model

    def choose_move(self, battle):
        state = torch.tensor(SimpleRLPlayer.embed_battle(None,battle))
        with torch.no_grad():
            predictions = self.model(state)[0]
        action = np.argmax(predictions)

        # from action_to_move
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return Player.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return Player.create_order(battle.available_switches[action - 4])
        print(f"INVALID MOVE {action}, CHOOSING RANDOM")
        return self.choose_random_move(battle)
