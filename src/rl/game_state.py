import numpy as np
from gym.spaces import Box
from poke_env.environment import AbstractBattle

from src.utils.pokemon import pokemon_to_index, GEN_DATA, STAT_IDX, SideCondition, Field, Weather


class Component:
    """Helper to make it easier to construct game low and high values"""
    length: int
    low: np.ndarray
    high: np.ndarray


class MovePower(Component):
    """Move base power (divided by 100, so it isn't too large compared to others)"""
    length = 4
    low = np.full(length, -1, dtype=np.float32)  # min of -1 for none
    high = np.full(length, 4, dtype=np.float32)  # max of 4


class MoveMultiplier(Component):
    """Move Damage Multiplier"""
    length = 4
    low = np.zeros(length, dtype=np.float32)
    high = np.full(length, 6, dtype=np.float32)  # max of 6 (4x effective, 1.5x same type bonus)


class NumFainted(Component):
    """2 numbers, the number of fainted Pokémon on a team (normalized from 0-1)"""

    length = 2
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)


class Statuses(Component):
    """Status effects on a both teams (ally, then opponent), one for an active status, zero otherwise for no active
     status"""
    length = 12
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)


class TeamHP(Component):
    """HP values for a both teams (ally, then opponent), normalized between 0 and 1"""
    length = 12
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)


class StatBoosts(Component):
    """Pokemon stat changes, -6 low, 6 high, (ally first, then opponent)"""
    length = 14
    low = np.full(length, -6, dtype=np.float32)
    high = np.full(length, 6, dtype=np.float32)


class OneSideEffects(Component):
    """Player-dependent field effects (stealth rock, light screen, etc.) (ally first, then opponents)"""
    length = 14
    low = np.zeros(length, dtype=np.float32)
    high = np.full(length, 3, dtype=np.float32)  # 3 stacks of spikes/perish song


class FullFieldEffects(Component):
    """Effects that are in the entire field (weather, trick room, etc.)"""
    length = 8
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)


class PokemonIDX(Component):
    """Pokemon OHE vector"""
    length = 67
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)


class GameState:
    def __init__(self):
        self.components: list[Component] = [
            # active pokemon description
            MovePower(),
            MoveMultiplier(),
            # num fainted per team
            NumFainted(),
            # each Pokémon's status effects
            Statuses(),
            # HP for each pokemon in both teams
            TeamHP(),
            # stat boosts for active pokemon (ally, opponent)
            StatBoosts(),
            # side effects for both teams
            OneSideEffects(),
            # current field effects
            FullFieldEffects(),
            # ally pokemon (active, then switch order)
            PokemonIDX(),
            PokemonIDX(),
            PokemonIDX(),
            PokemonIDX(),
            PokemonIDX(),
            PokemonIDX(),
            # opponent active
            PokemonIDX()
        ]

        # attributes for describing battle embeddings
        self.length = sum(c.length for c in self.components)
        self.low = np.concatenate([
            c.low for c in self.components
        ], dtype=np.float32)
        self.high = np.concatenate([
            c.high for c in self.components
        ], dtype=np.float32)
        self.description = Box(self.low, self.high, dtype=np.float32)

    @staticmethod
    def embed_state(battle: AbstractBattle) -> np.ndarray:
        """Converts battle object to a numpy array representing the battle state"""
        # we could have just made "encode()" methods on each component to make it fully modular, but that results in a
        # LOT of repeated calculations (like iterating over the team for the statuses, hp values, and fainted values
        # separately). This does mean that if the state ordering is changed, we have to change it in the final vector
        # creation as well.

        num_pokemon = PokemonIDX().length

        # define different arrays for each component, we will combine them all at the end
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        status_vals = np.zeros(6)
        opponent_status_vals = np.zeros(6)
        hp_vals = np.zeros(6)
        opponent_hp_vals = np.zeros(6)
        ally_boosts = np.zeros(7)
        opponent_boosts = np.zeros(7)
        ally_active_arr = np.zeros(num_pokemon)
        opponent_active_arr = np.zeros(num_pokemon)
        fainted_mon_team = 0
        fainted_mon_opponent = 0
        # if an ally is fainted, we set its indices to 0
        ally_indices = [ally_active_arr, *(np.zeros(num_pokemon) for _ in range(5))]
        # single-sided effects (first 7 ally, next 7 opponent)
        side_conditions = np.zeros(14)
        field_conditions = np.zeros(4)
        weather_effects = np.zeros(4)

        # get move information
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
                if move.type in battle.active_pokemon.types:
                    moves_base_power[i] *= 1.5


        # encode opponent active pokemon
        opponent_idx = pokemon_to_index(battle.opponent_active_pokemon)
        opponent_active_arr[opponent_idx] = 1

        # encode ally pokemon
        ally_active_arr[pokemon_to_index(battle.active_pokemon)] = 1
        for i, switch in enumerate(battle.available_switches):
            ally_indices[i][pokemon_to_index(switch)] = 1

        # encode stat boosts
        for boost, val in battle.active_pokemon.boosts.items():
            ally_boosts[STAT_IDX.__getitem__(boost.upper()).value] = val
        for boost, val in battle.opponent_active_pokemon.boosts.items():
            opponent_boosts[STAT_IDX.__getitem__(boost.upper()).value] = val

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

        # encode single-sided conditions for both teams
        # poke-env marks non-stackable conditions with the turn they were played, so we have to check if it is
        # stackable or not to return the correct state
        for env_condition, val in battle.side_conditions.items():
            condition = SideCondition.__getitem__(env_condition.name.upper())
            idx = condition.value
            real_val = val if condition.is_stackable() else 1
            side_conditions[idx] = real_val
        for env_condition, val in battle.opponent_side_conditions.items():
            condition = SideCondition.__getitem__(env_condition.name.upper())
            idx = condition.value
            real_val = val if condition.is_stackable() else 1
            side_conditions[idx + 7] = real_val

        # different field conditions
        for field in battle.fields.keys():
            idx = Field.__getitem__(field.name.upper()).value
            field_conditions[idx] = 1

        # weather effects
        for weather in battle.weather.keys():
            idx = Weather.__getitem__(weather.name.upper()).value
            weather_effects[idx] = 1

        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                status_vals,
                opponent_status_vals,
                hp_vals,
                opponent_hp_vals,
                ally_boosts,
                opponent_boosts,
                side_conditions,
                field_conditions,
                weather_effects,
                *ally_indices,
                opponent_active_arr
            ], dtype=np.float32
        )

        return final_vector
