from abc import abstractmethod

import numpy as np
from gym.spaces import Box
from poke_env.environment import AbstractBattle, Pokemon

from src.utils.pokemon import pokemon_to_index, GEN_DATA, STAT_IDX, SideCondition, Field, Weather, NotableAbility


class StateComponent:
    """Helper to make it easier to construct game low and high values"""
    length: int
    low: np.ndarray
    high: np.ndarray

    @abstractmethod
    def embed(self, battle: AbstractBattle) -> np.ndarray:
        pass


class MovePower(StateComponent):
    """Move base power (divided by 100, so it isn't too large compared to others)"""
    length = 4
    low = np.full(length, -1, dtype=np.float32)  # min of -1 for none
    high = np.full(length, 4, dtype=np.float32)  # max of 4

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        moves_base_power = np.zeros(self.length)
        active_pokemon = battle.active_pokemon

        # get move information
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type and move.type in active_pokemon.types:
                moves_base_power[i] *= 1.5
        return moves_base_power


class MoveMultiplier(StateComponent):
    """Move Damage Multiplier"""

    length = 4
    low = np.zeros(length, dtype=np.float32)
    high = np.full(length, 6, dtype=np.float32)  # max of 6 (4x effective, 1.5x same type bonus)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        moves_dmg_multiplier = np.zeros(self.length)

        # get move information
        for i, move in enumerate(battle.available_moves):
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_DATA.type_chart
                )
        return moves_dmg_multiplier


class NotableAbilities(StateComponent):
    """If a Pokémon can have levitate, flash fire, or any other damage altering abilities. First 5 are ally, next are
    opponent"""

    # the abilities are levitate, volt/water absorb, flash fire, magic guard
    num_abilities = len(NotableAbility)
    length = num_abilities * 2
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        arr = np.zeros(self.length, dtype=np.float32)

        for ability in battle.active_pokemon.possible_abilities:
            uppercase = ability.upper()
            if hasattr(NotableAbility, uppercase):
                arr[getattr(NotableAbility, uppercase).value] = 1
        for ability in battle.opponent_active_pokemon.possible_abilities:
            uppercase = ability.upper()
            if hasattr(NotableAbility, uppercase):
                arr[getattr(NotableAbility, uppercase).value + self.num_abilities] = 1

        return arr


class NumFainted(StateComponent):
    """2 numbers, the number of fainted Pokémon on a team (normalized from 0-1)"""

    length = 2
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        ally_fainted = 0
        opponent_fainted = 0
        for mon in battle.team.values():
            if mon.fainted:
                ally_fainted += 1

        for mon in battle.opponent_team.values():
            if mon.fainted:
                opponent_fainted += 1

        return np.array([ally_fainted / 6, opponent_fainted / 6], dtype=np.float32)


class Statuses(StateComponent):
    """Status effects on a both teams (ally, then opponent), one for an active status, zero otherwise for no active
     status"""
    length = 12
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        statuses = np.zeros(self.length, dtype=np.float32)
        for i, mon in enumerate(battle.team.values()):
            if mon.status:
                statuses[i] = 1

        for i, mon in enumerate(battle.opponent_team.values()):
            if mon.status:
                statuses[i + 6] = 1

        return statuses


class TeamHP(StateComponent):
    """HP values for a both teams (ally, then opponent), normalized between 0 and 1"""
    length = 12
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        hp_vals = np.zeros(self.length, dtype=np.float32)
        for i, mon in enumerate(battle.team.values()):
            hp_vals[i] = mon.current_hp_fraction

        for i, mon in enumerate(battle.opponent_team.values()):
            hp_vals[i + 6] = mon.current_hp_fraction

        return hp_vals


class StatBoosts(StateComponent):
    """Pokemon stat changes, -6 low, 6 high, (ally first, then opponent)"""

    num_boostable_stats = len(STAT_IDX)
    length = 2*num_boostable_stats
    low = np.full(length, -6, dtype=np.float32)
    high = np.full(length, 6, dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        ally_boosts = np.zeros(self.num_boostable_stats, dtype=np.float32)
        opponent_boosts = np.zeros(self.num_boostable_stats, dtype=np.float32)
        # encode stat boosts
        for boost, val in battle.active_pokemon.boosts.items():
            ally_boosts[STAT_IDX.__getitem__(boost.upper()).value] = val
        for boost, val in battle.opponent_active_pokemon.boosts.items():
            opponent_boosts[STAT_IDX.__getitem__(boost.upper()).value] = val
        return np.concatenate([ally_boosts, opponent_boosts])


class OneSideEffects(StateComponent):
    """Player-dependent field effects (stealth rock, light screen, etc.) (ally first, then opponents)"""

    num_side_conditions = len(SideCondition)
    length = 14
    low = np.zeros(length, dtype=np.float32)
    high = np.full(length, 3, dtype=np.float32)  # 3 stacks of spikes/perish song

    def embed(self, battle: AbstractBattle):
        side_conditions = np.zeros(self.length, dtype=np.float32)

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
            side_conditions[idx + self.num_side_conditions] = real_val

        return side_conditions


class FullFieldEffects(StateComponent):
    """Effects that are in the entire field (weather, trick room, etc.)"""

    num_field_conditions = len(Field)
    num_weather = len(Weather)
    length = 8
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        field_conditions = np.zeros(self.num_field_conditions, dtype=np.float32)
        weather_effects = np.zeros(self.num_weather, dtype=np.float32)

        # different field conditions
        for field in battle.fields.keys():
            idx = Field.__getitem__(field.name.upper()).value
            field_conditions[idx] = 1

        # weather effects
        for weather in battle.weather.keys():
            idx = Weather.__getitem__(weather.name.upper()).value
            weather_effects[idx] = 1

        return np.concatenate([field_conditions, weather_effects], dtype=np.float32)


class PokemonIDX(StateComponent):
    """Pokemon OHE vector"""
    num_pokemon = 67  # 67 unique Pokémon,
    length = num_pokemon * 7  # 6 on the current team, 1 for the opponent's active
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        opponent_active_arr = np.zeros(self.num_pokemon, dtype=np.float32)
        ally_active_arr = np.zeros(self.num_pokemon, dtype=np.float32)
        ally_indices = np.zeros(self.num_pokemon * 5, dtype=np.float32)

        # encode opponent active pokemon
        opponent_idx = pokemon_to_index(battle.opponent_active_pokemon)
        opponent_active_arr[opponent_idx] = 1

        # encode ally pokemon
        ally_active_arr[pokemon_to_index(battle.active_pokemon)] = 1
        for i, switch in enumerate(battle.available_switches):
            ally_indices[i][pokemon_to_index(switch)] = 1

        return np.concatenate([
            ally_active_arr,
            ally_indices,
            opponent_active_arr
        ], dtype=np.float32)

class EstimatedMatchups(StateComponent):
    """Estimated matchups for all ally Pokémon vs enemy Pokémon, idea and code are from the 'SimpleHeuristicPlayer'
    from poke_env"""
    length = 6
    low = np.full(length, -10, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)


    def embed(self, battle: AbstractBattle) -> np.ndarray:
        matchups = np.zeros(self.length, dtype=np.float32)
        opponent = battle.opponent_active_pokemon
        for i, pokemon in enumerate([battle.active_pokemon,*battle.available_switches]):
            score = max([opponent.damage_multiplier(t) for t in pokemon.types if t is not None])
            score -= max(
                [pokemon.damage_multiplier(t) for t in opponent.types if t is not None]
            )
            if pokemon.base_stats["spe"] > opponent.base_stats["spe"]:
                score += 0.1
            elif opponent.base_stats["spe"] > pokemon.base_stats["spe"]:
                score -= 0.1

            score += pokemon.current_hp_fraction * 0.4
            score -= opponent.current_hp_fraction * 0.4

            matchups[i] = score
        return matchups

class MoveComponents(StateComponent):
    # the different possible subcomponents in the right order (matches args order and vector embedding order)
    possible_components = [MovePower, MoveMultiplier]

    def __init__(self, base_power: bool = True, multiplier: bool = True):
        """Wrapper for all move components"""

        self.components = []

        self.base_power = base_power
        self.multiplier = multiplier

        if base_power:
            self.components.append(MovePower())

        if multiplier:
            self.components.append(MoveMultiplier())

        # attributes for describing battle embeddings
        self.length = sum(c.length for c in self.components)
        self.low = np.concatenate([
            c.low for c in self.components
        ], dtype=np.float32)
        self.high = np.concatenate([
            c.high for c in self.components
        ], dtype=np.float32)

    def embed(self, battle: AbstractBattle):
        moves_base_power = np.zeros(MovePower.length)
        moves_dmg_multiplier = np.zeros(MoveMultiplier.length)

        active_pokemon = battle.active_pokemon
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
                if move.type in active_pokemon.types:
                    moves_base_power[i] *= 1.5

        final_vector = []

        if self.base_power:
            final_vector.append(moves_base_power)
        if self.multiplier:
            final_vector.append(moves_dmg_multiplier)

        return np.concatenate(final_vector, dtype=np.float32)


class TeamComponents(StateComponent):
    # the different possible subcomponents in the right order (matches args order and vector embedding order)
    possible_components = [NumFainted, Statuses, TeamHP]

    def __init__(self, num_fainted: bool = True, statuses: bool = True, hp_vals: bool = True):
        """Wrapper for all team components, will process all the values that need to be found for each team member
        in the battle. The arguments can turn on or off different components when embedding"""
        self.components = []

        self.num_fainted = num_fainted
        self.statuses = statuses
        self.hp_vals = hp_vals

        if num_fainted:
            self.components.append(NumFainted())

        if statuses:
            self.components.append(Statuses())

        if hp_vals:
            self.components.append(TeamHP)

        # attributes for describing battle embeddings
        self.length = sum(c.length for c in self.components)
        self.low = np.concatenate([
            c.low for c in self.components
        ], dtype=np.float32)
        self.high = np.concatenate([
            c.high for c in self.components
        ], dtype=np.float32)

    def embed(self, battle: AbstractBattle) -> np.ndarray:
        fainted_mon_team = 0
        fainted_mon_opponent = 0
        status_vals = np.zeros(Statuses.length, dtype=np.float32)
        hp_vals = np.zeros(TeamHP.length, dtype=np.float32)

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
            status_vals[i + 6] = 1 if mon.status else 0
            hp_vals[i + 6] = mon.current_hp_fraction

        fainted_mon_team /= 6
        fainted_mon_opponent /= 6

        final_vector = []

        if self.num_fainted:
            final_vector.append([fainted_mon_team, fainted_mon_opponent])

        if self.statuses:
            final_vector.append(status_vals)

        if self.hp_vals:
            final_vector.append(hp_vals)

        return np.concatenate(final_vector, dtype=np.float32)


class GameState:
    def __init__(self, components: list[StateComponent] = None):
        if components is None:
            components = [
                # active pokemon description
                MoveComponents(),
                TeamComponents(),
                # stat boosts for active Pokémon (ally, opponent)
                StatBoosts(),
                # side effects for both teams
                OneSideEffects(),
                # current field effects
                FullFieldEffects(),
                # notable abilities
                NotableAbilities(),
                # estimated matchups for all switches
                EstimatedMatchups()
            ]

        self._components: list[StateComponent] = components

        # attributes for describing battle embeddings
        self.length = sum(c.length for c in self._components)
        self.low = np.concatenate([
            c.low for c in self._components
        ], dtype=np.float32)
        self.high = np.concatenate([
            c.high for c in self._components
        ], dtype=np.float32)
        self.description = Box(self.low, self.high, dtype=np.float32)

    def embed_state(self, battle: AbstractBattle) -> np.ndarray:
        """Converts battle object to a numpy array representing the battle state"""

        # sometimes this isnt updated yet so we wait here for that
        while battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            pass

        # pass battle to each component to embed
        final_vector = np.concatenate([component.embed(battle) for component in self._components], dtype=np.float32)

        return final_vector

    @property
    def components(self) -> list[StateComponent]:
        """Returns the flattened list of components for this game state object"""
        components = []
        for c in self._components:
            if hasattr(c, 'components'):
                components.extend(c.components)
            else:
                components.append(c)
        return components

    @classmethod
    def from_component_list(cls, cfg_components: list[str]):
        """Returns a new GameState object constructed from the given component list"""
        move_components = [c.__name__ for c in MoveComponents.possible_components]
        team_components = [c.__name__ for c in TeamComponents.possible_components]

        # args for constructing combined objects to speed up embedding
        move_component_args = [component in cfg_components for component in move_components]
        team_component_args = [component in cfg_components for component in team_components]

        processed = []

        # check if we have added the MoveComponents() or TeamComponents() objects yet
        added_moves = False
        added_teams = False
        for component in cfg_components:
            if component in move_components:
                if not added_moves:
                    processed.append(MoveComponents(*move_component_args))
                    added_moves = True
                continue
            elif component in team_components:
                if not added_teams:
                    processed.append(TeamComponents(*team_component_args))
                    added_teams = True
                continue
            # create a new instance of that component class
            processed.append(globals()[component]())

        return cls(components=processed)
