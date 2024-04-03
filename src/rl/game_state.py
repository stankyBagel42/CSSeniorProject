from abc import abstractmethod
from pathlib import Path

import numpy as np
from gym.spaces import Box
from poke_env.environment import Battle, PokemonType, Effect

from src.utils.pokemon import pokemon_to_index, GEN_DATA, STAT_IDX, SideCondition, Field, Weather, NotableAbility, Status
from src.utils.general import read_yaml


class StateComponent:
    """Helper to make it easier to construct game low and high values"""
    length: int
    low: np.ndarray
    high: np.ndarray

    @abstractmethod
    def embed(self, battle: Battle) -> np.ndarray:
        pass

class PokemonStats(StateComponent):
    """Base stats of both active Pokémon, excluding HP"""
    length = 10
    low = np.full(length, 1, dtype=np.float32)
    high = np.full(length, 255, dtype=np.float32)
    def embed(self, battle: Battle) -> np.ndarray:
        stats = np.ones(self.length)
        active_pokemon = battle.active_pokemon
        opponent_pokemon = battle.opponent_active_pokemon
        for i, stat in enumerate(['atk','def','spa','spd','spe']):
            stats[i] = active_pokemon.base_stats[stat]
            stats[i+5] = opponent_pokemon.base_stats[stat]
        return stats

# TODO: Maybe add a negative 1 to the move damage so the value is negative if 0 damage instead of neutral (0).
class MoveDamage(StateComponent):
    """Combination of MovePower, MoveMultiplier, and move accuracy. It will just calculate the power * multiplier *
    accuracy/100. It is the expected power of a move."""
    length = 4
    low = np.full(length, -1, dtype=np.float32)
    # 4x multiplier, 400 base power, 100% accuracy would result in a 16 for the maximum damage, 150% for STAB
    high = np.full(length, 24, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
        moves_base_power = np.full(self.length, -1, dtype=np.float32)
        active_pokemon = battle.active_pokemon
        opponent_mon = battle.opponent_active_pokemon

        # get move information
        for i, move in enumerate(battle.available_moves):
            # Simple rescaling to facilitate learning
            base_power = move.base_power / 100
            if move.type:
                # type multiplier
                base_power *= move.type.damage_multiplier(
                    opponent_mon.type_1,
                    opponent_mon.type_2,
                    type_chart=GEN_DATA.type_chart
                )
                # STAB
                if move.type in active_pokemon.types:
                    base_power *= 1.5
                # treat all abilities as equally likely, so if the opponent can ONLY have levitate
                # (or a similar ability), ground moves will never do anything
                if (move.type == PokemonType.GROUND and 'levitate' in opponent_mon.possible_abilities) or \
                   (move.type == PokemonType.WATER and 'waterabsorb' in opponent_mon.possible_abilities) or \
                   (move.type == PokemonType.FIRE and 'flashfire' in opponent_mon.possible_abilities) or \
                   (move.type == PokemonType.ELECTRIC and 'voltabsorb' in opponent_mon.possible_abilities):
                    ability_factor = (1 / len(opponent_mon.possible_abilities))
                    base_power *= (1 - ability_factor)

            # scale based on the accuracy to get the expected move power
            moves_base_power[i] = base_power * move.accuracy
        return moves_base_power


class MovePower(StateComponent):
    """Move base power (divided by 100, so it isn't too large compared to others)"""
    length = 4
    low = np.full(length, -1, dtype=np.float32)  # min of -1 for none
    high = np.full(length, 4, dtype=np.float32)  # max of 4

    def embed(self, battle: Battle) -> np.ndarray:
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

    def embed(self, battle: Battle) -> np.ndarray:
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


class MoveTags(StateComponent):
    """Broad categories of moves generally, also includes some specific move categories for specific moves like 'protect'."""
    # tags are defined as such:
    # MOVE CATEGORY (physical, special, status)
    # ALLY_SIDE EFFECT (tailwind, light screen, etc.)
    # ENEMY SIDE EFFECT (Stealth rock, spikes, etc.)
    # SELF SWITCH (baton pass, u-turn, etc.)
    # ENEMY SWITCH (roar, whirlwind, etc.)
    # WEATHER (starts a weather condition)
    # PRIORITY
    # SELF-DESTRUCT
    # RECOIL

    num_tags = 14
    length = 4*num_tags
    low = np.full(length, -1, dtype=np.float32)  # -1 when the moves aren't available
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
        flags = []
        moves = battle.available_moves

        for i in range(4):
            # fill unavailable moves with -1
            if i >= len(moves):
                flags.extend(-1 for _ in range(self.num_tags))
                continue
            move = moves[i]
            move_flags = []
            # add the move category
            move_category = [0, 0, 0]  # physical, special, status
            move_category[move.category.value - 1] = 1
            move_flags.extend(move_category)

            # first 2 flags are ALLY_SIDE (tailwind, light screen, etc.) then ENEMY_SIDE (stealth rock, spikes, etc.)
            if move.side_condition is not None:
                if 'ally' in move.target:
                    move_flags.extend([True, False])
                else:
                    move_flags.extend([False, True])
            else:
                move_flags.extend([False, False])
            priority = move.priority
            if priority > 1:
                priority = 1
            elif priority < 0:
                priority = -1
            # store flags here for easier readability, will convert to bool/int when adding
            raw_flags = [
                move.self_switch,
                move.weather,
                move.status,
                move.self_boost,
                move.heal,
                move.is_protect_move,
                (priority + 1) / 2,
                move.self_destruct,
                move.recoil
            ]

            move_flags.extend(bool(flag) for flag in raw_flags)
            flags.extend(move_flags)

        return np.array(flags, dtype=np.float32)


class NotableAbilities(StateComponent):
    """If a Pokémon can have levitate, flash fire, or any other damage altering abilities. First 5 are ally, next are
    opponent"""

    # the abilities are levitate, volt/water absorb, flash fire, magic guard
    num_abilities = len(NotableAbility)
    length = num_abilities * 2
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
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

    def embed(self, battle: Battle) -> np.ndarray:
        ally_fainted = 0
        opponent_fainted = 0
        for mon in battle.team.values():
            if mon.fainted:
                ally_fainted += 1

        for mon in battle.opponent_team.values():
            if mon.fainted:
                opponent_fainted += 1

        return np.array([ally_fainted / 6, opponent_fainted / 6], dtype=np.float32)

class ActiveStatus(StateComponent):
    num_statuses = len(Status)
    length = 2 * num_statuses
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
        status = np.zeros(self.length, dtype=np.float32)
        active_pokemon = battle.active_pokemon
        opponent_pokemon = battle.opponent_active_pokemon
        if active_pokemon.status:
            idx = Status.__getitem__(active_pokemon.status.name.upper()).value
            status[idx] = 1
        if opponent_pokemon.status:
            idx = Status.__getitem__(opponent_pokemon.status.name.upper()).value
            status[idx + self.num_statuses] = 1
        return status

class IsSubstitute(StateComponent):
    """If the ally has a substitute, then if the enemy does"""
    length = 2
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
        subs = []
        for mon in (battle.active_pokemon, battle.opponent_active_pokemon):
            has_substitute = 1 if Effect.SUBSTITUTE in mon.effects.keys() else 0
            subs.append(has_substitute)
        return np.array(subs,dtype=np.float32)

class Statuses(StateComponent):
    """Status effects on a both teams (ally, then opponent), one for an active status, zero otherwise for no active
     status"""
    length = 12
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
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

    def embed(self, battle: Battle) -> np.ndarray:
        hp_vals = np.zeros(self.length, dtype=np.float32)
        for i, mon in enumerate(battle.team.values()):
            hp_vals[i] = mon.current_hp_fraction

        for i, mon in enumerate(battle.opponent_team.values()):
            hp_vals[i + 6] = mon.current_hp_fraction

        return hp_vals


class StatBoosts(StateComponent):
    """Pokemon stat changes, -6 low, 6 high, (ally first, then opponent)"""

    num_boostable_stats = len(STAT_IDX)
    length = 2 * num_boostable_stats
    low = np.full(length, -6, dtype=np.float32)
    high = np.full(length, 6, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
        ally_boosts = np.zeros(self.num_boostable_stats, dtype=np.float32)
        opponent_boosts = np.zeros(self.num_boostable_stats, dtype=np.float32)
        # encode stat boosts
        for boost, val in battle.active_pokemon.boosts.items():
            if hasattr(STAT_IDX, boost.upper()):
                ally_boosts[STAT_IDX.__getitem__(boost.upper()).value] = val
        for boost, val in battle.opponent_active_pokemon.boosts.items():
            if hasattr(STAT_IDX, boost.upper()):
                ally_boosts[STAT_IDX.__getitem__(boost.upper()).value] = val
        return np.concatenate([ally_boosts, opponent_boosts])


class OneSideEffects(StateComponent):
    """Player-dependent field effects (stealth rock, light screen, etc.) (ally first, then opponents)"""

    num_side_conditions = len(SideCondition)
    length = 10
    low = np.zeros(length, dtype=np.float32)
    high = np.full(length, 3, dtype=np.float32)  # 3 stacks of spikes/perish song

    def embed(self, battle: Battle):
        side_conditions = np.zeros(self.length, dtype=np.float32)

        # encode single-sided conditions for both teams
        # poke-env marks non-stackable conditions with the turn they were played, so we have to check if it is
        # stackable or not to return the correct state
        for env_condition, val in battle.side_conditions.items():
            condition = SideCondition.__getitem__(env_condition.name.upper())
            idx = condition.value
            real_val = val if condition.is_stackable() else 3
            side_conditions[idx] = real_val
        for env_condition, val in battle.opponent_side_conditions.items():
            condition = SideCondition.__getitem__(env_condition.name.upper())
            idx = condition.value
            real_val = val if condition.is_stackable() else 3
            side_conditions[idx + self.num_side_conditions] = real_val

        return side_conditions


class FullFieldEffects(StateComponent):
    """Effects that are in the entire field (weather, trick room, etc.)"""

    num_field_conditions = len(Field)
    num_weather = len(Weather)
    length = 8
    low = np.zeros(length, dtype=np.float32)
    high = np.ones(length, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
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

    def embed(self, battle: Battle) -> np.ndarray:
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
    """Estimated matchups for all ally Pokémon vs enemy Pokémon, code is modified from the poke_env
    SimpleHeuristicPlayer"""
    length = 6
    low = np.full(length, -4.5, dtype=np.float32)
    high = np.full(length, 7.1, dtype=np.float32)

    def embed(self, battle: Battle) -> np.ndarray:
        matchups = np.zeros(self.length, dtype=np.float32)
        opponent = battle.opponent_active_pokemon
        for i, pokemon in enumerate([battle.active_pokemon, *battle.available_switches]):
            score = max([opponent.damage_multiplier(t) for t in pokemon.types if t is not None]) # 4 max, 0.25 min
            score -= max(
                [pokemon.damage_multiplier(t) for t in opponent.types if t is not None]
            ) # 4 max, 0.25 min
            if pokemon.base_stats["spe"] > opponent.base_stats["spe"]:
                score += 0.1
            elif opponent.base_stats["spe"] > pokemon.base_stats["spe"]:
                score -= 0.1
                if pokemon.current_hp_fraction < 0.5:
                    score -= 0.3

            # move effectiveness
            for move in pokemon.moves.values():
                if move.base_power > 0:
                    if opponent.damage_multiplier(move) > 1:
                        score += opponent.damage_multiplier(move)/6

            # notable abilities
            if pokemon.ability == 'waterabsorb' and PokemonType.WATER in opponent.types:
                score += 0.1
            elif pokemon.ability == 'levitate' and PokemonType.GROUND in opponent.types:
                score += 0.1
            elif pokemon.ability == 'voltabsorb' and PokemonType.ELECTRIC in opponent.types:
                score += 0.1
            elif pokemon.ability == 'flashfire' and PokemonType.FIRE in opponent.types:
                score += 0.1

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

    def embed(self, battle: Battle):
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

    def embed(self, battle: Battle) -> np.ndarray:
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
    def __init__(self, components: list[StateComponent] = None, normalize:bool = True):
        if components is None:
            components = [
                # active pokemon description
                MoveComponents(),
                MoveTags(),  # move tags
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

        self.normalize = normalize

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

    def embed_state(self, battle: Battle) -> np.ndarray:
        """Converts battle object to a numpy array representing the battle state"""

        # sometimes this isn't updated yet, so we wait here for that
        while battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            pass

        # pass battle to each component to embed
        final_vector = np.concatenate([component.embed(battle) for component in self._components], dtype=np.float32)
        if self.normalize:
            final_vector = final_vector - self.low
            final_vector = final_vector / (self.high - self.low)
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
    def from_component_list(cls, cfg_components: list[str], normalize:bool = True):
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

        return cls(components=processed, normalize=normalize)


def get_game_state(model_path: str | Path, model_cfg_dict: dict) -> GameState:
    """Helper function to get the game state definition for a given checkpoint, used to ensure even older checkpoints
    (before the game state was saved in the checkpoint cfg) can be reloaded easily."""
    model_path = Path(model_path)

    # if the game state was saved in the model config, just load it there
    if 'game_state' in model_cfg_dict.keys():
        return model_cfg_dict['game_state']

    # if the game state was saved in the train config, load if from there
    train_cfg = read_yaml(model_path.parents[1] / 'train_config.yaml')
    if 'state_components' in train_cfg.keys():
        return GameState.from_component_list(train_cfg['state_components'])

    # otherwise, the checkpoint is too old and we throw an exception
    raise RuntimeError(f"Model at {model_path} didn't have game state in the model or training configs, so the game "
                       f"state must be provided when creating the player/agent object.")
