# Game State Vector

In the reinforcement learning training, we need to be able to convert a Pokémon battle to a state vector that we can 
feed to the network. This 'state' vector is described here. I have not included all known moves by ally Pokémon because 
there are 122 unique moves in the teams we are using, and that would add 2,928 items to the state vector. The different
components were not all being used at once but this is an overview of all the different state components that were used
throughout the process of creating this project.

## Components
The number following the state is how many numbers it takes to describe in the vector.
* Move Base Power (4)
* Move Damage Multiplier (4)
* Move Damage Estimate (4)
  * This is used in place of the previous 2 sometimes, and it takes into account the accuracy of a move
* Move Tags (14*4)
  * Extra info for each of the 4 moves, such as if it is physical, special, or neither.
  * Tags are: Category, ally effect (ex. light screen), opponent effect (stealth rock), self switch (u-turn), 
  enemy switch (roar), weather, the move's priority, if the move self-destructs, and the recoil of the move.
* Number of Fainted Allies (1)
* Number of Fainted Opponents (1)
* Statuses on Ally Pokémon (6)
  * Paralysis, Sleep, Frozen, etc. all are 1 in this vector of 0s or 1s.
* Statuses on Opponents (6)
* Statuses on active Pokémon (7 * 2)
  * One-hot encoded vector for each possible status on the 2 active Pokémon
* Ally HP Amount (6)
* Opponent HP Amount (6)
* Ally Stat Changes (7)
* Opponent Stat Changes (7)
* OHE (One-hot encoding) Ally Pokémon (67 * 6)
  * Only the 67 Pokémon possible from the [pre-defined teams](teams.txt) are in this vector to make it smaller.
* OHE Opponent Pokémon (67)
* A flag for if there is a substitute out in each side (2)
* Active Pokémon Stats (5 * 2)
  * The base statistics for each active Pokémon (excluding HP as that information is provided elsewhere)
* Estimated Matchups (6)
  * Assigns a score to how well Pokémon are expected to perform against the opposing 
  Pokémon using simple calculations with the Pokémon stats and types.
  * Included as the type matchup information isn't provided elsewhere in the state vector.

### Field Effects
#### For each player
* Light Screen (2)
* Reflect (2)
* Stealth Rock (2)
* Spikes (2)
  * Layers from 0-3
* Toxic Spikes (2)
  * Layers from 0-2
* Substitute (2)
* Perish Song (2)
  * 0-3, counts turns  since it was played (0 means it hasn't been played)

#### Full Field
* Rain (1)
* Sandstorm (1)
* Hail (1)
* Sun (1)
* Trick Room (1)
* Gravity (1)
* Mud Sport (1)
* Water Sport (1)