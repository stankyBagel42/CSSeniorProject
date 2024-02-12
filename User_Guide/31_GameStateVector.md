### [prev](./22_StartingTheServer) | [next](./41_TheTeams.md)
# Game State Vector

In the reinforcement learning training, we need to be able to convert a Pokémon battle to a state vector that we can 
feed to the network. This 'state' vector is described here. I have not included all known moves by ally Pokémon because 
there are 122 unique moves in the teams we are using, and that would add 2,928 items to the state vector. 


## Components
The number following the state is how many numbers it takes to describe in the vector.
* Move Base Power (4)
* Move Damage Multiplier (4)
* Number of Fainted Allies (1)
* Number of Fainted Opponents (1)
* Statuses on Ally Pokémon (6)
  * Paralysis, Sleep, Frozen, etc.
* Statuses on Opponents (6)
* Ally HP Amount (6)
* Opponent HP Amount (6)
* Ally Stat Changes (7)
* Opponent Stat Changes (7)
* OHE (One-hot encoding) Ally Pokémon (67 * 6)
  * Only the 67 Pokémon possible from the [pre-defined teams](teams.txt) are in this vector to make it smaller.
* OHE Opponent Pokémon (67)


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

### [prev](./22_StartingTheServer) | [next](./41_TheTeams.md)