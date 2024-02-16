# Project Setup
To set up the environment, follow these steps:

1. Install the latest versions of [anaconda](https://anaconda.org/) and [Node.js](https://nodejs.org/en/download/current).
2. Create the anaconda environment by running the following command from the repository root:

    ```conda env create -f environment.yml```
3. Pull the most recent Pokémon showdown repo with ``git submodule update --init``
4. Run the following command from the [Pokémon showdown directory](pokemon-showdown) to build the project:
   
    ```node build```

   *NOTE*: On Linux, follow the directions found in the Pokémon showdown submodule

## Starting the Server
Once the environment setup is complete, before running any of the python scripts make sure to start running the Pokémon 
showdown server on localhost by running the following command from the Pokémon showdown submodule directory.

```node pokemon-showdown start --no-security```

*The no-security flag is there so the RL agent users don't need a registered account to connect*

This will start the Pokémon Showdown server
## Running Scripts

To run the different scripts found in the `src` directory, first make sure the conda environment is active with
```conda activate pokemon``` and then run the scripts with the following command:

```python -m src.SCRIPT_NAME```

from the repository root directory. 

### The Different Scripts
* [selfplay](src/selfplay.py) is the main script for training RL agents against each other, all the configuration 
options are found in the [train_config.yaml](train_config.yaml) file
* [challenge_bot](src/challenge_bot.py) is the script used for playing against the trained agents. Just set the `model_path` variable to 
the .pt file and the `opponent_username` variable to whatever username you want the bot to challenge.
* [benchmark](src/benchmark.py) is the script used for benchmarking a specific model file. Just change the `MODEL_PATH` 
to the .pt file and it should benchmark for however many challenges are specified in the `NUM_CHALLENGES` variable. 
These benchmarks are automatically performed at the very end of training new agents with [selfplay](src/selfplay.py).
* [create_initial_replays](src/create_initial_replays.py) is the script used for creating replay buffers between bot 
players, such as the SimpleHeuristicPlayer provided by poke-env. These are used in the training process as a way to 
pre-train the networks with good data before they start playing (replays are specified in the `replay_buffer_path` list in the train config file).