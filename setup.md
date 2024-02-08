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
