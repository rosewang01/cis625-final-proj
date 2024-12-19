# cis625-final-proj

Marwan Gedeon Achi, Rose Wang

### Getting Started
1. Create a game using the `Game` class. The following code creates a random 2 player, 2 action game.
   ```{Python}
   game = Game(2, [2, 2], game_type=Game.RANDOM)
   ```
2. Instantiate the solvers as follows:
   ```{Python}
   lp_solver = LinearProgrammingSolver(game)
   lp_welfare_solver = LinearProgrammingSolver(game, maximize_welfare=True)
   sr_solver = SwapRegretSolver(game, epsilon=0.1)
   ```
3. Run solve!
   ```{Python}
   distribution = lp_solver.solve()
   ```
4. From the outputted correlated equilibria, you can assess the expected welfare, whether there were any violations of the constrains in th eoutputted distribution, and visualize the violations.

### Setting up your environment
If you'd like a quick start to your conda environment, run either of the following commands to create the environment and install the necessary packages:
- `conda create -n <environment-name> --file env/req.txt python=3.10`
- `conda create -n <environment-name> --file env/environment.yml python=3.10`

### Benchmarking
All benchmarking logs are stored in the `log/` folder. All analyses on the logs are in the `benchmarking.ipynb` notebook. Unit tests for specific games are in `unittests.py` and the centralized benchmarking framework is in `benchmarker.py` and `tester.py`

Source code for the `Game` class is in `game.py`, and the solvers are outlined in `lp.py` for linear programming and `sr.py` for swap regret.
