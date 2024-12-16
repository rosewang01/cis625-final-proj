from game import Game
from lp import LinearProgrammingSolver
from sr import SwapRegretSolver
import numpy as np
from tester import benchmark_solvers, social_welfare

def log_results(nplayers, nactions, file_path, include_sr=True):
    game = Game(nplayers, [nactions] * nplayers, game_type=Game.RANDOM)

    lp_solver = LinearProgrammingSolver(game)
    lp_welfare_solver = LinearProgrammingSolver(game, maximize_welfare=True)
    
    if include_sr:
        sr_solver = SwapRegretSolver(game, epsilon=0.1)
        solvers = [lp_solver, lp_welfare_solver, sr_solver]
    else:
        solvers = [lp_solver, lp_welfare_solver]

    results = benchmark_solvers(game, solvers, social_welfare)

    with open(file_path, "a") as f:
        for solver, result in results.items():
            max_violation = max([v["magnitude"] for v in result["violations"]]) if result["violations"] else 0
            n_violations = len(result["violations"])
            welfare = result["welfare"]
            runtime = result["runtime"]
            f.write(f"{nplayers},{nactions},{solver},{runtime},{max_violation},{n_violations},{welfare}\n")

def lp_benchmark():
    nplayers_arr = [2, 4, 7, 10]
    nactions_arr = [2, 10, 25, 50]

    file_path = "logs/lp_benchmarking.csv"
    with open(file_path, "w") as f:
        f.write("NPlayers,MaxNActions,Solver,Runtime,MaxViolation,NViolations,Welfare\n")
    
    for nplayers in nplayers_arr:
        nactions = 2
        print(f"Benchmarking for {nplayers} players and {nactions} actions...")

        for i in range(10):
            log_results(nplayers, nactions, file_path, include_sr=False)

    for nactions in nactions_arr:
        nplayers = 2
        print(f"Benchmarking for {nplayers} players and {nactions} actions...")

        for i in range(10):
            log_results(nplayers, nactions, file_path, include_sr=False)

    print(f"Benchmarking complete. Results logged to {file_path}.")

def sr_benchmark():
    nplayers_arr = [2, 3, 4, 5]
    nactions_arr = [2, 3, 4, 5]

    file_path = "logs/sr_benchmarking.csv"
    with open(file_path, "w") as f:
        f.write("NPlayers,MaxNActions,Solver,Runtime,MaxViolation,NViolations,Welfare\n")
    
    for nplayers in nplayers_arr:
        nactions = 2
        print(f"Benchmarking for {nplayers} players and {nactions} actions...")

        for i in range(5):
            log_results(nplayers, nactions, file_path)
    
    for nactions in nactions_arr:
        nplayers = 2
        print(f"Benchmarking for {nplayers} players and {nactions} actions...")

        for i in range(5):
            log_results(nplayers, nactions, file_path)

    print(f"Benchmarking complete. Results logged to {file_path}.")

if __name__ == "__main__":
    sr_benchmark()