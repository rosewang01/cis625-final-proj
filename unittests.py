import numpy as np
from game import Game
from lp import LinearProgrammingSolver
from sr import SwapRegretSolver
from tester import collect_violations, visualize_violations_heatmap

def chicken_test():
    num_players = 2
    num_actions = [2, 2]
    game = Game(num_players, num_actions, game_type=Game.CHICKEN)
    print("Game type: Chicken")
    print(game.payoff_matrices)
    print("Solving with Linear Programming...")
    solver = LinearProgrammingSolver(game)
    distribution = solver.solve()
    print(distribution)
    violations = collect_violations(game, distribution)
    print(violations)
    print("Solving with Swap Regret...")
    solver = SwapRegretSolver(game)
    distribution = solver.solve()
    print(distribution)
    violations = collect_violations(game, distribution, epsilon=solver.epsilon)
    print(violations)

def congestion_test():
    num_players = 2
    num_actions = [2, 2]
    game = Game(num_players, num_actions, game_type=Game.CUSTOM, payoff_matrices=[np.array([[-5, -2], [-3, -6]]), np.array([[-5, -3], [-2, -6]])])
    print("Game type: Congestion")
    print(game.payoff_matrices)
    print("Solving with Linear Programming...")
    solver = LinearProgrammingSolver(game, maximize_welfare=True)
    distribution = solver.solve()
    print(distribution)
    violations = collect_violations(game, distribution)
    print("Solving with Swap Regret...")
    solver = SwapRegretSolver(game)
    distribution = solver.solve()
    print(distribution)
    violations = collect_violations(game, distribution, epsilon=solver.epsilon)
    print(violations)

def main():
    chicken_test()
    congestion_test()

if __name__ == "__main__":
    chicken_test()