import numpy as np
from game import Game
from lp import LinearProgrammingSolver
from sr import SwapRegretSolver
from tester import collect_violations, visualize_violations_heatmap

def chicken_test():
    num_players = 2
    num_actions = [2, 2]
    game = Game(num_players, num_actions, game_type=Game.CHICKEN)
    # solver = LinearProgrammingSolver(game, maximize_welfare=True)
    solver = SwapRegretSolver(game, T=10000, learning_rate=0.1)
    distribution = solver.solve()
    print(distribution)
    violations = collect_violations(game, distribution, epsilon=1e-3)
    if len(violations) > 0:
        if len(violations) > 1:
            visualize_violations_heatmap(game, violations)
        else:
            print(violations)
    return violations

def congestion_test():
    num_players = 2
    num_actions = [2, 2]
    game = Game(num_players, num_actions, game_type=Game.CONGESTION)
    solver = LinearProgrammingSolver(game)
    distribution = solver.solve()
    print(distribution)
    violations = collect_violations(game, distribution)
    visualize_violations_heatmap(game, violations)
    return violations

def main():
    chicken_violations = chicken_test()
    congestion_violations = congestion_test()
    return chicken_violations, congestion_violations

if __name__ == "__main__":
    chicken_test()