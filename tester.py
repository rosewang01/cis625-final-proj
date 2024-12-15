import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from collections import defaultdict
from rnfgame import NormalFormGame
from lp import LinearProgrammingSolver
from sr import SwapRegretSolver

def collect_violations(game, distribution, epsilon=1e-6):
    """
    Collect violations of correlated equilibrium constraints.

    Returns:
    - list[dict]: A list of violation records.
    """
    violations = []

    if not np.isclose(sum(distribution.values()), 1.0, atol=epsilon):
        violations.append("The probabilities do not sum to 1.")
        return violations

    if any(prob < 0 for prob in distribution.values()):
        violations.append("The distribution contains negative probabilities.")
        return violations

    for player in range(game.num_players):
        for current_action in range(game.num_actions[player]):
            for alt_action in range(game.num_actions[player]):
                if current_action != alt_action:
                    lhs = 0
                    rhs = 0
                    for joint_action, prob in distribution.items():
                        if prob > 0:
                            payoff_current = game.payoff_matrices[player][joint_action]
                            joint_action_alt = list(joint_action)
                            joint_action_alt[player] = alt_action
                            joint_action_alt = tuple(joint_action_alt)
                            payoff_alt = game.payoff_matrices[player][joint_action_alt]
                            lhs += prob * payoff_current
                            rhs += prob * payoff_alt
                    if lhs < rhs - epsilon:
                        violations.append({
                            "player": player,
                            "current_action": current_action,
                            "alt_action": alt_action,
                            "magnitude": rhs - lhs
                        })
    return violations

def visualize_violations_heatmap(game, violations):
    """
    Visualize incentive constraint violations as a heatmap for each player.

    Parameters:
    - game (NormalFormGame): The normal form game.
    - violations (list[dict]): A list of violation records with player, current action,
                               alternative action, and magnitude.
    """
    for player in range(game.num_players):
        num_actions = game.num_actions[player]
        violation_matrix = np.zeros((num_actions, num_actions))

        for violation in violations:
            if violation['player'] == player:
                i, j = violation['current_action'], violation['alt_action']
                violation_matrix[i, j] = violation['magnitude']

        plt.figure(figsize=(8, 6))
        sns.heatmap(violation_matrix, annot=True, fmt=".2f", cmap="Reds", cbar=True)
        plt.title(f"Player {player}'s Violation Heatmap")
        plt.xlabel("Alternative Action")
        plt.ylabel("Current Action")
        plt.show()

def social_welfare(distribution, game):
    """
    Compute the social welfare of a given distribution over action profiles.

    Parameters:
    - distribution (dict): A distribution over action profiles (keys: action profiles, values: probabilities).
    - game (NormalFormGame): The game object.

    Returns:
    - float: The computed social welfare.
    """
    welfare = 0
    for action_profile, prob in distribution.items():
        if prob > 0:
            welfare += prob * sum(
                game.payoff_matrices[player][action_profile] for player in range(game.num_players)
            )
    return welfare

def benchmark_solvers(game, solvers, welfare_func):
    """
    Benchmark the performance and welfare outcome of various solvers on a given game.

    Parameters:
    - game (NormalFormGame): The game to evaluate solvers on.
    - solvers (list[class instances]): A list of solver classes. Each solver should accept the game as input and return a distribution over action profiles.
    - welfare_func (callable): A function that computes the welfare of a given distribution over action profiles.
        Input: (distribution, game)
        Output: Welfare value (float).

    Returns:
    - dict: A dictionary where each solver is a key, mapping to its runtime, violations, and welfare outcome.
    """
    results = defaultdict(dict)

    for solver in solvers:
        solver_name = solver.get_name()
        print(f"Benchmarking {solver_name}...")
        
        start_time = time.perf_counter()
        distribution = solver.solve()
        end_time = time.perf_counter()
        runtime = end_time - start_time

        welfare = welfare_func(distribution, game)

        violations = collect_violations(game, distribution)

        results[solver_name] = {
            "runtime": runtime,
            "violations": violations,
            "welfare": welfare
        }
        print(f"Runtime: {runtime:.4f} seconds")

    return results


def main():
    file_path = "benchmarking.csv"
    game = NormalFormGame(2, [2, 2])

    lp_solver = LinearProgrammingSolver(game)
    lp_welfare_solver = LinearProgrammingSolver(game, maximize_welfare=True)
    sr_solver = SwapRegretSolver(game, T=1000, learning_rate=0.1)

    solvers = [lp_solver, lp_welfare_solver, sr_solver]

    results = benchmark_solvers(game, solvers, social_welfare)

    # log results
    with open(file_path, "w") as f:
        f.write("NPlayers,MaxNActions,Solver,Runtime,Violations,Welfare\n")
        for solver, result in results.items():
            runtime = result["runtime"]
            violations = len(result["violations"])
            welfare = result["welfare"]
            f.write(f"{game.num_players},{max(game.num_actions)},{solver},{runtime},{violations},{welfare}\n")

    print("Benchmarking complete. Results logged to benchmarking.csv.")

if __name__ == "__main__":
    main()
