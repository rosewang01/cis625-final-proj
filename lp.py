import numpy as np
import pulp
from pulp import PULP_CBC_CMD
class LinearProgrammingSolver:
    def __init__(self, game, maximize_welfare=False):
        """
        Initialize the solver with a NormalFormGame instance.

        Parameters:
        - game (NormalFormGame): An instance of the NormalFormGame class.
        """
        self.game = game
        self.maximize_welfare = maximize_welfare

    def get_name(self):
        if self.maximize_welfare:
            return "Linear Programming-Maximize Welfare"
        else:
            return "Linear Programming"
        

    def solve(self):
        """
        Find a correlated equilibrium using linear programming.

        Parameters:
        - maximize_welfare (bool): If True, maximizes social welfare.

        Returns:
        - dict: A dictionary of joint action probabilities.
        """
        prob = pulp.LpProblem("Correlated_Equilibrium", pulp.LpMaximize)

        action_profiles = self.game.get_action_profiles()
        
        p = pulp.LpVariable.dicts("p", action_profiles, lowBound=0, upBound=1)

        prob += pulp.lpSum([p[a] for a in action_profiles]) == 1, "Normalization"

        for i in range(self.game.num_players):
            for a_i in range(self.game.num_actions[i]):
                for b_i in range(self.game.num_actions[i]):
                    if a_i == b_i:
                        continue
                    prob += (
                        pulp.lpSum(
                            p[a] * self.game.payoff_matrices[i][a] for a in action_profiles if a[i] == a_i
                        )
                        >= pulp.lpSum(
                            p[a] * self.game.payoff_matrices[i][a[:i] + (b_i,) + a[i + 1:]] 
                            for a in action_profiles if a[i] == a_i
                        ),
                        f"Player_{i}_Action_{a_i}_to_{b_i}"
                    )

        if self.maximize_welfare:
            prob += pulp.lpSum(
                p[a] * sum(self.game.payoff_matrices[i][a] for i in range(self.game.num_players)) for a in action_profiles
            ), "Maximize_Welfare"
        else:
            prob += 0, "Dummy_Objective"

        status = prob.solve(PULP_CBC_CMD(msg=False))

        # for i in range(self.game.num_players):
        #     for a_i in range(self.game.num_actions[i]):
        #         for b_i in range(self.game.num_actions[i]):
        #             if a_i != b_i:
        #                 lhs = sum(pulp.value(p[a]) * self.game.payoff_matrices[i][a] for a in action_profiles if a[i] == a_i)
        #                 rhs = sum(pulp.value(p[a]) * self.game.payoff_matrices[i][a[:i] + (b_i,) + a[i + 1:]] for a in action_profiles if a[i] == a_i)
        #                 print(f"Player {i}, Action {a_i} -> {b_i}: LHS={lhs:.4f}, RHS={rhs:.4f}")

        if pulp.LpStatus[status] == "Optimal":
            # Return the solution as a dictionary
            return {a: pulp.value(p[a]) for a in action_profiles}
        else:
            return None

    def print_correlated_equilibrium(self, correlated_eq):
        """
        Print the correlated equilibrium in a readable format.

        Parameters:
        - correlated_eq (dict): A dictionary of joint action probabilities.
        """
        print("Game:")
        print(self.game) 
        print("Correlated Equilibrium:")
        for joint_action, prob in correlated_eq.items():
            print(f"Joint Action {joint_action}: {prob:.2f}")
