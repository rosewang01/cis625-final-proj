import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

class LPSolver:
    def __init__(self, game):
        """
        Initialize the solver with a NormalFormGame instance.

        Parameters:
        - game (NormalFormGame): An instance of the NormalFormGame class.
        """
        self.game = game

    def find_correlated_equilibrium(self, maximize_welfare=False):
        """
        Find a correlated equilibrium using linear programming.

        Parameters:
        - maximize_welfare (bool): If True, maximizes social welfare.

        Returns:
        - dict: A dictionary of joint action probabilities.
        """
        joint_action_space = tuple(self.game.num_actions)
        all_actions = np.ndindex(joint_action_space)
        prob_vars = {action: LpVariable(f"p_{action}", lowBound=0) for action in all_actions}

        prob = LpProblem("CorrelatedEquilibrium", LpMaximize if maximize_welfare else LpProblem)

        prob += lpSum(prob_vars.values()) == 1

        for player in range(self.game.num_players):
            for action in range(self.game.num_actions[player]):
                for alt_action in range(self.game.num_actions[player]):
                    if action != alt_action:
                        constraint = 0
                        for joint_action, prob_var in prob_vars.items():
                            if joint_action[player] == action:
                                joint_action_alt = list(joint_action)
                                joint_action_alt[player] = alt_action
                                joint_action_alt = tuple(joint_action_alt)
                                constraint += (self.game.payoff_matrices[player][joint_action] -
                                               self.game.payoff_matrices[player][joint_action_alt]) * prob_var
                        prob += constraint >= 0

        if maximize_welfare:
            prob += lpSum(
                prob_var * sum(self.game.payoff_matrices[player][joint_action]
                                for player in range(self.game.num_players))
                for joint_action, prob_var in prob_vars.items()
            )

        prob.solve()

        return {action: prob_vars[action].value() for action in prob_vars}
    
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
