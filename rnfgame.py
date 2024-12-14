import numpy as np

class NormalFormGame:
    def __init__(self, num_players, num_actions):
        """
        Initialize a normal form game.

        Parameters:
        - num_players (int): Number of players in the game.
        - num_actions (list[int]): List of actions for each player, where num_actions[i] is the number of actions for player i.
        """
        self.num_players = num_players
        self.num_actions = num_actions

        self.payoff_matrices = self._generate_random_payoff_matrices()

    def _generate_random_payoff_matrices(self):
        """
        Generate random payoff matrices for each player.

        Returns:
        - list[np.ndarray]: A list of payoff matrices, one for each player.
        """
        payoff_matrices = []
        joint_action_space = tuple(self.num_actions)
        
        for _ in range(self.num_players):
            # Each player's payoff matrix has the same shape as the joint action space
            payoff_matrix = np.random.uniform(-10, 10, size=joint_action_space)
            payoff_matrices.append(payoff_matrix)
        
        return payoff_matrices

    def get_payoff(self, actions):
        """
        Get the payoffs for all players given a specific joint action.

        Parameters:
        - actions (tuple[int]): A tuple specifying the action of each player.

        Returns:
        - list[float]: A list of payoffs, one for each player.
        """
        return [self.payoff_matrices[player][actions] for player in range(self.num_players)]

    def __repr__(self):
        """String representation of the game."""
        repr_str = f"Normal Form Game with {self.num_players} players\n"
        for i, matrix in enumerate(self.payoff_matrices):
            repr_str += f"Player {i+1}'s Payoff Matrix:\n{matrix}\n\n"
        return repr_str