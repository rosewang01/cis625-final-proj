import numpy as np

class Game:
    """
    Enum for the game type.
    """
    RANDOM = 0
    CHICKEN = 1
    CONGESTION = 2
    CUSTOM = 3
    def __init__(self, num_players, num_actions, game_type = RANDOM, payoff_matrices = None):
        """
        Initialize a normal form game.

        Parameters:
        - num_players (int): Number of players in the game.
        - num_actions (list[int]): List of actions for each player, where num_actions[i] is the number of actions for player i.
        - game_type (int): Type of game to generate (RANDOM, CHICKEN, CONGESTION, CUSTOM).
        - payoff_matrices (list[np.ndarray]): A list of payoff matrices, one for each player. Required if game_type is CUSTOM.
        """
        self.num_players = num_players
        self.num_actions = num_actions

        if game_type == Game.RANDOM:
            self.payoff_matrices = self._generate_random_payoff_matrices()
        elif game_type == Game.CHICKEN:
            self.payoff_matrices = self._generate_chicken_payoff_matrices()
        elif game_type == Game.CONGESTION:
            self.payoff_matrices = self._generate_congestion_payoff_matrices()
        elif game_type == Game.CUSTOM and payoff_matrices is not None:
            self.payoff_matrices = self._generate_custom_payoff_matrices(payoff_matrices)

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

    def _generate_chicken_payoff_matrices(self):
        """
        Generate payoff matrices for the Chicken game.

        Returns:
        - list[np.ndarray]: A list of payoff matrices, one for each player.
        """
        payoff_matrices = []

        if (self.num_players != 2):
            raise ValueError("Chicken game can only be played by two players.")
        if (self.num_actions[0] != self.num_actions[1]):
            raise ValueError("Both players must have the same number of actions.")
        if (self.num_actions[0] < 2 or self.num_actions[1] < 2):
            raise ValueError("Both players must have at least two actions.")
        
        player_1_payoffs = np.array([[0, 1], [-1, -10]])
        player_2_payoffs = np.array([[0, -1], [1, -10]])

        payoff_matrices.append(player_1_payoffs)
        payoff_matrices.append(player_2_payoffs)
        return payoff_matrices
    
    def _generate_congestion_payoff_matrices(self):
        """
        Generate payoff matrices for the Congestion game.

        Returns:
        - list[np.ndarray]: A list of payoff matrices, one for each player.
        """
        payoff_matrices = []
        joint_action_space = tuple(self.num_actions)

        for _ in range(self.num_players):
            payoff_matrix = np.zeros(joint_action_space)
            for i in range(self.num_actions[0]):
                for j in range(self.num_actions[1]):
                    if i == j:
                        payoff_matrix[i, j] = 0
                    else:
                        payoff_matrix[i, j] = -1 if i == 0 else -2
            payoff_matrices.append(payoff_matrix)

        return payoff_matrices

    def _generate_custom_payoff_matrices(self, payoff_matrices):
        """
        Generate custom payoff matrices for each player.

        Parameters:
        - payoff_matrices (list[np.ndarray]): A list of payoff matrices, one for each player.

        Returns:
        - list[np.ndarray]: A list of payoff matrices, one for each player.
        """
        for i, matrix in enumerate(payoff_matrices):
            if matrix.shape != tuple(self.num_actions):
                raise ValueError(f"Payoff matrix for player {i} has incorrect shape.")
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
        repr_str = f"Game with {self.num_players} players\n"
        for i, matrix in enumerate(self.payoff_matrices):
            repr_str += f"Player {i+1}'s Payoff Matrix:\n{matrix}\n\n"
        return repr_str