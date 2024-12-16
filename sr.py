import numpy as np
from game import Game
import math

class SwapRegretPlayer:
    def __init__(self, payoff_matrix, num_actions, player_index, eta=0.1):
        """
        Initialize the SwapRegretPlayer.

        Parameters:
        - payoff_matrix (np.ndarray): The payoff matrix indexed by tuples representing actions of all players.
        - num_actions (int): The number of actions available to the player.
        - player_index (int): The ID/index of the player with respect to the payoff matrix
        - eta (float): Learning rate for the Multiplicative Weights algorithm.
        """
        self.num_actions = num_actions
        self.eta = eta
        self.player_index = player_index
        
        # Turn the payoff matrix into a loss matrix with a linear transformation to apply the MW setting
        max_value = np.max(payoff_matrix) 
        min_value = np.min(payoff_matrix)

        if max_value == min_value:
            normalized_matrix = np.zeros_like(payoff_matrix)
        else:
            normalized_matrix = (payoff_matrix - min_value) / (max_value - min_value)
        self.loss_matrix = 1 - normalized_matrix

        print(self.loss_matrix)
        # Initialize weights for k copies of the Multiplicative Weights algorithm
        # Each row corresponds to the weights of a particular action being replaced with another action.
        self.weights = np.ones((num_actions, num_actions))
        
        # Initialize our meta-distribution actions
        self.p = np.ones((num_actions)) / num_actions

    def sample_action(self):
        """
        Sample an action based on the stationary distribution of the weight matrix Q.

        Returns:
        - action (int): The sampled action index.
        """
        # Sample an action based on the stationary distribution
        return np.random.choice(self.num_actions, p=self.p)

    def update_distributions(self, action_profile):
        """
        Update the player's weights based on the observed action profile.

        Parameters:
        - action_profile (tuple): The actions chosen by all players in the game.
        """
        # Compute the loss vector l
        losses = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            # Replace this player's action in the action profile with action i
            modified_profile = list(action_profile)
            modified_profile[self.player_index] = i
            modified_profile = tuple(modified_profile)
            
            # Compute the loss for playing action i in the current profile
            losses[i] = self.loss_matrix[modified_profile]

        # Update weights for each copy of MW
        for j in range(self.num_actions):
            # Loss vector l scaled by p(j)
            scaled_losses = losses * self.p[j]

            # Update the jth row of weights using the scaled losses
            self.weights[j] -= (self.eta * scaled_losses)
        
        # Normalize the weights to ensure they form a probability distribution
        # Also clamp small negative weights at 0
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / self.weights.sum(axis=1, keepdims=True)
        
        # Compute the stationary distibution of our MW matrix
        self.p = self._stationary_distribution(self.weights);
    
    # Helper method to calculate the stationary distribution of our k MW copies
    def _stationary_distribution(self, Q):
        eigenvalues, eigenvectors = np.linalg.eig(Q.T)
        # Find the eigenvector corresponding to eigenvalue 1
        stationary = eigenvectors[:, np.isclose(eigenvalues, 1)]

        stationary = stationary[:, 0]  # Take the first (and only) eigenvector
        stationary = stationary / stationary.sum()  # Normalize to ensure sum = 1
        return stationary.real

    def __repr__(self):
        return (
            f"SwapRegretPlayer(num_actions={self.num_actions}, eta={self.eta}, "
            f"weights=\n{self.weights})"
        )


class SwapRegretSolver:
    def __init__(self, game: Game, T=None, learning_rate=None, epsilon=0.01):
        """
        Initialize the Swap Regret Solver.

        Parameters:
        - game (NormalFormGame): The game instance.
        - T (int): Number of iterations to run the regret minimization algorithm.
        - learning_rate (float): Step size for adjusting probabilities.
        """
        self.game = game
        self.T = T
        self.epsilon = epsilon
        self.num_players = game.num_players
        self.num_actions = game.num_actions

        if T:
            self.T = T
        else:
            self.T = int((4 * (np.max(self.num_actions) ** 2) * math.log(np.max(self.num_actions))) / (self.epsilon ** 2))
        
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = math.sqrt(math.log(np.max(self.num_actions))/self.T)
        self.players = [
            SwapRegretPlayer(game.get_payoff_matrix(player), game.num_actions[player], player, eta=self.learning_rate)
            for player in range(self.num_players)
        ]

    def get_name(self):
        return "Swap Regret"

    def solve(self):
        """
        Find an approximate correlated equilibrium using swap regret minimization.

        Returns:
        - dict: An approximate correlated equilibrium as a probability distribution.
        """
        empirical_distribution = {}
        action_counts = {}

        for _ in range(self.T):
            # Sample actions for each player
            action_profile = tuple(player.sample_action() for player in self.players)

            # Update action counts
            if action_profile not in action_counts:
                action_counts[action_profile] = 0
            action_counts[action_profile] += 1

            # Update each player with the joint action profile
            for player in self.players:
                player.update_distributions(action_profile)

        # Normalize action counts to form the empirical distribution
        for action_profile, count in action_counts.items():
            empirical_distribution[action_profile] = count / self.T

        return empirical_distribution
