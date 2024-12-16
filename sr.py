import numpy as np
from game import Game

class SwapRegretPlayer:
    def __init__(self, payoff_matrix, num_actions, player_index, eta=0.1):
        self.payoff_matrix = payoff_matrix
        self.num_actions = num_actions
        self.player_index = player_index
        self.eta = eta

        # Initialize a uniform probability distribution
        self.p = np.ones(num_actions) / num_actions
        
        # Initialize cumulative regrets
        self.cumulative_regrets = np.zeros(num_actions)

    def sample_action(self):
        """
        Sample an action based on the stationary distribution.
        """
        return np.random.choice(self.num_actions, p=self.p)

    def update_distributions(self, action_profile):
        """
        Update the player's distribution based on the observed action profile.
        """
        losses = np.zeros(self.num_actions)

        for i in range(self.num_actions):
            modified_profile = list(action_profile)
            modified_profile[self.player_index] = i
            modified_profile = tuple(modified_profile)

            alt_payoff = self.payoff_matrix[modified_profile]
            current_payoff = self.payoff_matrix[action_profile]

            losses[i] = alt_payoff - current_payoff

        self.cumulative_regrets += losses
        positive_regrets = np.maximum(self.cumulative_regrets, 0)

        # Update distribution using a softmax-like function
        exp_regrets = np.exp(self.eta * positive_regrets - np.max(self.eta * positive_regrets))  # Stabilized softmax
        self.p = exp_regrets / np.sum(exp_regrets)

    def get_distribution(self):
        """
        Return the current probability distribution over actions.
        """
        return self.p

    def __repr__(self):
        return f"SwapRegretPlayer(num_actions={self.num_actions}, eta={self.eta}, distribution={self.p})"

class SwapRegretSolver:
    def __init__(self, game: Game, T=10000, learning_rate=0.1):
        """
        Initialize the Swap Regret Solver.

        Parameters:
        - game (NormalFormGame): The game instance.
        - T (int): Number of iterations to run the regret minimization algorithm.
        - learning_rate (float): Step size for adjusting probabilities.
        """
        self.game = game
        self.T = T
        self.learning_rate = learning_rate
        
        # Create a SwapRegretPlayer for each player in the game
        self.players = [
            SwapRegretPlayer(
                payoff_matrix=game.payoff_matrices[i], 
                num_actions=game.num_actions[i], 
                player_index=i, 
                eta=learning_rate
            ) 
            for i in range(game.num_players)
        ]

    def get_name(self):
        return "Swap Regret"

    def solve(self):
        """
        Find an approximate correlated equilibrium using swap regret minimization.

        Returns:
        - dict: An approximate correlated equilibrium as a probability distribution.
        """
        # Run the swap regret algorithm for T iterations
        for i in range(self.T):
            # breakFlag = True
            # Sample actions for all players
            action_profile = tuple(
                player.sample_action() for player in self.players
            )
            
            # Update each player's distribution based on the observed action profile
            for player in self.players:
                # original_weights = player.get_distribution().copy()
                player.update_distributions(action_profile)
                # new_weights = player.get_distribution().copy()
                # if i < 100 or np.any(np.abs(original_weights - new_weights)) > 1e-6:
                #     if (np.abs(original_weights[0] - new_weights[0]) > 1e-6):
                #         print(f"Player {player.player_index} difference: {original_weights - new_weights}")
                #     if (np.abs(original_weights[1] - new_weights[1]) > 1e-6):
                #         print(f"Player {player.player_index} difference: {original_weights - new_weights}")
                #     breakFlag = False
                # else:
                #     print(f"Player {player.player_index} difference: {original_weights - new_weights}")
                #     print(f"Breaking at iteration {i}")

            # if breakFlag:
            #     break
        
        # Compute the final strategy profile
        strategy_profile = {
            i: self.players[i].get_distribution() 
            for i in range(len(self.players))
        }
        
        return strategy_profile