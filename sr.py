import numpy as np

class SwapRegretSolver:
    def __init__(self, game, T=10000, learning_rate=0.1):
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
        self.num_players = game.num_players
        self.num_actions = game.num_actions

    def get_name(self):
        return "Swap Regret"

    def solve(self):
        """
        Find an approximate correlated equilibrium using swap regret minimization.

        Returns:
        - dict: An approximate correlated equilibrium as a probability distribution.
        """
        action_probs = [np.ones(actions) / actions for actions in self.num_actions]
        empirical_distribution = np.zeros(tuple(self.num_actions))

        cumulative_rewards = [np.zeros((actions, actions)) for actions in self.num_actions]

        for t in range(1, self.T + 1):
            actions = tuple(np.random.choice(actions, p=probs) for probs, actions in zip(action_probs, self.num_actions))
            
            empirical_distribution[actions] += 1

            payoffs = self.game.get_payoff(actions)

            for player in range(self.num_players):
                for current_action in range(self.num_actions[player]):
                    for swap_action in range(self.num_actions[player]):
                        temp_action_profile = list(actions)
                        if temp_action_profile[player] == current_action:
                            temp_action_profile[player] = swap_action
                            temp_action_profile = tuple(temp_action_profile)
                            cumulative_rewards[player][current_action, swap_action] += payoffs[player]

            for player in range(self.num_players):
                expected_regret = np.zeros(self.num_actions[player])
                for current_action in range(self.num_actions[player]):
                    for swap_action in range(self.num_actions[player]):
                        expected_regret[current_action] += (
                            cumulative_rewards[player][current_action, swap_action]
                            - cumulative_rewards[player][current_action, current_action]
                        )
                action_probs[player] = np.exp(-self.learning_rate * expected_regret)
                action_probs[player] /= np.sum(action_probs[player])

        empirical_distribution /= np.sum(empirical_distribution)

        ce_distribution = {
            actions: empirical_distribution[actions]
            for actions in np.ndindex(empirical_distribution.shape)
            if empirical_distribution[actions] > 0
        }

        return ce_distribution
