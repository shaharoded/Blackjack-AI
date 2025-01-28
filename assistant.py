# Local Code
from agent import *
from blackjack import *


class BlackjackRecommender:
    """
    A class to assist with Blackjack decisions using a trained agent.
    It allows the user to input their hand, dealer's visible card,
    and receive a recommendation based on the agent's learned policy.
    """

    def __init__(self, agent):
        self.agent = agent
        self.env = Blackjack()
        self.running_count = 0 if isinstance(agent, CardCountingAgent) else None  # Only for card counters

    def reset_running_count(self):
        """Resets the running count for card counting agents."""
        if self.running_count is not None:
            self.running_count = 0
            print("[🔄] Running count reset.")

    def update_running_count(self, cards):
        """Updates the running count when new cards are seen (for card counters)."""
        if self.running_count is not None:
            for card in cards:
                self.agent.update_running_count(card)
                self.running_count = self.agent.running_count

    # def recommend_action(self, player_hand, dealer_card):
    #     """
    #     Recommends an action ('hit' or 'stick') based on the current game state.
        
    #     Args:
    #         player_hand (list): The player's hand as a list of integers.
    #         dealer_card (int): The dealer's visible card.

    #     Returns:
    #         str: Recommended action ('hit' or 'stick').
    #     """
    #     player_value, usable_ace = self.env.hand_value(player_hand)  # Static method for hand value
    #     state = (player_value, dealer_card, usable_ace, len(player_hand))

    #     if self.running_count is not None:  # If using a card counter
    #         running_count_state = self.agent.discretize_running_count()
    #         state = (*state, running_count_state)  # Include running count

    #     # Get the recommended action from the trained policy
    #     action = 'hit' if self.agent.policy.get(state, 1) == 0 else 'stick'
    #     return action
    
    def recommend_action(self, player_hand, dealer_card):
        """
        Recommends an action ('hit' or 'stick') based on the current game state and
        displays the estimated probability of winning.

        Args:
            player_hand (list): The player's hand as a list of integers.
            dealer_card (int): The dealer's visible card.

        Returns:
            tuple: (Recommended action, estimated win probability).
        """
        player_value, usable_ace = self.env.hand_value(player_hand)
        state = (player_value, dealer_card, usable_ace, len(player_hand))

        if self.running_count is not None:  # If using a card counter
            running_count_state = self.agent.discretize_running_count()
            state = (*state, running_count_state)  # Include running count

        # Get Q-values
        q_values = self.agent.Q.get(state, [0, 0])  # Default if unseen state

        # Compute estimated win probability as max(Q[state])
        win_probability = ((max(q_values) + 1) / 2) * 100  # Normalize to 0-100%

        # Choose best action
        action = 'hit' if q_values[0] > q_values[1] else 'stick'

        return action, win_probability