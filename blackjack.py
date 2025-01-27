import random

import random


class Blackjack:
    """
    A class to simulate a simplified game of Blackjack with a rule that the dealer
    must hit on a soft 17.
    """
    def __init__(self, reshuffle_after=5):
        """
        Initializes the Blackjack environment.

        Args:
            reshuffle_after (int): Number of rounds after which the deck is reshuffled.
        """
        self.reshuffle_after = reshuffle_after  # Number of rounds before reshuffling
        self.round_counter = 0  # Count how many rounds have been played
        self.deck = []  # Initialize the deck
        self.player_hand = []
        self.dealer_hand = []
        self.reshuffle_deck()  # Shuffle the deck to start

    def reset(self):
        """
        Resets the game and reshuffles the deck if the reshuffle threshold is reached.

        Returns:
            tuple: The initial state of the game as (player value, dealer visible card, usable ace).
        """
        if self.round_counter % self.reshuffle_after == 0:
            self.reshuffle_deck()
        self.round_counter += 1
        self.player_hand = []
        self.dealer_hand = []
        self.deal_initial_cards()
        return self.get_state()

    def reshuffle_deck(self):
        """
        Reshuffles the deck to reset it for a new cycle of rounds.
        """
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        random.shuffle(self.deck)

    def draw_card(self):
        """
        Draws a card from the deck.

        Returns:
            int: The value of the drawn card.

        Raises:
            RuntimeError: If the deck is empty (shouldn't happen with reshuffling logic).
        """
        if not self.deck:
            raise RuntimeError("The deck is empty, which should not happen with reshuffling logic.")
        return self.deck.pop()

    def deal_initial_cards(self):
        """
        Deals two cards to both the player and the dealer.
        """
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]

    def hand_value(self, hand):
        """
        Calculates the value of a given hand.

        Args:
            hand (list): A list of card values.

        Returns:
            tuple: Total hand value and whether the hand has a usable ace.
        """
        value = sum(hand)
        usable_ace = 1 in hand and value + 10 <= 21
        return value + 10 if usable_ace else value, usable_ace

    def is_bust(self, hand):
        """
        Checks if the hand value exceeds 21 (bust).

        Args:
            hand (list): A list of card values.

        Returns:
            bool: True if the hand is busted, False otherwise.
        """
        return self.hand_value(hand)[0] > 21

    def dealer_hits(self):
        """
        Determines whether the dealer should hit based on their current hand.

        Returns:
            bool: True if the dealer must hit, False otherwise.
        """
        value, usable_ace = self.hand_value(self.dealer_hand)
        return value < 17 or (value == 17 and usable_ace)

    def get_state(self):
        """
        Gets the current state of the game.

        Returns:
            tuple: The state as (player value, dealer visible card, usable ace).
        """
        player_value, usable_ace = self.hand_value(self.player_hand)
        dealer_card = self.dealer_hand[0]

        # Keep player value exact to preserve learning fidelity
        return (player_value, dealer_card, usable_ace)

    def step(self, action, is_dealer=False):
        """
        Takes an action ('hit' or 'stick') and progresses the game.

        Args:
            action (str): The player's action, either 'hit' or 'stick'.
            is_dealer (bool): If True, applies the action to the dealer's hand.

        Returns:
            tuple: The next state, reward, and whether the game is done.
        """
        if action == 'hit':
            # Dealer or player takes a card
            if is_dealer:
                self.dealer_hand.append(self.draw_card())
                if self.is_bust(self.dealer_hand):
                    # Dealer loses if they go over 21
                    return self.get_state(), -1, True
            else:
                self.player_hand.append(self.draw_card())
                if self.is_bust(self.player_hand):
                    # Player loses if they go over 21
                    return self.get_state(), -1, True

        elif action == 'stick':
            if not is_dealer:
                # Dealer takes cards until the hitting rule is satisfied
                while self.dealer_hits():
                    self.dealer_hand.append(self.draw_card())

                # Evaluate the final hands
                dealer_value, _ = self.hand_value(self.dealer_hand)
                player_value, _ = self.hand_value(self.player_hand)

                # Determine the outcome
                if dealer_value > 21 or player_value > dealer_value:
                    return self.get_state(), 1, True  # Player wins
                elif player_value == dealer_value:
                    return self.get_state(), 0, True  # Draw
                else:
                    return self.get_state(), -1, True  # Dealer wins

        # If action was 'hit' and the player hasn't busted, return the ongoing state
        return self.get_state(), 0, False