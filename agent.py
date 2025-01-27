from collections import defaultdict
import numpy as np


class Agent:
    """
    An AI agent that learns to play Blackjack using Monte Carlo methods.

    Attributes:
        Q (defaultdict): The action-value function mapping states to actions.
        policy (defaultdict): The current policy mapping states to actions.
        returns (defaultdict): A record of returns for state-action pairs.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for epsilon-greedy policy.
    """

    def __init__(self):
        """
        Initializes the agent with default parameters.
        """
        self.Q = defaultdict(self.default_Q)  # Replaces the lambda
        self.policy = defaultdict(int)  # 0: hit, 1: stick
        self.returns = defaultdict(list)
        self.gamma = 1.0
        self.epsilon = 0.1

    @staticmethod
    def default_Q():
        """
        Default initialization for Q-values.

        Returns:
            np.ndarray: A zero-initialized array of size 2 for actions [hit, stick].
        """
        return np.zeros(2)

    def generate_episode(self, env):
        """
        Generates a single episode by interacting with the environment.

        Args:
            env (Blackjack): The Blackjack environment.

        Returns:
            list: A list of (state, action, reward) tuples representing the episode.
        """
        episode = []
        state = env.reset()
        while True:
            # Dynamically adjust epsilon based on the Q-value difference
            q_diff = abs(self.Q[state][1] - self.Q[state][0])
            dynamic_epsilon = max(self.epsilon, self.epsilon + (1 - q_diff))  # More exploration if Q-values are close

            if np.random.rand() < dynamic_epsilon:  # Exploration
                action = np.random.choice(['hit', 'stick'])
            else:  # Exploitation
                action = 'hit' if self.policy[state] == 0 else 'stick'

            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def update_Q(self, episode):
        """
        Updates the action-value function using the generated episode.

        Args:
            episode (list): A list of (state, action, reward) tuples.
        """
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G  # Compute cumulative discounted reward

            # Avoid updating if the state-action pair has already appeared in this episode
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                action_idx = 0 if action == 'hit' else 1
                self.returns[(state, action_idx)].append(G)
                self.Q[state][action_idx] = np.mean(self.returns[(state, action_idx)])

    def improve_policy(self):
        """
        Updates the policy to be greedy with respect to the current action-value function.
        """
        for state in self.Q:
            self.policy[state] = np.argmax(self.Q[state])

    def train(self, env, episodes=10000):
        """
        Trains the agent using Monte Carlo methods.

        Args:
            env (Blackjack): The Blackjack environment.
            episodes (int): The number of training episodes.
        """
        for _ in range(episodes):
            episode = self.generate_episode(env)
            self.update_Q(episode)
            self.improve_policy()
            

class CardCountingAgent(Agent):
    """
    An AI agent that learns to play Blackjack using Monte Carlo methods
    and incorporates card counting for strategy adjustments.
    """

    def __init__(self):
        """
        Initializes the card counting agent with default parameters
        and adds card counting attributes.
        """
        super().__init__()
        self.running_count = 0  # Track the running count of cards
        self.deck_size = 52  # Default to a single deck

    def update_running_count(self, card):
        """
        Updates the running count based on the card dealt.

        Args:
            card (int): The value of the card dealt.
        """
        if card in [2, 3, 4, 5, 6]:
            self.running_count += 1
        elif card in [10, 11]:  # 11 is the Ace
            self.running_count -= 1

    def discretize_running_count(self):
        """
        Discretizes the running count into 'low', 'neutral', or 'high'.

        Returns:
            str: The discretized running count.
        """
        if self.running_count > 5:
            return 'high'
        elif self.running_count < -5:
            return 'low'
        else:
            return 'neutral'

    def generate_episode(self, env):
        """
        Generates an episode while incorporating card counting into decisions.

        Args:
            env (Blackjack): The Blackjack environment.

        Returns:
            list: A list of (state, action, reward) tuples representing the episode.
        """
        episode = []
        state = env.reset()

        # Reset running count if deck is reshuffled
        if env.round_counter % env.reshuffle_after == 0:
            self.running_count = 0

        # Update running count for initial cards
        for card in env.player_hand + [env.dealer_hand[0]]:
            self.update_running_count(card)

        while True:
            # Incorporate running count into state
            running_count_state = self.discretize_running_count()
            enhanced_state = (state[0], state[1], state[2], running_count_state)

            # Adjust exploration probability based on running count
            dynamic_epsilon = self.epsilon * (1 / (1 + abs(self.running_count)))
            if np.random.rand() < dynamic_epsilon:
                action = np.random.choice(['hit', 'stick'])  # Exploration
            else:
                action = 'hit' if self.policy[enhanced_state] == 0 else 'stick'  # Exploitation

            next_state, reward, done = env.step(action)

            # Update running count for new cards dealt
            for card in env.player_hand:
                self.update_running_count(card)

            episode.append((enhanced_state, action, reward))
            if done:
                break
            state = next_state
        return episode