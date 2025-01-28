import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt



class Agent:
    """
    An AI agent that learns to play Blackjack using Monte Carlo methods.
    NOTE: The only parameter that can be optimized here is epsilon.

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
        self.Q = defaultdict(self.default_Q)  # Action-value function
        self.policy = defaultdict(int)  # 0: hit, 1: stick
        self.returns = defaultdict(list)  # Track returns for state-action pairs
        self.visits = defaultdict(self.default_visits)  # Track visits for actions [hit, stick]
        self.gamma = 1.0  # Discount factor, future rewards as important as current ones.
        self.epsilon = 0.5  # High Exploration rate as it reduces through episodes.

    @staticmethod
    def default_Q():
        """
        Default initialization for Q-values.
        Introduces small random values to reduce bias toward equal exploration.
        
        Returns:
            np.ndarray: A randomly initialized array for actions [hit, stick].
        """
        return np.random.uniform(low=0.01, high=0.1, size=2)
    
    @staticmethod
    def default_visits():
        """
        Default initialization for visit counts.
        """
        return [0, 0]  # Initialize visit counts for 'hit' and 'stick'
    
    def update_visits(self, state, action):
        action_idx = 0 if action == 'hit' else 1
        self.visits[state][action_idx] += 1

    def generate_episode(self, env):
        """
        Generates a single episode (game) by interacting with the environment.

        Args:
            env (Blackjack): The Blackjack environment.

        Returns:
            list: A list of (state, action, reward) tuples representing the episode.
        """
        episode = []
        decay_factor = 0.999  # Decay rate per episode
        state = env.reset()
        while True:          
            # Apply decay-based epsilon adjustment
            dynamic_epsilon = max(0.01, self.epsilon * (decay_factor ** env.round_counter))  # Ensure epsilon doesn't go below 0.01

            # Exploration vs exploitation
            if np.random.rand() < dynamic_epsilon:
                action = np.random.choice(['hit', 'stick'])
            else:
                action = 'hit' if self.policy[state] == 0 else 'stick'
                
            # Track visits
            action_idx = 0 if action == 'hit' else 1
            aggregated_state = state[:3]  # Only use (player_total, dealer_card, usable_ace)
            self.visits[aggregated_state][action_idx] += 1

            # Take action in the environment
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def update_Q(self, episode):
        """
        Updates the action-value function using the generated episode.
        Updates are performed using the Monre-Carlo method, using a complete episode
        to calculate Q.

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
            
    def train(
        self, env, episodes=100000, model_path=None,
        update_interval=5000, acceptable_proficiency=0.6,
        eval_games=1000  # Number of deterministic games for evaluation
    ):
        """
        Train the agent using Monte Carlo methods, but only log performance based on deterministic plays.

        Args:
            env (Blackjack): The Blackjack environment.
            episodes (int): Number of training episodes.
            model_path (str): Path to save the trained model.
            update_interval (int): Frequency to log the deterministic win percentage during training.
            acceptable_proficiency (float): Early stop condition when reaching a certain winning ratio.
            eval_games (int): Number of evaluation games played at each interval.
        """
        if model_path and os.path.exists(model_path):
            print(f"[Training Status]: Loading existing model from {model_path}...")
            with open(model_path, 'rb') as f:
                loaded_agent = pickle.load(f)
                self.__dict__.update(loaded_agent.__dict__)  # Copy state from loaded agent
        else:
            print("[Training Status]: No existing model found. Training a new agent...")

        win_percentages = []
        cumulative_wins = 0
        cumulative_games = 0

        for episode in tqdm(range(1, episodes + 1), desc="Training Progress"):
            episode_data = self.generate_episode(env)  # Train with exploration

            # Update Q-values and policy
            self.update_Q(episode_data)
            self.improve_policy()

            # Log training statistics
            reward = episode_data[-1][-1]  # Last reward in the episode
            if reward == 1:
                cumulative_wins += 1
            cumulative_games += 1

            # Perform deterministic evaluation every update_interval episodes
            if episode % update_interval == 0:
                eval_results = {"win": 0, "lose": 0, "draw": 0}

                for _ in range(eval_games):  # Evaluate using a fixed number of deterministic games
                    state = env.reset()

                    # If using the card counting agent, reset and update the running count
                    if isinstance(self, CardCountingAgent):
                        self.running_count = 0
                        for card in env.player_hand + [env.dealer_hand[0]]:
                            self.update_running_count(card)

                    while True:
                        # Determine the correct state format for the agent
                        if isinstance(self, CardCountingAgent):
                            running_count_state = self.discretize_running_count()
                            enhanced_state = (state[0], state[1], state[2], state[3], running_count_state)
                        else:
                            enhanced_state = state

                        # Choose action based on policy (deterministic, no exploration)
                        action = 'hit' if self.policy.get(enhanced_state, 1) == 0 else 'stick'

                        # Step in the environment
                        next_state, reward, done = env.step(action)

                        # Update running count for new cards if using the counter agent
                        if isinstance(self, CardCountingAgent):
                            for card in env.player_hand:
                                self.update_running_count(card)

                        if done:
                            if reward == 1:
                                eval_results["win"] += 1
                            elif reward == -1:
                                eval_results["lose"] += 1
                            else:
                                eval_results["draw"] += 1
                            break

                        state = next_state

                # Compute deterministic win rate
                eval_win_percentage = (eval_results["win"] / eval_games)
                win_percentages.append(eval_win_percentage * 100)

                # Early stopping condition
                if len(win_percentages) > 5 and np.mean(win_percentages[-5:]) >= acceptable_proficiency*100:
                    print(f"Early stopping at episode {episode}: Avg win percentage = {np.mean(win_percentages[-5:]):.2f}%")
                    break
            
        # Save model
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved to {model_path}")

        # Plot win percentage
        plt.figure(figsize=(10, 6))
        plt.plot(range(update_interval, len(win_percentages) * update_interval + 1, update_interval), win_percentages, marker='o')
        plt.title('Win Percentage During Training')
        plt.xlabel('Episodes')
        plt.ylabel('Win Percentage (%)')
        plt.grid(True)
        plt.show()


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
        Updates the running count based on the card dealt using Hi-Low method.

        Args:
            card (int): The value of the card dealt.
        """
        if card in [2, 3, 4, 5, 6]:
            self.running_count += 1
        elif card in [10, 1]:  # High cards (10, Jack, Queen, King, Ace)
            self.running_count -= 1

    def discretize_running_count(self, thresh=5):
        """
        Discretizes the running count into 'low', 'neutral', or 'high'.
        Discretization based on thresh.

        Returns:
            str: The discretized running count.
        """
        if self.running_count >= thresh:
            return 'high'
        elif self.running_count <= -thresh:
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
            enhanced_state = (state[0], state[1], state[2], state[3], running_count_state)

            # Reduce exploration if the deck is hot\cold.
            dynamic_epsilon = self.epsilon * (1 / (1 + abs(self.running_count)))
            if np.random.rand() < dynamic_epsilon:
                action = np.random.choice(['hit', 'stick'])  # Exploration
            else:
                action = 'hit' if self.policy[enhanced_state] == 0 else 'stick'  # Exploitation

            next_state, reward, done = env.step(action)

            # Reward shaping based on running count
            if running_count_state == 'low' and abs(env.hand_value(env.player_hand)[0] - 21) < 3:
                hand_value_diff = abs(env.hand_value(env.player_hand)[0] - 21)
                if hand_value_diff == 0:
                    reward += 0.1 * 1  # Maximum reward for exactly hitting 21
                else:
                    reward += 0.1 * (1 / hand_value_diff)  # Reward for drawing near 21 when the count is low
            elif running_count_state == 'high' and action == 'stick' and env.hand_value(env.player_hand)[0] >= 17:
                reward += 0.1 * ((env.hand_value(env.player_hand)[0] - 17) / 4)  # Scales between 0.1 (at 17) to 1.0 (at 21)

            # Update running count for new cards dealt
            for card in env.player_hand:
                self.update_running_count(card)
            
            # Track visits
            action_idx = 0 if action == 'hit' else 1
            aggregated_state = state[:3]  # Only use (player_total, dealer_card, usable_ace)
            self.visits[aggregated_state][action_idx] += 1

            episode.append((enhanced_state, action, reward))
            if done:
                break
            state = next_state
        return episode