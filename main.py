import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local Code
from agent import Agent, CardCountingAgent  # Both regular and card-counting agents
from blackjack import Blackjack


def train_model(env, episodes=50000, model_path="Trained Agents/blackjack_agent.pkl", agent_type="regular", update_interval=1000, acceptable_proficiency=0.8):
    """
    Train a Blackjack agent using Monte Carlo methods, save the model, and plot win percentages.
    Automatically resumes training if a saved model exists.

    Args:
        env (Blackjack): The Blackjack environment.
        episodes (int): Number of training episodes.
        model_path (str): Path to save the trained model.
        agent_type (str): "regular" for the default agent, "card_counter" for card-counting agent.
        update_interval (int): Frequency to log the win percentage during training.
        acceptable_proficiency (float): Early stop condition when reached a certain winning ratio.
    """
    # Check if a model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
    else:
        print(f"No existing model found. Initializing a new {agent_type} agent...")
        agent = Agent() if agent_type == "regular" else CardCountingAgent()

    win_percentages = []
    cumulative_wins = 0
    cumulative_games = 0
    decay_rate = 0.999
    min_epsilon = 0.01

    for episode in tqdm(range(1, episodes + 1), desc="Training Progress"):
        # Epsilon decay
        agent.epsilon = max(min_epsilon, agent.epsilon * decay_rate)

        # Play a single episode
        episode_data = []
        state = env.reset()
        while True:
            action = 'hit' if agent.policy[state] == 0 else 'stick'
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            if done:
                if reward == 1:
                    cumulative_wins += 1
                cumulative_games += 1
                break
            state = next_state

        # Update Q-values and policy
        agent.update_Q(episode_data)
        agent.improve_policy()

        # Log win percentage at intervals
        if episode % update_interval == 0:
            win_percentage = (cumulative_wins / cumulative_games)
            win_percentages.append(win_percentage * 100)

            # Early stopping condition
            if win_percentage >= acceptable_proficiency:
                print(f"Early stopping at episode {episode}: Win percentage = {win_percentage:.2f}%")
                break

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
    with open(model_path, 'wb') as f:
        pickle.dump(agent, f)
    print(f"Model saved to {model_path}")

    # Plot win percentage
    plt.figure(figsize=(10, 6))
    plt.plot(range(update_interval, len(win_percentages) * update_interval + 1, update_interval), win_percentages, marker='o')
    plt.title('Win Percentage During Training')
    plt.xlabel('Episodes')
    plt.ylabel('Win Percentage (%)')
    plt.grid(True)
    plt.show()


def test_model(env, agent_type="regular", episodes=1000):
    """
    Test a trained Blackjack agent against the dealer.

    Args:
        env (Blackjack): The Blackjack environment.
        agent_type (str): "regular" for the default agent, "card_counter" for card-counting agent.
        episodes (int): Number of test episodes.

    Returns:
        dict: Win, lose, and draw counts.
    """
    # Determine the default model path
    model_path = "Trained Agents/blackjack_agent.pkl" if agent_type == "regular" else "Trained Agents/blackjack_counter_agent.pkl"

    # Load the agent
    try:
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please train the agent first.")
        return

    results = {"win": 0, "lose": 0, "draw": 0}

    for _ in range(episodes):
        state = env.reset()
        while True:
            action = 'hit' if agent.policy[state] == 0 else 'stick'
            next_state, reward, done = env.step(action)
            if done:
                if reward == 1:
                    results["win"] += 1
                elif reward == -1:
                    results["lose"] += 1
                else:
                    results["draw"] += 1
                break
            state = next_state

    # Print and return the results
    print(f"Test Results: {results}")
    win_percentage = (results["win"] / episodes) * 100
    print(f"Win Percentage: {win_percentage:.2f}%")
    return results


if __name__ == "__main__":
    env = Blackjack()

    print("\n🃏 Blackjack AI Trainer 🃏")
    print("1. 🎓 Train Model")
    print("2. 📊 Test Model")
    print("3. ❌ Exit")
    choice = input("Select an option: ").strip()

    if choice == "1":
        episodes = int(input("Enter the number of training episodes: ").strip())
        agent_type = input("Select agent type ('regular' or 'card_counter'): ").strip().lower()

        # Determine the default model save path
        if agent_type == "regular":
            model_path = "Trained Agents/blackjack_agent.pkl"
        elif agent_type == "card_counter":
            model_path = "Trained Agents/blackjack_counter_agent.pkl"
        else:
            print("Invalid agent type. Defaulting to 'regular'.")
            agent_type = "regular"
            model_path = "Trained Agents/blackjack_agent.pkl"

        # Train the model
        train_model(env, episodes, model_path, agent_type)
    elif choice == "2":
        episodes = int(input("Enter the number of test episodes: ").strip())
        model_path = input("Enter the model path (default: 'blackjack_agent.pkl'): ").strip() or "blackjack_agent.pkl"
        test_model(env, model_path, episodes)
    else:
        print("Invalid option. Exiting.")