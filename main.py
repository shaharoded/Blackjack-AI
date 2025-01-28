import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Local Code
from agent import Agent, CardCountingAgent  # Both regular and card-counting agents
from blackjack import Blackjack
from assistant import BlackjackRecommender


def train_model(
    env, episodes=50000, model_path="Trained Agents/blackjack_agent.pkl",
    agent_type="naive", acceptable_proficiency=0.5
):
    """
    Train a Blackjack agent using Monte Carlo methods, save the model, and plot win percentages.
    Automatically resumes training if a saved model exists.

    Args:
        env (Blackjack): The Blackjack environment.
        episodes (int): Number of training episodes.
        model_path (str): Path to save the trained model.
        agent_type (str): "naive" for the default agent, "card_counter" for card-counting agent.
        acceptable_proficiency (float): Early stop condition when reaching a certain winning ratio.
    """
    # Load or initialize the agent
    agent = Agent()  if agent_type == 'naive' else CardCountingAgent()
    agent.train(
        env,
        episodes=episodes,
        model_path=model_path,
        update_interval=1000,
        acceptable_proficiency=acceptable_proficiency
    )


def test_model(env, model_path, episodes=1000):
    """
    Test a trained Blackjack agent against the dealer.

    Args:
        env (Blackjack): The Blackjack environment.
        model_path (str): The path to load the model from.
        episodes (int): Number of test episodes.

    Returns:
        dict: Win, lose, and draw counts.
    """
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

        # If using the card counting agent, reset and update the running count
        if isinstance(agent, CardCountingAgent):
            agent.running_count = 0  # Ensure running count starts fresh per game
            for card in env.player_hand + [env.dealer_hand[0]]:
                agent.update_running_count(card)

        while True:
            # Determine the correct state format for the agent
            if isinstance(agent, CardCountingAgent):
                running_count_state = agent.discretize_running_count()
                enhanced_state = (state[0], state[1], state[2], state[3], running_count_state)
            else:
                enhanced_state = state

            # Choose action based on policy
            action = 'hit' if agent.policy.get(enhanced_state, 1) == 0 else 'stick'  # Default to stick

            # Step in the environment
            next_state, reward, done = env.step(action)

            # Update running count for new cards if using the counter agent
            if isinstance(agent, CardCountingAgent):
                for card in env.player_hand:
                    agent.update_running_count(card)

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
    
    # Plot agent's learned policy and visitation heatmap
    plot_policy_heatmap(agent)
    plot_visitation_heatmap(agent)
    
    return results

def plot_policy_heatmap(agent):
    """
    Plots a heatmap of the agent's policy with player totals on the X-axis 
    and dealer's visible cards on the Y-axis, averaged over additional state components.

    Args:
        agent (Agent): The trained Blackjack agent.
    """
    # Define ranges for player totals and dealer visible cards
    player_totals = range(4, 22)  # Player totals from 4 to 21
    dealer_cards = range(1, 11)  # Dealer's visible card (1 = Ace, 2-10)

    # Create a matrix to store the average policy actions
    policy_matrix = np.full((len(dealer_cards), len(player_totals)), np.nan)

    # Aggregate actions across states
    state_action_counts = defaultdict(lambda: [0, 0])  # {state: [sum(actions), count]}

    for state, action_probs in agent.Q.items():
        # Extract (player_total, dealer_card) from the state
        player_total, dealer_card, _ = state[:3]  # Ignore extra components
        if 4 <= player_total <= 21 and 1 <= dealer_card <= 10:
            state_action_counts[(player_total, dealer_card)][0] += np.argmax(action_probs)  # Sum action (0 or 1)
            state_action_counts[(player_total, dealer_card)][1] += 1  # Count occurrences

    # Compute average policy for each (player_total, dealer_card)
    for (player_total, dealer_card), (action_sum, count) in state_action_counts.items():
        avg_action = action_sum / count if count > 0 else np.nan
        policy_matrix[dealer_card - 1, player_total - 4] = avg_action  # Flip orientation

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(policy_matrix, cmap="coolwarm", interpolation="nearest", origin="lower")

    # Add labels and colorbar
    ax.set_xticks(np.arange(len(player_totals)))
    ax.set_xticklabels(player_totals)
    ax.set_yticks(np.arange(len(dealer_cards)))
    ax.set_yticklabels(dealer_cards)
    ax.set_xlabel("Player's Total")
    ax.set_ylabel("Dealer's Visible Card")
    ax.set_title("Agent Policy Heatmap (0=Hit, 1=Stick, Averaged)")

    fig.colorbar(cax, ax=ax, label="Action (0 = Hit, 1 = Stick)")
    plt.show()

def plot_visitation_heatmap(agent):
    """
    Plots heatmaps for state-action visitation frequencies side by side.

    Args:
        agent (Agent): The trained Blackjack agent.
    """
    player_totals = range(4, 22)  # Player totals from 4 to 21
    dealer_cards = range(1, 11)  # Dealer's visible card
    actions = ["Hit", "Stick"]
    color_maps = {"Hit": "Blues", "Stick": "Reds"}  # Define color maps for actions

    # Create visitation matrices
    heatmaps = {action: np.zeros((len(player_totals), len(dealer_cards))) for action in actions}

    # Aggregate visitation counts
    for state, visits in agent.visits.items():
        if isinstance(state, tuple):
            player_total, dealer_card, usable_ace = state[:3]
            if 4 <= player_total <= 21 and 1 <= dealer_card <= 10:
                heatmaps["Hit"][player_total - 4, dealer_card - 1] += visits[0]
                heatmaps["Stick"][player_total - 4, dealer_card - 1] += visits[1]

    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for idx, action in enumerate(actions):
        sns.heatmap(
            heatmaps[action],
            annot=False,
            xticklabels=dealer_cards,
            yticklabels=list(reversed(player_totals)),  # Ensure Y-axis is ascending
            cmap=color_maps[action],
            ax=axes[idx],
            cbar=True,
        )
        axes[idx].set_title(f"{action} Visitation Heatmap")
        axes[idx].set_xlabel("Dealer's Visible Card")
        if idx == 0:  # Add Y-axis label only to the first heatmap
            axes[idx].set_ylabel("Player's Total")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def get_assistant(model_path):
    """
    Initiate an assistant class using a trained model to recommend a play
    given a current game status.
    """
    # Load the agent
    try:
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
        print(f"✅ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}. Please train the agent first.")
        return None

    # Initialize the recommender
    assistant = BlackjackRecommender(agent)

    # Game Loop
    while True:
        print("\n[🎴] New Round. Enter the game state or type 'shuffle' to reset count, 'end' to stop.")
        
        # Get dealer's visible card
        dealer_card = input("Enter the dealer's visible card (1-10, where 1 = Ace): ").strip()
        if dealer_card.lower() == "shuffle":
            assistant.reset_running_count()
            continue
        if dealer_card.lower() == "end":
            print("🔚 Exiting the assistant.")
            break

        try:
            dealer_card = int(dealer_card)
            if dealer_card < 1 or dealer_card > 10:
                raise ValueError
        except ValueError:
            print("❌ Invalid input. Please enter a number between 1 and 10.")
            continue
        
        # Get player hand
        player_hand = input("Enter your hand as comma-separated values (e.g., 5,8 or 10,1): ").strip()
        if player_hand.lower() == "end":
            print("🔚 Exiting the assistant.")
            break

        try:
            player_hand = [int(card) for card in player_hand.split(",")]
        except ValueError:
            print("❌ Invalid input. Please enter valid card values (1-10).")
            continue

        # Update running count if needed
        assistant.update_running_count(player_hand + [dealer_card])

        # Start the round
        while True:
            # Get recommendation
            # Get recommendation & probability
            suggestion, win_probability = assistant.recommend_action(player_hand, dealer_card)
            print(f"\n🤖 Recommendation: **{suggestion.upper()}** (🃏 Estimated win chance: {win_probability:.2f}%)")

            if suggestion == "stick":
                print("✅ Ending turn with current hand.")
                break  # Exit the round, start a new one

            # If suggested to hit, ask for new card
            new_card = input("🔥 You drew a new card. Enter its value (or type 'end' to stop): ").strip()
            if new_card.lower() == "end":
                print("🔚 Exiting the assistant.")
                return

            try:
                new_card = int(new_card)
                if new_card < 1 or new_card > 10:
                    raise ValueError
                player_hand.append(new_card)  # Add new card to hand
                assistant.update_running_count([new_card])  # Update the count
            except ValueError:
                print("❌ Invalid input. Please enter a valid card value (1-10).")
                continue

            # Check if the player is busted
            player_value, _ = assistant.env.hand_value(player_hand)
            if player_value > 21:
                print("\n💥 You went **BUST**! Round over.")
                break  # Exit the round loop

    
if __name__ == "__main__":
    env = Blackjack()

    print("\n🃏 Blackjack AI Trainer 🃏")
    print("1. 🎓 Train Model")
    print("2. 📊 Test Model")
    print("3. 🤖 The Don speaks – Get your next move")
    print("4. ❌ Exit")
    choice = input("Select an option: ").strip()

    if choice == "1":
        episodes = int(input("Enter the number of training episodes: ").strip())
        agent_type = input("Select agent type ('naive' or 'counter'): ").strip().lower()

        # Determine the default model save path
        if agent_type == "naive":
            model_path = "Trained Agents/blackjack_agent.pkl"
        elif agent_type == "counter":
            model_path = "Trained Agents/blackjack_counter_agent.pkl"
        else:
            print("Invalid agent type. Defaulting to 'naive'.")
            agent_type = "naive"
            model_path = "Trained Agents/blackjack_agent.pkl"

        # Train the model
        train_model(env, episodes, model_path, agent_type)
    
    elif choice == "2":
        episodes = int(input("Enter the number of test episodes: ").strip())
        agent_type = input("Select agent type ('naive' or 'counter'): ").strip().lower()
        model_path = "Trained Agents/blackjack_agent.pkl" if agent_type == 'naive' else 'Trained Agents/blackjack_counter_agent.pkl'
        test_model(env, model_path, episodes)
    
    elif choice == "3":
        agent_type = input("Select assistnt type ('honest' or 'counter'): ").strip().lower()
        model_path = "Trained Agents/blackjack_agent.pkl" if agent_type == 'honest' else 'Trained Agents/blackjack_counter_agent.pkl' 
        get_assistant(model_path)
    else:
        print("OK bye Now.")