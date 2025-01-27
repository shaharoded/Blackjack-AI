# **Blackjack AI**

A simplified Blackjack simulation with AI agents that learn to play using **Monte Carlo methods**. The coding experiment includes a regular agent and a card-counting agent, both trained to optimize their strategies against a dealer, and was created as an intro to RL and also to humiliate my broter at Black-Jack.

## **Features**
- Simplified Blackjack rules:
  - No splitting or doubling down.
  - Dealer must hit on a soft 17.
- Two AI agents:
  - **Regular Agent**: Learns using Monte Carlo sampling.
  - **Card Counting Agent**: Incorporates running count to adjust its decisions dynamically.
- Deck reshuffles automatically after every 5 games.
- Performance tracking during training with win percentage visualization.
- Early stopping during training if the agent achieves a win rate of **98% or higher**.

---

## **Project Structure**
.
├── images/
│   ├── training_progress_plot.png
│   └── win_percentage_plot.png
├── blackjack.py
├── agent.py
├── main.py
├── README.md
└── Trained Agents        # Existing model files to load.
---

## **How to Run**

### **1. Clone the Repository**
```bash
git clone https://github.com/shaharoded/Blackjack-AI.git
cd Blackjack-AI
```

### **2. Install Dependencies, If Exists**

```bash
pip install -r requirements.txt
```

### **3. Run the Script**

```bash
python main.py
```

You'll be prompted withthe following menu:

```bash
🃏 Blackjack AI Trainer 🃏
1. 🎓 Train Model
2. 📊 Test Model
3. 🕹️ Play Against Agent
4. ❌ Exit
```

## Performance

- **Regular Agent:** Stabilizes at approximately 36% win rate after training for 100,000 episodes. This performance is expected due to the simplified rules and inherent house edge.

- **Card Counting Agent:** Shows a slight improvement (1-2%) by exploiting running count information but remains constrained by the simplified state representation.

### **Win Percentage Plot (Sample Training Run)**

#### **Training Progress Plot - Regular Agent**
![Training Progress Plot - Regular](Images/regular_agent.png)

#### **Training Progress Plot - Card Counter**
![Training Progress Plot - Card Counter](Images/card_counter_agent.png)


## Considerations
Agent Limitations:

 - The simplified rules (no doubling down, splitting) reduce the potential for strategy optimization.
 - Monte Carlo methods are computationally expensive and converge slowly for large state-action spaces.

## Environment Constraints:

State representation is simplified (player_value, dealer_card, usable_ace), which limits the agent's ability to generalize.

## Git Updates

```bash
git add .
git commit -m "commit message"
git branch -M main
git push -f origin main
```