write me an outline for robotics project final presentation.
The project is a Reinforcement learning based tradebot that is trained and tested using custom backtesting simulation
Raw stock data were fetchted from Alpaca-trade, and 2020-1~2022-12 data isused to train the model, and 2023-1~2023-7 is used for testing. Top50 volumn stock from SNP500 were used.
Since unlike the RNN algorithm which directly track hidden information about previous states, DQN algorithm doesn't have access to such hidden information when making new decision; So stock indicators are used to provide the model with Price trends, Price channels, Oscillation indicator, Stop and reverse, Volume-based Moving averages, Price transforms, Price characteristics.
Also, macro economic data from FERN were fetched and used to support hte model in decision making; including Federal Funds Rate,  University of Michigan: Consumer Sentiment, onsumer Price Index for All Urban Consumers: All Items in U.S. City Average, Retail Sales, Industrial Capacity Utilization: Total Index).

The model used is a DQN model, which differs from basic RL algoirthms (such as ...) in that it can adapt to continous state space, which is what is needed here.

The action space for the model consists of 3 discrete actions: hold, buy, sell.

The provided code defines a **Deep Q-Network (DQN) agent** tailored for trading tasks. Here's a summary:
More details about the mode:
### Components:
1. **Trading Network (`TradingNetwork`):**
   - A neural network built with PyTorch.
   - Input: `state_size` (number of features in the environment's state).
   - Hidden layers: Two fully connected layers with ReLU activation, each having a configurable size (`hidden_size`, default: 64).
   - Output: 3 actions—hold, buy, sell (linear layer with 3 neurons).

2. **DQN Agent (`DQNAgent`):**
   - Implements the DQN algorithm with enhancements like experience replay and a target network.

### Key Features:
1. **Exploration vs Exploitation:**
   - Uses an epsilon-greedy policy for exploration.
   - Epsilon starts at `epsilon_start` (1.0) and decays to `epsilon_end` (0.01) with a decay rate (`epsilon_decay`, 0.995).

2. **Experience Replay:**
   - Experiences (state, action, reward, next state, done) are stored in a replay buffer (`deque`) with a size of `memory_size` (10,000 by default).
   - During training, random batches of experiences are sampled to decorrelate data.

3. **Target Network:**
   - A separate, periodically updated target network stabilizes training by holding frozen parameters for future Q-value calculations.

4. **Training:**
   - Uses the Bellman equation to compute target Q-values:
     \[
     Q_{\text{target}} = \text{reward} + \gamma \cdot \max(Q_{\text{next}})
     \]
   - Loss: Mean Squared Error (MSE) between predicted and target Q-values.
   - Optimizer: Adam with a learning rate of \(1 \times 10^{-5}\).

5. **Save and Load:**
   - Can save and restore the policy network, target network, optimizer state, and epsilon value to/from disk.

6. **Actions:**
   - The agent can act based on the current policy (`act`) or perform a training step (`train_step`).

### Usage Flow:
1. Initialize the `DQNAgent` with environment parameters (e.g., `state_size`).
2. Train the agent using experiences (state transitions) collected during interactions with the environment.
3. Periodically update the target network (`update_target_network`) to align with the policy network.
4. Save or load models for continuity.

This model is ideal for tasks where the agent learns to make sequential decisions (like trading) through trial and error, using reinforcement learning principles.




The normalization of data is done by examining the nature of each feature, methods used include min-max normalization, mean normalization, standard deviation normalization, and some other custome methods.

The model doesn't significantly outperform the background strategy in the testing period, but considering the trade fee & failed trade that is incorporated in the simulation, it is not too bad.






Example flow of the slide:

### Robotics Project Final Presentation Outline: **Reinforcement Learning-Based Tradebot**

---

#### 1. **Introduction**  
   - **Project Overview**
     - Motivation for trying a reinforcement learning-based tradebot: previously tried recurrent neural network based model, want to perform comparison 
   - **Key Objectives**  
     - Develop a tradebot using Deep Q-Learning.  
     - Train and test the bot using custom backtesting simulations.  
   - **Why Reinforcement Learning?**  
     - Advantages of RL in sequential decision-making.  
     - Specific relevance to trading tasks.

---

#### 2. **Data Preparation**  
   - **Data Sources**  
     - Raw stock data (daily) fetched from Alpaca-trade (2020-2022 for training, 2023 first half year for testing).  
     - Train on all of Top 50 volume stocks from S&P500.
     - Macro-economic data fetched from FRED (Federal Reserve Economic Database).  
   - **Feature Engineering**  
     - Stock indicators: price trends, channels, oscillation indicators, volume-based moving averages, etc.  
     - Macro-economic features: retail sales, CPI, industrial capacity utilization.  
   - **Data Normalization**  
     - Techniques applied (min-max scaling, mean normalization, etc.).  
     - Challenges in handling different scales and data types.  

---

#### 3. **Reinforcement Learning Framework**  
   - **Model Architecture**  
     - Overview of the DQN model.  
     - Adaptations for continuous state space.  
     - Input: stock indicators, macro-economic data, position in cash-position ratio and average price for getting the poisition), output (action space: hold, buy, sell).  
   - **Components**  
     - Policy Network: Structure and layers.  
     - Target Network: Purpose and how it improves stability.  
   - **Key Enhancements**  
     - Experience Replay: How memory improves training.  
     - Epsilon-Greedy Exploration: Balancing exploration and exploitation.  

---

#### 4. **Training and Testing Setup**  
   - **Custom Backtesting Simulator**  
     - How trades are simulated.  
     - Integration of trading fees and failed trades.  
   - **Training Details**  
     - Bellman equation and Q-value updates.  
     - Loss function and optimizer (Mean Squared Error, Adam optimizer).  
     - Exploration decay strategy.  
   - **Testing Setup**  
     - Performance on unseen 2023 data.  
     - Benchmarking against baseline strategies (e.g., buy-and-hold).  

---

#### 5. **Results and Analysis**  
   - **Performance Metrics**  
     - Profitability, risk-adjusted returns (e.g., Sharpe ratio).  
     - Comparison with background strategies.  
   - **Challenges and Observations**  
     - Limitations due to lack of hidden state information (compared to RNNs).  
     - Impact of feature engineering on decision quality.  

---

#### 6. **Insights and Takeaways**  
   - **Key Findings**  
     - Model strengths and weaknesses.  
     - Role of macro-economic data in improving decisions.  
   - **Comparison with RNN-based Models**  
     - Advantages and limitations of DQN for this application.  
   - **Lessons Learned**  
     - Data quality and preprocessing challenges.  
     - Importance of simulation fidelity.  

---

#### 7. **Conclusion**  
   - **Project Summary**  
     - How well the objectives were met.  
     - Final performance of the tradebot.  
   - **Future Work**  
     - Exploring RNNs or other algorithms for hidden state tracking.  
     - Improving feature selection and engineering.  
     - Real-time deployment considerations.  

---

#### 8. **Demonstration and Q&A**  
   - **Live Demo**  
     - Trading bot simulation on test data.  
     - Visualization of trades and performance.  
   - **Questions and Discussion**  
     - Address audience queries and discuss potential improvements.  

---

This outline provides a clear and logical structure for presenting your project, focusing on both technical details and overall findings.