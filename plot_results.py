import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from auxFunctions import load_obj

# Load saved score arrays
q_scores = load_obj('./results/Q-Learning-scores')
sarsa_scores = load_obj('./results/SARSA-scores')

# Moving average function
def moving_average(data, window_size):
    return np.convolve(data, 
                       np.ones(window_size)/window_size, 
                       mode='valid')

window = 500  # smooth curve (since you have 50,000 episodes)

q_avg = moving_average(q_scores, window)
sarsa_avg = moving_average(sarsa_scores, window)

# Plot
plt.figure(figsize=(10,6))

plt.plot(q_avg, label="Q-learning", linewidth=2)
plt.plot(sarsa_avg, label="SARSA", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning vs SARSA (Moving Average)")
plt.legend()
plt.grid(True)

# Set y-axis ticks every 100
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(25))

plt.show()