import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from auxFunctions import load_obj

folder_name = ''

# Load saved score arrays
q_scores = load_obj(f'./results/{folder_name}/Q-Learning-scores')
sarsa_scores = load_obj(f'./results/{folder_name}/SARSA-scores')

# Moving average function
def moving_average(data, window_size):
    return np.convolve(data, 
                       np.ones(window_size)/window_size, 
                       mode='valid')

window = 500  # smooth curve

q_avg = moving_average(q_scores, window)
sarsa_avg = moving_average(sarsa_scores, window)

# Plot
plt.figure(figsize=(12,8))

plt.plot(q_avg, label="Q-learning", linewidth=2)
plt.plot(sarsa_avg, label="SARSA", linewidth=2)

plt.xticks(rotation=45)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Total Reward", fontsize=14)
plt.title("Q-learning vs SARSA", fontsize=16)
plt.legend()
plt.grid(True)

ax = plt.gca()

ax.yaxis.set_ticks_position('both')   # ticks on left and right
ax.tick_params(axis='y', labelright=True)

# Set y-axis ticks every 100
ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

for i, line in enumerate(ax.get_ygridlines()):
    line.set_color('#4A90E2' if i % 2 == 0 else '#A7C7E7')
    line.set_linewidth(0.8)

plt.show()