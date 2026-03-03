import numpy as np
from auxFunctions import getState, load_obj, maxAction
import gymnasium as gym

EVAL_EPISODES = 100
RENDER = False

env = gym.make('MountainCar-v0', render_mode="human" if RENDER else None)
env._max_episode_steps = 200

Q = load_obj('SARSA-eval')

returns = []
successes = 0
steps_list = []

for episode in range(EVAL_EPISODES):
    observation, info = env.reset()
    state = getState(observation)

    done = False
    total_reward = 0.0
    steps = 0
    terminated_final = False

    while not done:
        action = maxAction(Q, state)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = getState(observation)
        total_reward += reward
        steps += 1
        terminated_final = terminated

    returns.append(total_reward)
    steps_list.append(steps)
    if terminated_final:
        successes += 1

env.close()

print(f"Q-learning evaluation over {EVAL_EPISODES} episodes:")
print(f"  Success rate: {successes / EVAL_EPISODES:.3f}")
print(f"  Mean return:  {np.mean(returns):.2f} (std {np.std(returns):.2f})")
print(f"  Mean steps:   {np.mean(steps_list):.1f}")