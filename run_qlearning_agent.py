from auxFunctions import getState, load_obj, maxAction
import gymnasium as gym

env = gym.make('MountainCar-v0', render_mode="human")
env._max_episode_steps = 200

# Load Q-table trained with Q-learning
Q = load_obj('pre-trained-Q-Learning')

# Run 10 episodes
for episode in range(10):
    done = False
    observation, info = env.reset()
    state = getState(observation)
    # While the car don't reach the goal or number of steps < 200
    while not done:
        env.render()
        print(observation)
        # Take the best action for that state given trained values
        action = maxAction(Q, state)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Go to next state
        state = getState(observation)

env.close()