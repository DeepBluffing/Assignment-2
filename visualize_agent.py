import gymnasium as gym
import numpy as np
import time
from frozen_lake_agent import train_agent

def watch_trained_agent():
    print("Training agent in the background...")
    # Get the student's trained Q-table
    q_table = train_agent()
    print("Training complete! Opening the visualizer...")

    # Initialize environment with human render mode
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    state, _ = env.reset()
    done = False

    # Add a slight delay before starting so the human can see the window
    time.sleep(1)

    while not done:
        # Exploit the best action from the Q-Table
        action = np.argmax(q_table[state])
        
        # Take the step
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Slow down the frames so the human eye can follow the elf
        time.sleep(0.5)

    if reward == 1.0:
        print("The Elf reached the treasure!")
    else:
        print("The Elf fell in a hole.")
        
    # Keep the window open for a second before closing
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    watch_trained_agent()
