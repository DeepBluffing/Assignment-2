import gymnasium as gym
import numpy as np
import sys
from frozen_lake_agent import train_agent

def test_frozen_lake():
    print("Running Autograder for Frozen Lake...")
    
    # 1. Attempt to run the student's training loop
    try:
        q_table = train_agent()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    print("Training complete. Testing the agent's policy...")
    
    # 2. Setup the test environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    state, _ = env.reset()
    
    # 3. Test the agent (Max 20 steps to prevent infinite loops)
    reward = 0
    for step in range(20):
        # Always exploit the best learned action during testing
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    # 4. Evaluate the final outcome
    if reward == 1.0:
        print("Success! Agent safely navigated the ice and found the treasure.")
        sys.exit(0) 
    else:
        print("Failure! Agent fell in a hole or wandered aimlessly.")
        sys.exit(1)

if __name__ == "__main__":
    test_frozen_lake()
