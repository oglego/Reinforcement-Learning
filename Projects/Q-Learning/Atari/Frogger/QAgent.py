#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-learning agent to play Frogger -

The agent will need the following parameters:
    
    num_actions: int,                 # Number of possible actions the agent can take
    learning_rate: float,             # Learning rate for the Q-learning algorithm
    initial_epsilon: float,           # Initial exploration probability (epsilon-greedy strategy)
    epsilon_decay: float,             # Rate at which epsilon decays over time
    final_epsilon: float,             # Final minimum exploration probability
    discount_factor: float = 0.09,    # Discount factor for future rewards  

The agent class will also include the below functions:
    
    get_action:
        
        Returns the best action with probability (1 - epsilon) otherwise a random
        action with probability epsilon to ensure exploration.
        
    update:
        
        Updates the Q-value of an action based on the observed reward and next state.
        
    decay_epsilon:
        
        Decay the exploration probability (epsilon) over time.
        
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

class Agent:
    def __init__(
            self,
            num_actions: int,                 # Number of possible actions the agent can take
            learning_rate: float,             # Learning rate for the Q-learning algorithm
            initial_epsilon: float,           # Initial exploration probability (epsilon-greedy strategy)
            epsilon_decay: float,             # Rate at which epsilon decays over time
            final_epsilon: float,             # Final minimum exploration probability
            discount_factor: float = .09      # Discount factor for future rewards - the larger the value
                                              # the more we value future rewards 
                                              
    ):
            """
            Initialize a Reinforcement Learning agent with the following parameters:
        
            Parameters:
                        
            learning_rate (float): The learning rate.
            initial_epsilon (float): The initial epsilon value.
            epsilon_decay (float): The rate at which epsilon decays.
            final_epsilon (float): The final epsilon value.
            discount_factor (float): The discount factor used in Q-value computation.
            """
            # Initialize Q-values for each state-action pair using defaultdict
            self.q_values = defaultdict(lambda: np.zeros(num_actions))
        
            # Store the learning rate and the discount factor
            self.lr = learning_rate
            self.discount_factor = discount_factor
        
            # Store the initial and final exploration probabilities
            self.epsilon = initial_epsilon
            self.epsilon_decay = epsilon_decay
            self.final_epsilon = final_epsilon
        
            # Create a list to store the training errors during the agent's learning process
            self.training_error = []
            
    def get_action(self, obs: tuple[int, int, bool], num_actions: int) -> int:
        """
        Returns the best action with probability (1 - epsilon) otherwise a random
        action with probability epsilon to ensure exploration.
        
        From the documentation on
        
        https://gymnasium.farama.org/environments/atari/frogger/
        
        we can see that the obs type is a numpy array so we are going to use the
        flatten function below and create a tuple of the observations.
        
        Args:
            obs (tuple): A tuple representing the current observation or state, 
                typically containing integers and a boolean value.
            num_actions (int): The total number of possible actions.
        
        Returns:
            int: The selected action, either chosen greedily (exploit) with a 
            probability of (1 - epsilon) or randomly with a probability of epsilon.
        """
        # Create a tuple of the values from flattening observations
        obs_tuple = tuple(obs.flatten())
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return np.random.choice(num_actions)
        
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs_tuple]))

    def update(
        self,
        obs: tuple[int, int, bool],         # Current observation or state
        action: int,                        # Action taken
        reward: float,                      # Reward received
        terminated: bool,                   # Whether the episode terminated after this step
        next_obs: tuple[int, int, bool],    # Next observation or state after taking the action
    ):
        """
        Updates the Q-value of an action based on the observed reward and next state.
        
        From the documentation on
        
        https://gymnasium.farama.org/environments/atari/frogger/
        
        we can see that the obs type is a numpy array so we are going to use the
        flatten function below and create a tuple of the observations.
        
        Args:
            obs (tuple): Current observation or state, typically containing integers and a boolean value.
            action (int): Action taken by the agent.
            reward (float): Reward received after taking the action.
            terminated (bool): Whether the episode terminated after this step.
            next_obs (tuple): Next observation or state after taking the action.
        
        Returns:
            None
        """
        # Create a tuple of observations by flattening next_obs
        next_obs_tuple = tuple(next_obs.flatten())
        # Calculate the estimated future Q-value based on whether the episode terminated
        future_q_value = (not terminated) * np.max(self.q_values[next_obs_tuple])
        
        # Calculate the temporal difference (error) between the estimated and observed Q-values
        obs_tuple = tuple(obs.flatten())
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs_tuple][action]
        )

        # Update the Q-value for the observed state-action pair using the learning rate
        obs_tuple = tuple(obs.flatten())
        self.q_values[obs_tuple][action] = (
            self.q_values[obs_tuple][action] + self.lr * temporal_difference
        )
        
        # Record the temporal difference (error) for monitoring training progress
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """
        Decay the exploration probability (epsilon) over time.
    
        This function updates the exploration probability epsilon by reducing it according
        to the epsilon decay rate, but ensuring that it does not go below the final epsilon
        value.
    
        Args:
            None
    
        Returns:
            None
        """
        # Update epsilon by subtracting the epsilon decay rate, but ensure it doesn't go below final_epsilon
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        

def train(env, n_episodes, agent, num_actions):
    """
    Train the specified agent in the given environment for a specified number of episodes.
    
    Parameters:
        env (gym.Env): The Gym environment to train the agent in.
        n_episodes (int): The number of episodes to train the agent for.
        agent (object): The agent to be trained, implementing the necessary methods.
        num_actions (int): The number of possible actions the agent can take.
    
    Returns:
        gym.Env: The environment with episode statistics recorded.
    """
    # Episode Statistics
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    
    # Wrap the environment with RecordVideo, specifying the output path and video name
    env = RecordVideo(env, 'video', episode_trigger=lambda epi: epi == n_episodes-1)
    
    # Create a list to keep track of the rewards so that we can plot them later
    rewards = []
    
    for episode in tqdm(range(n_episodes)):
        # Store observations/information about the environment after resetting the environment
        obs, info = env.reset()
        # Set done to false for the while loop
        done = False
        # Variable for storing total rewards
        cumulative_reward = 0
        while not done:
            # Get agent action
            action = agent.get_action(obs, num_actions)
            # Determine below variables based on action
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Update total reward
            cumulative_reward += reward
            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)
    
            # update if the environment is done and the current obs
            done = terminated or truncated
            # Set obs to next observation
            obs = next_obs    
        # Decay epsilon
        agent.decay_epsilon()
        print(f"Episode {episode + 1}, Total Reward: {cumulative_reward}")
        # Add rewards to our list so we can plot it later
        rewards.append(cumulative_reward)
    return env, rewards
        
def visualize_training(env, agent):
    """
    Function to visualize the agents Episode rewards, lengths, and the training error.
    
    The code below was obtained from the below URL:
        
    https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
    """
    # Set the rolling length for computing moving averages
    rolling_length = 500
    # Plot Training Error
    plt.title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    plt.plot(range(len(training_error_moving_average)), training_error_moving_average)
    
    # Adjust layout for better visual
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
def visualize_rewards(rewards):
    """
    Function to visualize the scores obtained in each episode 
    that the agent completes.
    """
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Score")

def main():
    # Path for ffmpeg
    os.environ['IMAGEIO_FFMPEG_EXE'] = "/opt/homebrew/bin/ffmpeg"
    
    # Create environment
    env = gym.make("ALE/Frogger-ram-v5", obs_type="ram", render_mode="rgb_array", difficulty=0)
    
    # Hyperparameters
    num_actions = env.action_space.n
    # Larger learning rate adjusts aggressively and may not converge
    lr = .001              
    n_episodes = 16
    # Larger epsilons encourage more exploring while training
    start_epsilon = 1.0
    final_epsilon = 0.00001
    
    # Reduce the agent's exploration over time 
    # We use linear decay
    epsilon_decay = (start_epsilon-final_epsilon) / n_episodes 
    
    # Create an Agent object with the above hyperparameters
    agent = Agent(num_actions = num_actions,
                  learning_rate = lr,
                  initial_epsilon = start_epsilon,
                  epsilon_decay = epsilon_decay,
                  final_epsilon = final_epsilon)

    
    # Start training
    env, rewards = train(env, n_episodes, agent, num_actions)
    
    # Visualize training
    visualize_training(env, agent)
    
    visualize_rewards(rewards)
    
    # Close environment
    env.close()
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        