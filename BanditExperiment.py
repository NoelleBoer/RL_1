#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    
    # Assignment 1: e-greedy
    epsilon=0.1
    rewards = np.zeros((n_repetitions, n_timesteps))
    
    for rep in range(n_repetitions):
        q_values = np.zeros(n_actions)
        action_counts = np.zeros(n_actions)
        
        for t in range(n_timesteps):
            if np.random.rand() < epsilon:
                # Explore: Choose a random action
                action = np.random.randint(n_actions)
            else:
                # Exploit: Choose the action with maximum Q-value
                action = np.argmax(q_values)
            
            # Simulate taking the chosen action and observe the reward
            reward = np.random.normal(loc=0.0, scale=1.0)
            
            # Update action-value estimates
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
            
            # Record the reward
            rewards[rep, t] = reward
    
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison
    
    pass

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)
