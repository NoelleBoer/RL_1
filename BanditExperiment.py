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
    # Initialize rewards
    rewards_e = np.zeros((n_repetitions, n_timesteps))
    rewards_oi = np.zeros((n_repetitions, n_timesteps))
    rewards_ucb = np.zeros((n_repetitions, n_timesteps))

    # Assignment 1: e-greedy
    epsilon=0.1

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
            rewards_e[rep, t] = reward

    # Assignment 2: Optimistic init
    for rep in range(n_repetitions):
        q_values = np.full(n_actions, initial_value)
        action_counts = np.zeros(n_actions)

        for t in range(n_timesteps):
            # Choose the action with maximum Q-value
            action = np.argmax(q_values)

            # Simulate taking the chosen action and observe the reward
            reward = np.random.normal(loc=0.0, scale=1.0)

            # Update action-value estimates
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]

            # Record the reward
            rewards_oi[rep, t] = reward


    # Assignment 3: UCB
    for rep in range(n_repetitions):
        q_values = np.zeros(n_actions)
        action_counts = np.zeros(n_actions)

        for t in range(n_timesteps):
            # Calculate UCB values for each action
            ucb_values = q_values + c * np.sqrt(np.log(t + 1) / (action_counts + 1e-6))

            # Choose the action with maximum UCB value
            action = np.argmax(ucb_values)

            # Simulate taking the chosen action and observe the reward
            reward = np.random.normal(loc=0.0, scale=1.0)

            # Update action-value estimates
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]

            # Record the reward
            rewards_ucb[rep, t] = reward

    # Assignment 4: Comparison
    average_rewards_e = np.mean(rewards_e, axis=0)
    smoothed_rewards_e = np.convolve(average_rewards_e, np.ones(smoothing_window)/smoothing_window, mode='valid')

    average_rewards_oi = np.mean(rewards_oi, axis=0)
    smoothed_rewards_oi = np.convolve(average_rewards_oi, np.ones(smoothing_window)/smoothing_window, mode='valid')

    average_rewards_ucb = np.mean(rewards_ucb, axis=0)
    smoothed_rewards_ucb = np.convolve(average_rewards_ucb, np.ones(smoothing_window)/smoothing_window, mode='valid')

    return smoothed_rewards_e, smoothed_rewards_oi, smoothed_rewards_ucb

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)
