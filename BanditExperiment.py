#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland

Own code added by Daniël Zee (s2063131) and Noëlle Boer (s2505169)
"""
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth


def run_repititions(n_actions, n_timesteps, n_repetitions, policy_type, epsilon=0.01, initial_value=0.1, c=0.01):
    # Initialise the rewards for every repetition and timestep to 0
    rewards = np.zeros((n_repetitions, n_timesteps))

    for rep in range(n_repetitions):
        # Initialise a new bandit environment for every repetition
        env = BanditEnvironment(n_actions=n_actions)

        # Initialise a new policy for every repetition based on the given argument
        match policy_type:
            case 'egreedy':
                pi = EgreedyPolicy(n_actions=n_actions)
            case 'oi':
                pi = OIPolicy(n_actions=n_actions, initial_value=initial_value)
            case 'ucb':
                pi = UCBPolicy(n_actions=n_actions)
            case _:
                raise ValueError("Invalid policy type given")

        for t in range(n_timesteps):
            # Sample the next action based on the policy type
            match policy_type:
                case 'egreedy':
                    a = pi.select_action(epsilon=epsilon)
                case 'oi':
                    a = pi.select_action()
                case 'ucb':
                    a = pi.select_action(c=c, t=t)
            # Sample the reward from the environment
            r = env.act(a)
            # Update the policy based on the action and reward
            pi.update(a, r)
            # Save the reward at the given repitition and timestep
            rewards[rep, t] = r

    # Average the the rewards at each time step over all repetitions
    average_rewards = np.mean(rewards, axis=0)
    return average_rewards


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    ##########################
    # Assignment 1: e-greedy #
    ##########################

    # Define the values for epsilon to be tested
    epsilon_values = (0.01, 0.05, 0.1, 0.25)
    # Initialise a list for saving the rewards at every value for epsilon in the same order
    rewards_egreedy = []
    # Initialise a new learning curve plot
    plot_egreedy = LearningCurvePlot("ϵ-Greedy Learning Curves")
    for e in epsilon_values:
        # Run the repititions to get the mean rewards over all repititions for all timesteps
        rewards = run_repititions(n_actions, n_timesteps, n_repetitions, policy_type='egreedy', epsilon=e)
        # Smooth the rewards and add the curve to the plot
        plot_egreedy.add_curve(smooth(rewards, window=smoothing_window), label=f'epsilon = {e}')
        # Save the rewards for later use during the comparisons
        rewards_egreedy.append(rewards)
    # Save the plot as an image
    plot_egreedy.save(name="e-greedy.png")

    #################################
    # Assignment 2: Optimistic init #
    #################################

    # Define the initial values to be tested
    initial_values = (0.1, 0.5, 1.0, 2.0)
    # Initialise a list for saving the rewards at every initial value in the same order
    rewards_oi = []
    # Initialise a new learning curve plot
    plot_oi = LearningCurvePlot("Optimistic Initialization Learning Curves (leaning_rate = 0.1)")
    for v in initial_values:
        # Run the repititions to get the mean rewards over all repititions for all timesteps
        rewards = run_repititions(n_actions, n_timesteps, n_repetitions, policy_type='oi', initial_value=v)
        # Smooth the rewards and add the curve to the plot
        plot_oi.add_curve(smooth(rewards, window=smoothing_window), label=f'initial_value = {v}')
        # Save the rewards for later use during the comparisons
        rewards_oi.append(rewards)
    # Save the plot as an image
    plot_oi.save(name="oi.png")

    #####################
    # Assignment 3: UCB #
    #####################

    # Define the values for c to be tested
    c_values = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    # Initialise a list for saving the rewards at every initial value in the same order
    rewards_ucb = []
    # Initialise a new learning curve plot
    plot_ucb = LearningCurvePlot("Upper Conﬁdence Bounds Learning Curves")
    for c in c_values:
        # Run the repititions to get the mean rewards over all repititions for all timesteps
        rewards = run_repititions(n_actions, n_timesteps, n_repetitions, policy_type='ucb', c=c)
        # Smooth the rewards and add the curve to the plot
        plot_ucb.add_curve(smooth(rewards, window=smoothing_window), label=f'c = {c}')
        # Save the rewards for later use during the comparisons
        rewards_ucb.append(rewards)
    # Save the plot as an image
    plot_ucb.save(name="ucb.png")

    ############################
    # Assignment 4: Comparison #
    ############################

    # Compute the average rewards over all timesteps for all previous results
    mean_rewards_egreedy = np.mean(rewards_egreedy, axis=1)
    mean_rewards_oi = np.mean(rewards_oi, axis=1)
    mean_rewards_ucb = np.mean(rewards_ucb, axis=1)
    # Initialise a new comparison plot
    plot_comp = ComparisonPlot(title="Comparison Between ϵ-Greedy, OI and UCB")
    # Plot the parameter values of each approach against the average rewards over all timesteps
    plot_comp.add_curve(epsilon_values, mean_rewards_egreedy, label="ϵ-Greedy (Parameter: epsilon)")
    plot_comp.add_curve(initial_values, mean_rewards_oi, label="OI (Parameter: initial_value, learning_rate = 0.1)")
    plot_comp.add_curve(c_values, mean_rewards_ucb, label="UCB (Parameter: c)")
    # Save the plot as an image
    plot_comp.save(name="comparison.png")

    # Compute the index in the array of the highest average rewards over all timesteps for each approach
    best_epsilon_idx = np.argmax(mean_rewards_egreedy)
    best_initial_value_idx = np.argmax(mean_rewards_oi)
    best_c_idx = np.argmax(mean_rewards_ucb)
    # Initialise a new learning curve plot
    plot_best = LearningCurvePlot("Best Setting Learning Curves for ϵ-Greedy, OI and UCB")
    # For each approach, plot the smoothed curve of the rewards for the parameter setting with the highest average
    # rewards over all timesteps
    plot_best.add_curve(smooth(rewards_egreedy[best_epsilon_idx], window=smoothing_window),
                        label=f'ϵ-Greedy (epsilon = {epsilon_values[best_epsilon_idx]})')
    plot_best.add_curve(smooth(rewards_oi[best_initial_value_idx], window=smoothing_window),
                        label=f'OI (initial_value = {initial_values[best_initial_value_idx]})')
    plot_best.add_curve(smooth(rewards_ucb[best_c_idx], window=smoothing_window),
                        label=f'UCB (c = {c_values[best_c_idx]})')
    # Save the plot as an image
    plot_best.save(name="best_setting.png")


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
