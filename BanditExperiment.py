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


def run_repititions(n_actions, n_timesteps, n_repetitions, policy_type, epsilon=0.01, initial_value=0.1, c=0.01):
    rewards = np.zeros((n_repetitions, n_timesteps))

    for rep in range(n_repetitions):
        env = BanditEnvironment(n_actions=n_actions)

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
            match policy_type:
                case 'egreedy':
                    a = pi.select_action(epsilon=epsilon)
                case 'oi':
                    a = pi.select_action()
                case 'ucb':
                    a = pi.select_action(c=c, t=t)
            r = env.act(a)
            pi.update(a, r)
            rewards[rep, t] = r

    average_rewards = np.mean(rewards, axis=0)
    return average_rewards


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    # To Do: Write all your experiment code here

    # Assignment 1: e-greedy
    average_rewards = run_repititions(n_actions, n_timesteps, n_repetitions, policy_type='egreedy', epsilon=0.01)
    plot = LearningCurvePlot("e-greedy learning curve")
    plot.add_curve(average_rewards, label="raw")
    plot.add_curve(smooth(average_rewards, window=smoothing_window), label="smoothed")
    plot.save(name="e-greedy.png")

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

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
