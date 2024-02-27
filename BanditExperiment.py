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
    epsilon = (0.01, 0.05, 0.1, 0.25)
    plot = LearningCurvePlot("e-greedy learning curve")

    for e in epsilon:
        average_rewards = run_repititions(n_actions, n_timesteps, n_repetitions, policy_type='egreedy', epsilon=e)
        plot.add_curve(smooth(average_rewards, window=smoothing_window), label=f'epsilon: {e}')

    plot.save(name="e-greedy.png")

    # Assignment 2: Optimistic init
    learning_rate = (0.1,0.5,1.0,2.0)
    plot = LearningCurvePlot("Optimistic Initialization learning curve")

    for r in learning_rate:
    	average_rewards = run_repititions(n_actions, n_timesteps, n_repetitions, policy_type='oi', initial_value=r)
    	plot.add_curve(smooth(average_rewards, window=smoothing_window), label=f'Initial_value: {r}')

    plot.save(name="oi.png")

    # Assignment 3: UCB
    exploration_rate = (0.01,0.05,0.1,0.25,0.5,1.0)
    plot = LearningCurvePlot("UCB learning curve")

    for e in exploration_rate:
    	average_rewards = run_repititions(n_actions, n_timesteps, n_repetitions, policy_type='ucb', c=e)
    	plot.add_curve(smooth(average_rewards, window=smoothing_window), label=f'Exploration Rate: {e}')

    plot.save(name="ucb.png")

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
