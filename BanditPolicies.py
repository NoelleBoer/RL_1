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


class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # Initialise the means and counts to 0 for each arm
        self.means = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)
        pass

    def select_action(self, epsilon):
        # Find the index of the current best arm with the highest mean
        best_action = np.argmax(self.means)
        # Initialise the policy with the probability of not selecting the current best arm
        policy = np.full_like(self.means, (epsilon / (self.n_actions - 1)))
        # Set the probabily of selecting the current best arm
        policy[best_action] = 1 - epsilon
        # Sample from the arms using the policy
        a = np.random.choice(self.n_actions, p=policy)
        return a

    def update(self, a, r):
        # Update the number of times arm a has been selected
        self.counts[a] += 1
        # Update the mean of arm a using an incremental update
        self.means[a] += (1 / self.counts[a]) * (r - self.means[a])
        pass


class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        # Initialise the means to the initial value for each arm
        self.means = np.full(n_actions, initial_value)
        pass

    def select_action(self):
        # Find the index of the current best arm with the highest mean
        best_action = np.argmax(self.means)
        # The policy is greedy so we always pick the current best arm
        a = best_action
        return a

    def update(self, a, r):
        # Update the mean of arm a using a learning-based update
        self.means[a] += self.learning_rate * (r - self.means[a])
        pass


class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # Initialise the means and counts to 0 for each arm
        self.means = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)
        pass

    def select_action(self, c, t):
        # Calculate the standard error for all arms (becomes nan when count is 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            std_error = np.sqrt(np.log(t) / self.counts)
        # Calculate the upper confidence bounds for all arms (becomes nan when standard error is nan)
        upper_bounds = self.means + c * std_error
        # Set the upper confidence bound where count was 0 to infinity (replacing every nan)
        np.nan_to_num(upper_bounds, copy=False, nan=np.inf)
        # Find the index of the arm with the highest upper confidence bound
        best_action = np.argmax(upper_bounds)
        # The policy always selects the arm with the highest upper confidence bound
        a = best_action
        return a

    def update(self, a, r):
        # Update the number of times arm a has been selected
        self.counts[a] += 1
        # Update the mean of arm a using an incremental update
        self.means[a] += (1 / self.counts[a]) * (r - self.means[a])
        pass


def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions)  # Initialize environment

    pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(epsilon=0.5)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a, r))

    pi = OIPolicy(n_actions=n_actions, initial_value=1.0)  # Initialize policy
    a = pi.select_action()  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a, r))

    pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(c=1.0, t=1)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test UCB policy with action {}, received reward {}".format(a, r))


if __name__ == '__main__':
    test()
