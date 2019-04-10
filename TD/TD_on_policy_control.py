import gym
import numpy as np
import itertools
import matplotlib
import pandas as pd
import sys

from lib.envs.windy_gridworld_king import WindyGridworldEnv
from collections import defaultdict
from lib import plotting
matplotlib.style.use('ggplot')

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy function from a given action-value function and epsilon
    :param Q: Dictionary that maps state -> action values. Each entry is of length nA,
    :param epsilon: Probability to select a random action (float, 0<=epsilon<1)
    :param nA: number of actions possible in the environment
    :return: policy function that takes the observation of the environment as an argument and returns the action
    choice probabilities in the form of an np.array of length nA
    """
    def policy_function(observation):
        A = np.ones(nA) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1 - epsilon)
        return A

    return policy_function

#def make_greedy_policy()

def sarsa(env, num_episodes, discount_factor = 1.0, alpha = 0.5, epsilon = 0.1):
    """
    SARSA algorithm: on-policy TD control, finds the optimal epsilon-greedy policy
    :param env:
    :param num_episodes:
    :param discount_factor:
    :param alpha:
    :param epsilon:
    :return:
    """

    # The (final) action-value function, nested dict
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Episode statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))

    # Policy-to-follow:
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Run through the episodes
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode+1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

        for t in itertools.count():
            # Perform action:
            next_state, reward, done, _ = env.step(action)

            # Based on results, pick the next action:
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

            # Update statistics from reward etc
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # TD update:
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break
            state = next_state
            action = next_action

    return Q, stats


if __name__ == "__main__":
    env = WindyGridworldEnv()
    Q, stats = sarsa(env, 2000)

    plotting.plot_episode_stats(stats)








