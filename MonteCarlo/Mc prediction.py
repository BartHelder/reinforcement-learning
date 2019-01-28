import gym
import numpy as np
import sys

from collections import defaultdict

from lib.envs.blackjack import BlackjackEnv
from lib import plotting



def mc_prediction(policy, env, num_episodes, discount_factor=1.0):

    returns_sum = defaultdict(float)    #
    returns_count = defaultdict(float)
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # Debug num of episodes
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            first_time_id = next(i for i, x in enumerate(episode) if x[0] == state)
            G = sum([x[2]*(discount_factor**i) for i, x in enumerate(episode[first_time_id:])])
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

    return V

def sample_policy(observation):

    score, dealer_score, usable_ace = observation
    return 0 if score >= 18 else 1


if __name__ == '__main__':
    env = BlackjackEnv()
    V_50k = mc_prediction(sample_policy, env, 500000, 1.0)
    plotting.plot_value_function(V_50k, title='50k >= 18')