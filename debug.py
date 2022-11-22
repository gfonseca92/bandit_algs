from bandits import EpsilonGreedy, Softmax, BoltzmanGumbelExploration
from simulation_paths import BernoulliArm
from simulation import test_algorithm

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
means = [0.51, 0.5, 0.5, 0.5, 0.5, 0.5]
n_arms = len(means)
random.shuffle(means)
arms = list(map(lambda x: BernoulliArm(x), means))

results_mtx = np.zeros(shape=(5000, 5))
count_epsilon = 0


plt.figure(figsize=(12, 6))
for const in [0.1, 0.2, 0.3, 0.4, 0.5]:
    bandit_alg = BoltzmanGumbelExploration(const, [], [])
    bandit_alg.initialize(n_arms)
    prob_select_best, cumulative_rewards = test_algorithm(bandit_alg, arms, 2000, 1000)
    plt.subplot(1, 2, 1)
    plt.plot(prob_select_best)
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_rewards)
    # results_mtx[:, count_epsilon] = results

plt.show()
print("oi")

