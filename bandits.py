import random
import math
import numpy as np


class Bandit(object):

    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = [1 for col in range(n_arms)]
        self.values = [1e-9 for col in range(n_arms)]

    def select_arm(self):
        pass

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        updated_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = updated_value


class EpsilonGreedy(Bandit):

    def __init__(self, epsilon, counts, values, t=None):
        super(EpsilonGreedy, self).__init__(counts, values)
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.t = t

    @staticmethod
    def ind_max(x):
        m = max(x)
        return x.index(m)

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.ind_max(self.values)
        else:
            return random.randrange(len(self.values))


class Softmax(Bandit):

    def __init__(self, temperature, counts, values):
        super(Softmax, self).__init__(counts, values)
        self.temperature = temperature

    @staticmethod
    def categorical_draw(probs):
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            cum_prob += probs[i]
            if cum_prob > z:
                return i

        return len(probs) - 1

    def select_arm(self):
        z = sum([math.exp(v / self.temperature) for v in self.values])
        probs = [math.exp(v / self.temperature) / z for v in self.values]
        return self.categorical_draw(probs)


class BoltzmanGumbelExploration(Bandit):

    def __init__(self, C, counts, values):
        super(BoltzmanGumbelExploration, self).__init__(counts, values)
        self.C = C

    def _update_beta(self):
        self.beta = [math.sqrt((self.C ** 2) / N) for N in self.counts]

    def calc_perturb(self):
        self._update_beta()
        return np.array([self.beta[i] * np.random.gumbel() for i in range(0, len(self.beta))])

    @staticmethod
    def categorical_draw(probs):
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            cum_prob += probs[i]
            if cum_prob > z:
                return i

        return len(probs) - 1

    def select_arm(self):
        perturb = self.calc_perturb()
        return np.argmax(self.values + perturb)
        # z = sum([v for v in self.values])
        # probs = np.array([v / z for v in self.values])
        # return self.categorical_draw(probs + perturb)

