import random
import numpy as np


def test_algorithm(bandit_obj, arms, num_sims, horizon):
    chosen_arms = np.zeros(shape=(num_sims, horizon), dtype=int)
    rewards = np.zeros(shape=(num_sims, horizon))
    cumulative_rewards = rewards.copy()
    sim_nums = chosen_arms.copy()
    times = chosen_arms.copy()

    for sim in range(num_sims):
        bandit_obj.initialize(len(arms))

        for t in range(horizon):
            sim_nums[sim, t] = sim + 1
            times[sim, t] = t + 1

            chosen_arm = bandit_obj.select_arm()
            chosen_arms[sim, t] = chosen_arm

            reward = arms[chosen_arms[sim, t]].draw()
            rewards[sim, t] = reward

            if t == 0:
                cumulative_rewards[sim, t] = reward
            else:
                cumulative_rewards[sim, t] = cumulative_rewards[sim, t - 1] + reward

            bandit_obj.update(chosen_arm, reward)

    arm_values = [arm.p for arm in arms]
    best_arm = arm_values.index(max(arm_values))
    prob_select_best = np.count_nonzero(chosen_arms == best_arm, axis=0) / num_sims
    cumulative_rewards = np.mean(cumulative_rewards, axis=0)
    return prob_select_best, cumulative_rewards #[sim_nums, times, chosen_arms, rewards, cumulative_rewards]
