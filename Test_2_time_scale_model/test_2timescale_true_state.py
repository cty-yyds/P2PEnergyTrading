import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from Residential_MES import ResidentialMicrogrid
from two_timescale_TD3 import TwoTimescaleTD3
# ReplayBuffer


def smooth(x, t):
    # last t average
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - (t-1))
        y[i] = float(sum(x[start:(i+1)])) / (i - start + 1)
    return y


def get_1h_action(s, noise_scale):
    a = td3.mu_1h(np.array([s])).numpy()[0]
    a += np.random.randn(len(a)) * noise_scale
    return np.clip(a, 0, 1)


def get_15min_action(s, noise_scale):
    a = td3.mu_15min(np.array([s])).numpy()[0]
    a += np.random.randn(len(a)) * noise_scale
    return np.clip(a, 0, 1)


def test_agent():
    test_return = []
    for test_episode in range(10):
        s_1h, s_15min_5states = env.reset()
        trading_a = get_1h_action(s_1h, 0)
        rewards = 0

        for k in range(96):
            # concatenate two trading actions into 15min states
            s_15min_7states = np.concatenate((s_15min_5states, trading_a))
            conversion_a = get_15min_action(s_15min_7states, 0)

            # Step the env
            reward, s2_15min_5states, s2_1h = env.convert_energy(trading_a, conversion_a)
            rewards += reward
            s_15min_5states = s2_15min_5states

            if i % 4 == 3:
                s_1h = s2_1h
                trading_a = get_1h_action(s_1h, 0)

        test_return.append(rewards)
    test_returns.append(np.mean(test_return))
    print('test return:', np.mean(test_return))


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    ss = 0
    env_fn = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    env, test_env = env_fn, env_fn

    num_states_1h = env.n_features_1h
    num_actions_1h = env.n_actions_1h
    num_states_15min = env.n_features_15min
    num_actions_15min = env.n_actions_15min

    num_train_episodes = 10000
    test_agent_every = 20
    replay_size = int(1e5)
    gamma = 0.99
    tau = 0.01
    q_lr = 1e-3
    mu_lr = 1e-4
    batch_size = 100
    start_steps = 5000
    action_noise = 0.05
    target_noise = 0.1
    noise_clip = 0.25  # 0.25
    policy_delay = 2

    td3 = TwoTimescaleTD3(num_states_1h, num_actions_1h, num_states_15min, num_actions_15min,
                          tau, q_lr, mu_lr, gamma, batch_size, replay_size)

    test_returns = []
    returns = []

    for episode in range(num_train_episodes):
        t0 = datetime.now()
        state_1h, state_15min_5states = env.reset()
        if ss > start_steps:
            trading_actions = get_1h_action(state_1h, action_noise)
        else:
            trading_actions = env.sample_trading()
        # concatenate two trading actions into 15min states
        state_15min_7states = np.concatenate((state_15min_5states, trading_actions))
        rewards_15min = []

        for i in range(96):

            if ss > start_steps:
                conversion_actions = get_15min_action(state_15min_7states, action_noise)
            else:
                conversion_actions = env.sample_conversion()

            # Step the env
            reward_15min, next_15min_5states, next_s_1h = env.convert_energy(trading_actions, conversion_actions)
            rewards_15min.append(reward_15min)

            if i == 95:
                done = True
            else:
                done = False
            # at the end of 1 hour, store the 1h memory
            if i % 4 == 3:
                reward_1h = np.sum(rewards_15min[-4:])
                td3.memory_1h.store(state_1h, trading_actions, reward_1h, next_s_1h, done)
                state_1h = next_s_1h
                if ss > start_steps:
                    trading_actions = get_1h_action(state_1h, action_noise)
                else:
                    trading_actions = env.sample_trading()

            next_15min_7states = np.concatenate((next_15min_5states, trading_actions))
            td3.memory_15min.store(state_15min_7states, conversion_actions, reward_15min, next_15min_7states, done)
            state_15min_7states = next_15min_7states

            # Keep track of the number of steps done
            ss += 1
            if ss == start_steps:
                print("USING AGENT ACTIONS NOW")

        for j in range(96):
            # update 15min critic networks
            _, experiences_15min = td3.train_critic_15min(target_noise, noise_clip)
            td3.soft_update(td3.q_15min, td3.t_q_15min)
            td3.soft_update(td3.q2_15min, td3.t_q2_15min)
            # delayed policy update
            if j % policy_delay == 0:
                td3.train_actor_15min(experiences_15min)
                td3.soft_update(td3.mu_15min, td3.t_mu_15min)

            if j % 4 == 0:  # update 1h critic networks
                _, experiences_1h = td3.train_critic_1h(target_noise, noise_clip)
                td3.soft_update(td3.q_1h, td3.t_q_1h)
                td3.soft_update(td3.q2_1h, td3.t_q2_1h)

                # delayed policy update
                if j % (policy_delay * 4) == 0:
                    td3.train_actor_1h(experiences_1h)
                    td3.soft_update(td3.mu_1h, td3.t_mu_1h)

        total_rewards = np.sum(rewards_15min)
        dt = datetime.now() - t0

        if episode % 20 == 0:
            print("Episode:", episode + 1, "Episode Return:", total_rewards, "one_epi Duration:", dt)
        returns.append(total_rewards)

        if episode > 0 and episode % test_agent_every == 0:
            test_agent()


    plt.plot(returns, alpha=0.2, c='b')
    plt.plot(smooth(returns, 500), c='b')
    plt.title("Train returns")
    plt.show()

    plt.plot(test_returns, alpha=0.2, c='b')
    plt.plot(smooth(test_returns, 50), c='b')
    plt.title("Test returns")
    plt.show()
