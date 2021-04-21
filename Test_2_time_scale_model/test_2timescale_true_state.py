import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
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
        s_1h, s_15min_6states, _ = env.reset()
        trading_a = get_1h_action(s_1h, 0)
        rewards = 0

        for k in range(96):
            # concatenate two trading actions into 15min states
            s_15min_8states = np.concatenate((s_15min_6states, trading_a))
            conversion_a = get_15min_action(s_15min_8states, 0)

            # Step the env
            reward, s2_15min_6states, s2_1h, _ = env.convert_energy(trading_a, conversion_a)
            rewards += reward
            s_15min_6states = s2_15min_6states

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
                          tau, q_lr, mu_lr, q_lr*0.1, mu_lr*0.1, gamma, batch_size, replay_size)

    test_returns = []
    returns = []
    q_losses_1h = []
    mu_losses_1h = []
    q_losses_15min = []
    mu_losses_15min = []

    for episode in range(num_train_episodes):
        t0 = datetime.now()
        state_1h, state_15min_6states, state_15min_2real = env.reset()
        if ss > start_steps:
            trading_actions = get_1h_action(state_1h, action_noise)
        else:
            trading_actions = env.sample_trading()
        rewards_15min = []
        # create a list to store 4 15min states, actions, next states
        states_15min_3plus2 = deque([], 4)
        next_15min_3plus2 = deque([], 4)
        conversion_actions_list = deque([], 4)

        for i in range(96):
            # delete e_price for adding to 1h whole states
            states_15min_3plus2.append(np.hstack((state_15min_6states[:3], state_15min_2real)))
            # concatenate two trading actions into 15min states
            state_15min_8states = np.concatenate((state_15min_6states, trading_actions))
            if ss > start_steps:
                conversion_actions = get_15min_action(state_15min_8states, action_noise)
            else:
                conversion_actions = env.sample_conversion()
            conversion_actions_list.append(conversion_actions)

            # Step the env
            reward_15min, next_15min_6states, next_s_1h, next_15min_2real = env.convert_energy(trading_actions, conversion_actions)
            rewards_15min.append(reward_15min)
            next_15min_3plus2.append(np.hstack((next_15min_6states[:3], next_15min_2real)))

            if i == 95:
                done = True
            else:
                done = False
            # at the end of 1 hour, store the 1h memory
            if i % 4 == 3:
                whole_state_1h = np.hstack((state_1h, np.concatenate(states_15min_3plus2)))
                whole_action_1h = np.hstack((trading_actions, np.sum(conversion_actions_list, 0)))
                whole_next_state_1h = np.hstack((next_s_1h, np.concatenate(next_15min_3plus2)))
                reward_1h = np.sum(rewards_15min[-4:])
                td3.memory_1h.store(whole_state_1h, whole_action_1h, reward_1h, whole_next_state_1h, done)
                # next trading actions
                if ss > start_steps:
                    trading_actions = get_1h_action(next_s_1h, action_noise)
                else:
                    trading_actions = env.sample_trading()
                state_1h = next_s_1h

            next_15min_8states = np.concatenate((next_15min_6states, trading_actions))
            td3.memory_15min.store(state_15min_8states, conversion_actions, reward_15min, next_15min_8states, done)

            # Keep track of the number of steps done
            ss += 1
            if ss == start_steps:
                print("USING AGENT ACTIONS NOW")

        for j in range(96):
            # update 15min critic networks
            q_loss_15min, experiences_15min = td3.train_critic_15min(target_noise, noise_clip)
            q_losses_15min.append(q_loss_15min)
            td3.soft_update(td3.q_15min, td3.t_q_15min)
            td3.soft_update(td3.q2_15min, td3.t_q2_15min)
            # delayed policy update
            if j % policy_delay == 0:
                mu_loss_15min = td3.train_actor_15min(experiences_15min)
                mu_losses_15min.append(mu_loss_15min)
                td3.soft_update(td3.mu_15min, td3.t_mu_15min)

            if j % 4 == 0:  # update 1h critic networks
                q_loss_1h, experiences_1h = td3.train_critic_1h(target_noise, noise_clip)
                q_losses_1h.append(q_loss_1h)
                td3.soft_update(td3.q_1h, td3.t_q_1h)
                td3.soft_update(td3.q2_1h, td3.t_q2_1h)

                # delayed policy update
                if j % (policy_delay * 4) == 0:
                    mu_loss_1h = td3.train_actor_1h(experiences_1h)
                    mu_losses_1h.append(mu_loss_1h)
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
    #
    # plt.plot(q_losses_1h, alpha=0.2, c='b')
    # plt.plot(smooth(q_losses_1h, 5000), c='b')
    # plt.title("q losses 1h")
    # plt.show()
    #
    # plt.plot(mu_losses_1h, alpha=0.2, c='b')
    # plt.plot(smooth(mu_losses_1h, 5000), c='b')
    # plt.title("mu losses 1h")
    # plt.show()
    #
    # plt.plot(q_losses_15min, alpha=0.2, c='b')
    # plt.plot(smooth(q_losses_15min, 5000), c='b')
    # plt.title("q losses 15min")
    # plt.show()
    #
    # plt.plot(mu_losses_15min, alpha=0.2, c='b')
    # plt.plot(smooth(mu_losses_15min, 5000), c='b')
    # plt.title("mu losses 15min")
    # plt.show()
