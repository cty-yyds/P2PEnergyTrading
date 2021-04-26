import numpy as np
import tensorflow as tf


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, batch_size=32):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.batch_size = batch_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        states = self.obs1_buf[idxs]
        actions = self.acts_buf[idxs]
        rewards = self.rews_buf[idxs]
        next_states = self.obs2_buf[idxs]
        dones = self.done_buf[idxs]
        return states, actions, rewards, next_states, dones


def create_actor(num_states, num_actions):
    print("Now we build the actor")
    input_ = tf.keras.layers.Input(shape=[num_states])
    h1 = tf.keras.layers.Dense(400, activation='relu')(input_)
    h2 = tf.keras.layers.Dense(300, activation='relu')(h1)
    mu = tf.keras.layers.Dense(num_actions, activation='sigmoid')(h2)
    model = tf.keras.Model(inputs=[input_], outputs=[mu])
    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    return model, target_model


def create_critic(num_states, num_actions):
    print("Now we build the critic")
    input_state = tf.keras.layers.Input(shape=[num_states])
    input_action = tf.keras.layers.Input(shape=[num_actions])
    concate1 = tf.keras.layers.concatenate([input_state, input_action])
    h1 = tf.keras.layers.Dense(400, activation='relu')(concate1)
    concate2 = tf.keras.layers.concatenate([h1, input_action])
    h2 = tf.keras.layers.Dense(300, activation='relu')(concate2)
    q = tf.squeeze(tf.keras.layers.Dense(1, activation=None)(h2), axis=1)
    model = tf.keras.Model(inputs=[input_state, input_action], outputs=[q])
    model2 = tf.keras.models.clone_model(model)
    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    target_model2 = tf.keras.models.clone_model(model)
    target_model2.set_weights(model2.get_weights())
    return model, target_model, model2, target_model2


# calculate energy level of ESS when finding target action in critic training
def calculate_ess_15min(energy_conversion, energy_trading, states, ess_level):

    k_we = 0.8  # water electrolyser efficiency
    k_fc_e = 0.6  # fuel cell to electricity efficiency
    k_h2q = 33.33  # hydrogen(kg) to Q(KWh) ratio
    pes_max = 25.0
    B_e = 300.0
    B_h2 = 30.0
    HT_p_max = 3.0

    # ---------------------energy conversion----------------------
    a_WE = energy_conversion[:, :1] * 200 / 4  # kwh
    a_FC = energy_conversion[:, 1:2] * 8 / 4  # kg
    a_HB = energy_conversion[:, 2:3]  # kg
    y_elec = (energy_trading[:, :1] * 2 - 1) * 100 / 4  # kwh

    E_FC = a_FC * k_h2q * k_fc_e  # electricity (kwh) output from FC
    h2_WE = a_WE * k_we / k_h2q  # hydrogen output (kg) from WE

    # ---------------------energy balance-------------------------
    solar = states[:, :1]  # solar output
    E_demand = states[:, 1:2]  # electricity load

    E_battery = solar + y_elec + E_FC - a_WE - E_demand  # battery charging (if>0) amount by electricity balance
    E_battery = tf.where(E_battery < -pes_max, -pes_max, E_battery)
    E_battery = tf.where(E_battery < 0, E_battery/0.9, E_battery)
    E_battery = tf.where(E_battery > pes_max, pes_max, E_battery)
    E_battery = tf.where(E_battery > 0, E_battery * 0.9, E_battery)

    battery_level = ess_level[:, :1]*B_e + E_battery
    battery_level = tf.where(battery_level > B_e, B_e, battery_level)
    battery_level = tf.where(battery_level < 0, 0., battery_level)

    h2_HT = h2_WE - a_FC - a_HB  # hydrogen tank inflow amount (if>0) kg by hydrogen balance
    h2_HT = tf.where(h2_HT < -HT_p_max, -HT_p_max, h2_HT)
    h2_HT = tf.where(h2_HT < 0, h2_HT/0.95, h2_HT)
    h2_HT = tf.where(h2_HT > HT_p_max, HT_p_max, h2_HT)
    h2_HT = tf.where(h2_HT > 0, h2_HT * 0.95, h2_HT)

    ht_level = ess_level[:, 1:2]*B_h2 + h2_HT
    ht_level = tf.where(ht_level > B_h2, B_h2, ht_level)
    ht_level = tf.where(ht_level < 0, 0., ht_level)

    return tf.concat([battery_level/B_e, ht_level/B_h2], 1)


class TwoTimescaleTD3:
    def __init__(self, n_states_1h, n_actions_1h, n_states_15min, n_actions_15min,
                 tau, q_lr_1h, mu_lr_1h, q_lr_15min, mu_lr_15min, gamma, batch_size, replay_capacity):
        self.tau = tau
        self.gamma = gamma
        self.n_whole_s_1h = n_states_1h + 5 * 4
        self.n_whole_a_1h = n_actions_1h + n_actions_15min
        self.n_whole_s_15min = n_states_15min + n_states_1h
        self.n_whole_a_15min = n_actions_15min + n_actions_1h
        self.loss_fn = tf.keras.losses.mean_squared_error
        self.q_optimizer_1h = tf.keras.optimizers.Nadam(learning_rate=q_lr_1h)
        self.mu_optimizer_1h = tf.keras.optimizers.Nadam(learning_rate=mu_lr_1h)
        self.q_optimizer_15min = tf.keras.optimizers.Nadam(learning_rate=q_lr_15min, clipnorm=1.0)
        self.mu_optimizer_15min = tf.keras.optimizers.Nadam(learning_rate=mu_lr_15min)
        self.memory_1h = ReplayBuffer(self.n_whole_s_1h, self.n_whole_a_1h, replay_capacity, batch_size)
        self.memory_15min = ReplayBuffer(self.n_whole_s_15min, self.n_whole_a_15min, replay_capacity, batch_size)

        # Now create the model (2 timescale, double Q, target networks) whole centralised critic with 4 15min s and a
        self.mu_1h, self.t_mu_1h = create_actor(n_states_1h, n_actions_1h)
        self.q_1h, self.t_q_1h, self.q2_1h, self.t_q2_1h = create_critic(n_states_1h, self.n_whole_a_1h)

        self.mu_15min, self.t_mu_15min = create_actor(n_states_15min, n_actions_15min)
        self.q_15min, self.t_q_15min, self.q2_15min, self.t_q2_15min = create_critic(n_states_15min, self.n_whole_a_15min)

    @tf.function
    def train_critic_1h(self, target_noise, noise_clip):
        experiences = self.memory_1h.sample_batch()
        states, actions, rewards, next_states, dones = experiences
        s_1h = states[:, :6]
        s2_1h = next_states[:, :6]
        next_mu_1h = self.t_mu_1h(s2_1h)

        # calculate next conversion actions sum in one hour
        s2_15min_1 = tf.concat([next_states[:, 6:9], next_states[:, 3:6]], 1)
        next_mu_15min_1 = self.t_mu_15min(s2_15min_1)
        new_ess_level = calculate_ess_15min(next_mu_15min_1, next_mu_1h, next_states[:, 9:11], next_states[:, 3:5])

        s2_15min_2 = tf.concat([next_states[:, 11:14], new_ess_level, next_states[:, 6:7]], 1)
        next_mu_15min_2 = self.t_mu_15min(s2_15min_2)
        new_ess_level = calculate_ess_15min(next_mu_15min_2, next_mu_1h, next_states[:, 14:16], new_ess_level)

        s2_15min_3 = tf.concat([next_states[:, 16:19], new_ess_level, next_states[:, 6:7]], 1)
        next_mu_15min_3 = self.t_mu_15min(s2_15min_3)
        new_ess_level = calculate_ess_15min(next_mu_15min_3, next_mu_1h, next_states[:, 19:21], new_ess_level)

        s2_15min_4 = tf.concat([next_states[:, 21:24], new_ess_level, next_states[:, 6:7]], 1)
        next_mu_15min_4 = self.t_mu_15min(s2_15min_4)

        next_mu_15min = next_mu_15min_1 + next_mu_15min_2 + next_mu_15min_3 + next_mu_15min_4
        next_mu_values = tf.concat([next_mu_1h, next_mu_15min], 1)

        epsilon = tf.random.normal(tf.shape(next_mu_values), stddev=target_noise)  # Target Policy Noise
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        next_mu_noise = next_mu_values + epsilon
        next_mu_noise = tf.clip_by_value(next_mu_noise, tf.zeros(tf.shape(next_mu_values)),
                                         tf.ones(tf.shape(next_mu_values)))  # valid range of actions
        next_q_values = self.t_q_1h((s2_1h, next_mu_noise))
        next_q2_values = self.t_q2_1h((s2_1h, next_mu_noise))
        # double q network to prevent overestimate the q value
        target_q_values = rewards + (1 - dones) * self.gamma * tf.minimum(next_q_values, next_q2_values)
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.q_1h((s_1h, actions))
            q2 = self.q2_1h((s_1h, actions))
            q1_loss = tf.reduce_mean(self.loss_fn(target_q_values, q1))
            q2_loss = tf.reduce_mean(self.loss_fn(target_q_values, q2))
            q_loss = q1_loss + q2_loss
        q1_grads = tape.gradient(q_loss, self.q_1h.trainable_variables)
        q2_grads = tape.gradient(q_loss, self.q2_1h.trainable_variables)
        del tape

        self.q_optimizer_1h.apply_gradients(zip(q1_grads, self.q_1h.trainable_variables))
        self.q_optimizer_1h.apply_gradients(zip(q2_grads, self.q2_1h.trainable_variables))
        return q_loss, experiences

    @tf.function
    def train_critic_15min(self, target_noise, noise_clip):
        experiences = self.memory_15min.sample_batch()
        states, actions, rewards, next_states, dones = experiences
        s_15min = states[:, :6]
        s2_15min = next_states[:, :6]
        next_mu_15min = self.t_mu_15min(s2_15min)

        # calculate next trading actions (1h)
        s2_1h = next_states[:, 6:]
        next_mu_1h = self.t_mu_1h(s2_1h)

        next_mu_values = tf.concat([next_mu_15min, next_mu_1h], 1)
        epsilon = tf.random.normal(tf.shape(next_mu_values), stddev=target_noise)  # Target Policy Noise
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        next_mu_noise = next_mu_values + epsilon
        next_mu_noise = tf.clip_by_value(next_mu_noise, tf.zeros(tf.shape(next_mu_values)),
                                         tf.ones(tf.shape(next_mu_values)))  # valid range of actions
        next_q_values = self.t_q_15min((s2_15min, next_mu_noise))
        next_q2_values = self.t_q2_15min((s2_15min, next_mu_noise))
        # double q network to prevent overestimate the q value
        target_q_values = rewards + (1 - dones) * self.gamma * tf.minimum(next_q_values, next_q2_values)
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.q_15min((s_15min, actions))
            q2 = self.q2_15min((s_15min, actions))
            q1_loss = tf.reduce_mean(self.loss_fn(target_q_values, q1))
            q2_loss = tf.reduce_mean(self.loss_fn(target_q_values, q2))
            q_loss = q1_loss + q2_loss
        q1_grads = tape.gradient(q_loss, self.q_15min.trainable_variables)
        q2_grads = tape.gradient(q_loss, self.q2_15min.trainable_variables)
        del tape

        grads_norm = tf.norm(q1_grads[0])
        # tf.print(grads_norm)
        self.q_optimizer_15min.apply_gradients(zip(q1_grads, self.q_15min.trainable_variables))
        self.q_optimizer_15min.apply_gradients(zip(q2_grads, self.q2_15min.trainable_variables))
        return grads_norm, experiences

    @tf.function
    def train_actor_1h(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        s_1h = states[:, :6]
        a_15min = actions[:, 2:]

        with tf.GradientTape() as tape:
            mu = self.mu_1h(s_1h)
            whole_actions = tf.concat([mu, a_15min], 1)
            q_mu = self.q_1h((s_1h, whole_actions))
            mu_loss = -tf.reduce_mean(q_mu)
        mu_grads = tape.gradient(mu_loss, self.mu_1h.trainable_variables)

        self.mu_optimizer_1h.apply_gradients(zip(mu_grads, self.mu_1h.trainable_variables))
        return mu_loss

    @tf.function
    def train_actor_15min(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        s_15min = states[:, :6]
        a_1h = actions[:, 3:]

        with tf.GradientTape() as tape:
            mu = self.mu_15min(s_15min)
            whole_actions = tf.concat([mu, a_1h], 1)
            q_mu = self.q_15min((s_15min, whole_actions))
            mu_loss = -tf.reduce_mean(q_mu)
        mu_grads = tape.gradient(mu_loss, self.mu_15min.trainable_variables)

        self.mu_optimizer_15min.apply_gradients(zip(mu_grads, self.mu_15min.trainable_variables))
        return mu_loss

    def soft_update(self, model, target_model):
        model_weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        target_model.set_weights(target_weights)
