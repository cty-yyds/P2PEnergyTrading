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


class TwoTimescaleTD3:
    def __init__(self, n_states_1h, n_actions_1h, n_states_15min, n_actions_15min,
                 tau, q_lr, mu_lr, gamma, batch_size, replay_capacity):
        self.tau = tau
        self.gamma = gamma
        self.loss_fn = tf.keras.losses.mean_squared_error
        self.q_optimizer = tf.keras.optimizers.Nadam(learning_rate=q_lr)
        self.mu_optimizer = tf.keras.optimizers.Nadam(learning_rate=mu_lr)
        self.memory_1h = ReplayBuffer(n_states_1h, n_actions_1h, replay_capacity, batch_size)
        self.memory_15min = ReplayBuffer(n_states_15min, n_actions_15min, replay_capacity, batch_size)

        # Now create the model (2 timescale, double Q, target networks)
        self.mu_1h, self.t_mu_1h = create_actor(n_states_1h, n_actions_1h)
        self.q_1h, self.t_q_1h, self.q2_1h, self.t_q2_1h = create_critic(n_states_1h, n_actions_1h)

        self.mu_15min, self.t_mu_15min = create_actor(n_states_15min, n_actions_15min)
        self.q_15min, self.t_q_15min, self.q2_15min, self.t_q2_15min = create_critic(n_states_15min,
                                                                                     n_actions_15min)

    def train_critic_1h(self, target_noise, noise_clip):
        experiences = self.memory_1h.sample_batch()
        states, actions, rewards, next_states, dones = experiences
        next_mu = self.t_mu_1h(next_states)
        epsilon = tf.random.normal(tf.shape(next_mu), stddev=target_noise)  # Target Policy Noise
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        next_mu_noise = next_mu + epsilon
        next_mu_noise = tf.clip_by_value(next_mu_noise, tf.zeros(tf.shape(next_mu)),
                                         tf.ones(tf.shape(next_mu)))  # valid range of actions
        next_q_values = self.t_q_1h((next_states, next_mu_noise))
        next_q2_values = self.t_q2_1h((next_states, next_mu_noise))
        # double q network to prevent overestimate the q value
        target_q_values = rewards + (1 - dones) * self.gamma * tf.minimum(next_q_values, next_q2_values)
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.q_1h((states, actions))
            q2 = self.q2_1h((states, actions))
            q1_loss = tf.reduce_mean(self.loss_fn(target_q_values, q1))
            q2_loss = tf.reduce_mean(self.loss_fn(target_q_values, q2))
            q_loss = q1_loss + q2_loss
        q1_grads = tape.gradient(q_loss, self.q_1h.trainable_variables)
        q2_grads = tape.gradient(q_loss, self.q2_1h.trainable_variables)
        del tape

        self.q_optimizer.apply_gradients(zip(q1_grads, self.q_1h.trainable_variables))
        self.q_optimizer.apply_gradients(zip(q2_grads, self.q2_1h.trainable_variables))
        return q_loss, experiences

    def train_critic_15min(self, target_noise, noise_clip):
        experiences = self.memory_15min.sample_batch()
        states, actions, rewards, next_states, dones = experiences
        next_mu = self.t_mu_15min(next_states)
        epsilon = tf.random.normal(tf.shape(next_mu), stddev=target_noise)  # Target Policy Noise
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        next_mu_noise = next_mu + epsilon
        next_mu_noise = tf.clip_by_value(next_mu_noise, tf.zeros(tf.shape(next_mu)),
                                         tf.ones(tf.shape(next_mu)))  # valid range of actions
        next_q_values = self.t_q_15min((next_states, next_mu_noise))
        next_q2_values = self.t_q2_15min((next_states, next_mu_noise))
        # double q network to prevent overestimate the q value
        target_q_values = rewards + (1 - dones) * self.gamma * tf.minimum(next_q_values, next_q2_values)
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.q_15min((states, actions))
            q2 = self.q2_15min((states, actions))
            q1_loss = tf.reduce_mean(self.loss_fn(target_q_values, q1))
            q2_loss = tf.reduce_mean(self.loss_fn(target_q_values, q2))
            q_loss = q1_loss + q2_loss
        q1_grads = tape.gradient(q_loss, self.q_15min.trainable_variables)
        q2_grads = tape.gradient(q_loss, self.q2_15min.trainable_variables)
        del tape

        self.q_optimizer.apply_gradients(zip(q1_grads, self.q_15min.trainable_variables))
        self.q_optimizer.apply_gradients(zip(q2_grads, self.q2_15min.trainable_variables))
        return q_loss, experiences

    def train_actor_1h(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape:
            mu = self.mu_1h(states)
            q_mu = self.q_1h((states, mu))
            mu_loss = -tf.reduce_mean(q_mu)
        mu_grads = tape.gradient(mu_loss, self.mu_1h.trainable_variables)

        self.mu_optimizer.apply_gradients(zip(mu_grads, self.mu_1h.trainable_variables))
        return mu_loss

    def train_actor_15min(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape:
            mu = self.mu_15min(states)
            q_mu = self.q_15min((states, mu))
            mu_loss = -tf.reduce_mean(q_mu)
        mu_grads = tape.gradient(mu_loss, self.mu_15min.trainable_variables)

        self.mu_optimizer.apply_gradients(zip(mu_grads, self.mu_15min.trainable_variables))
        return mu_loss

    def soft_update(self, model, target_model):
        model_weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        target_model.set_weights(target_weights)
