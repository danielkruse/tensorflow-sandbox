import argparse
import gym
import tensorflow as tf
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

SAVE_MODEL_FORMAT = "./saved_models/{}"
GYM_NAMES_1D = [
    'custom_gym:custom-1d-gym-v0'
]
GYM_NAMES_2D = [
    'custom_gym:custom-2d-gym-v0', 
    'custom_gym:custom-2d-obstacles-gym-v0'
]
CLASSIC_CONTROL_GYM_NAMES = [
    'Pendulum-v0'
]
VIABLE_GYM_NAMES = GYM_NAMES_1D + GYM_NAMES_2D + CLASSIC_CONTROL_GYM_NAMES

# Buffer for storing experience tuples of state, action, reward, next state
class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64, num_reward_bins = 100, num_state_bins = 10):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        
        self.state_distribution_edges = [np.linspace(-1., 1., num_state_bins+1)] * num_states
        self.state_distribution = np.zeros([num_state_bins] * num_states)
        self.reward_distribution_edges = [np.linspace(-1., 1., num_reward_bins)]
        self.reward_distribution = np.zeros(num_reward_bins)
        
        # Record of sampling history
        self.sampling_history = {}

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, state, action, reward, next_state):
        # Set index to zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity

        if index < self.buffer_counter:
            # remove prior element from histogram
            last_state_hist_index = get_histogram_index(self.state_buffer[index], self.state_distribution_edges)
            last_reward_hist_index = get_histogram_index(self.reward_buffer[index], self.reward_distribution_edges)
            self.state_distribution[last_state_hist_index] -= 1
            self.reward_distribution[last_reward_hist_index] -= 1
        
        # increment histograms for provided state and reward statistics
        state_hist_index = get_histogram_index(state, self.state_distribution_edges)
        reward_hist_index = get_histogram_index(reward, self.reward_distribution_edges)
        self.state_distribution[state_hist_index] += 1
        self.reward_distribution[reward_hist_index] += 1

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state

        self.buffer_counter += 1
        
    def sample(self, sample_weight_type='uniform'):

        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices using uniform sample_weight_type
        if sample_weight_type == 'uniform':
            sample_weights = 1./record_range * np.ones(record_range)
        elif sample_weight_type == 'equalized':
            reward_pdf = self.reward_distribution / sum(self.reward_distribution)
            reward_cdf = np.insert(np.cumsum(reward_pdf), 0, 0.0)
            r_max = self.reward_distribution_edges[0][-1]
            r_min = self.reward_distribution_edges[0][0]
            uniform_cdf = (self.reward_distribution_edges[0] - r_min)/(r_max - r_min)
            uniform_pdf = uniform_cdf[1:] - uniform_cdf[:-1]
            sample_weights = np.zeros((record_range, 1))
            for i, r in enumerate(self.reward_buffer[:record_range]):
                dist_idx = get_histogram_index([np.array(r)], self.reward_distribution_edges)
                sample_weights[i] = reward_pdf
        else:
            print("Unknown sample distribution type provided: {}, defaulting to uniform".format(sample_weight_type))
            sample_weights = None
        batch_indices = np.random.choice(record_range, size=self.batch_size, p=sample_weights)
        
        # Sample each buffer
        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]

        # Convert indices to subscripts to store in history
        buffer_index = self.buffer_counter % self.buffer_capacity
        buffer_counts = int((self.buffer_counter - buffer_index) / self.buffer_capacity)
        batch_subscripts = [(buffer_counts, i) if i <= buffer_index else (buffer_counts-1, i) for i in batch_indices]

        # Store state and reward from buffer into sample history
        for i, batch_subscript in enumerate(batch_subscripts):
            if batch_subscript in self.sampling_history:
                # update the entry counter
                state, reward, count = self.sampling_history[batch_subscript]
                self.sampling_history[batch_subscript] = (state, reward, count+1)
            else:
                # add sampled subscript into the sampling history
                self.sampling_history[batch_subscript] = (state_batch[i], reward_batch[i], 1)

        # Convert to tensors
        state_batch_tf = tf.convert_to_tensor(state_batch)
        action_batch_tf = tf.convert_to_tensor(action_batch)
        reward_batch_tf = tf.convert_to_tensor(reward_batch)
        reward_batch_tf = tf.cast(reward_batch_tf, tf.float32)
        next_state_batch_tf = tf.convert_to_tensor(next_state_batch)
        return state_batch_tf, action_batch_tf, reward_batch_tf, next_state_batch_tf
    
    def adjust_distribution(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)

        # Re-establish edges for the state buffer
        num_state_bins = np.size(self.state_distribution_edges[0]) - 1 
        self.state_distribution, self.state_distribution_edges = np.histogramdd(self.state_buffer[:record_range], num_state_bins)
        
        # Re-establish edges for the reward buffer
        num_reward_bins = np.size(self.reward_distribution_edges[0]) - 1 
        self.reward_distribution, self.reward_distribution_edges = np.histogramdd(self.reward_buffer[:record_range], num_reward_bins)
    
    def save_buffer(self, filepath):
        sampled_state_distribution = np.zeros(np.shape(self.state_distribution))
        sampled_reward_distribution = np.zeros(np.shape(self.reward_distribution))
        for (state, reward, count) in self.sampling_history.values():
            state_index = get_histogram_index(state, self.state_distribution_edges)
            reward_index = get_histogram_index(reward, self.reward_distribution_edges)
            sampled_state_distribution[state_index] += count
            sampled_reward_distribution[reward_index] += count

        np.savez(filepath + "/state_buffer_distribution", 
            observed_counts=self.state_distribution, 
            sampled_counts=sampled_state_distribution,
            edges=self.state_distribution_edges
        )
        np.savez(filepath + "/reward_buffer_distribution", 
            observed_counts=self.reward_distribution, 
            sampled_counts=sampled_reward_distribution,
            edges=self.reward_distribution_edges
        )

def get_histogram_index(value, dimension_edges):
    hist_indexes = []
    for i, edges_i in enumerate(dimension_edges):
        s_i = value[i]
        if s_i >= edges_i[-1]:
            idx = np.size(edges_i)-2
        elif s_i <= edges_i[0]:
            idx = 0
        else:
            idx = np.argmax(s_i < edges_i)-1
        hist_indexes.append(idx)
    return tuple(hist_indexes)

class DDPG:
    def __init__(self, env, actor_model, critic_model, buffer=None):
        self.env = env
        self.actor = actor_model
        self.critic = critic_model
        if not buffer:
            num_states = env.observation_space.shape[0]
            num_actions = env.action_space.shape[0]
            buffer = Buffer(num_states, num_actions, buffer_capacity=50000, batch_size=32)
        self.buffer = buffer

        # clone trainable models to the target models
        self.target_actor = tf.keras.models.clone_model(actor_model)
        self.target_critic = tf.keras.models.clone_model(critic_model)

        self._compiled = False

    def compile(self, actor_optimizer=tf.keras.optimizers.Adam(), critic_optimizer=tf.keras.optimizers.Adam(), buffer_sampling='uniform'):
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.buffer_sampler = buffer_sampling
        self._compiled = True
    
    def policy(self, state, action_noise_stddev):                
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.actor(tf_state))
        noise = np.random.normal(scale=action_noise_stddev, size=np.size(sampled_actions))
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # Make sure action is within bounds
        legal_action = np.clip(sampled_actions, 
                        self.env.action_space.low, 
                        self.env.action_space.high)

        return np.squeeze(legal_action)
    
    def train(self, total_episodes=100, tau=0.005, gamma=0.99, action_stddev=np.array([0.25]), render_env=False):
        if not self._compiled:
            self.compile()
        # Making the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        history = {
            'episode':[], 
            'critic_loss': [], 
            'actor_loss': [], 
            'reward': []
        }

        for episode in range(total_episodes):
            state = self.env.reset()
            episodic_reward = 0

            # run episode
            while True:
                # Render environment if requested
                if render_env:
                    self.env.render()
                
                action = self.policy(state, action_stddev)

                # Recieve state and reward from environment.
                next_state, reward, done, _ = self.env.step(action)
                
                self.buffer.record(state, action, np.array([reward]), next_state)
                episodic_reward += reward
                # sample buffer
                state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample(sample_weight_type=self.buffer_sampler)

                # train actor and critic
                critic_loss, actor_loss = self.train_models(gamma, state_batch, action_batch, reward_batch, next_state_batch)

                # propagate weights into target actor and critic
                self.update_targets(tau)

                # End this episode when `done` is True
                if done:
                    break
                
                # Propagate the state forward
                state = next_state
            
            print("Episode {} * Reward: {}".format(episode, episodic_reward))
            self.buffer.adjust_distribution()
            # Store metrics about the episode 
            history['episode'].append(episode)
            history['critic_loss'].append(critic_loss)
            history['actor_loss'].append(actor_loss)
            history['reward'].append(episodic_reward)
        
        return history
    
    def train_models(self, gamma, state_batch, action_batch, reward_batch, next_state_batch):
        # Train and update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Train and update actor
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        return critic_loss.numpy(), actor_loss.numpy()

    @tf.function
    def update_targets(self, tau):
        update_target(self.target_actor.variables, self.actor.variables, tau)
        update_target(self.target_critic.variables, self.critic.variables, tau)
    
    def save_models(self, filepath):
        self.actor.save(filepath + "/actor")
        self.target_actor.save(filepath + "/target_actor")
        self.critic.save(filepath + "/critic")
        self.target_critic.save(filepath + "/target_critic")


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor(num_states, num_actions, upper_bound):
    inputs = tf.keras.layers.Input(shape=(num_states,))
    out = tf.keras.layers.Dense(256, activation="relu")(inputs)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = tf.keras.layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)
    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states, num_actions):
    # State as input
    state_input = tf.keras.layers.Input(shape=(num_states))
    state_out = tf.keras.layers.Dense(16, activation="relu")(state_input)
    state_out = tf.keras.layers.Dense(32, activation="relu")(state_out)
    # Action as input
    action_input = tf.keras.layers.Input(shape=(num_actions))
    action_out = tf.keras.layers.Dense(32, activation="relu")(action_input)
    # Both are passed through seperate layer before concatenating
    concat = tf.keras.layers.Concatenate()([state_out, action_out])
    out = tf.keras.layers.Dense(256, activation="relu")(concat)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    outputs = tf.keras.layers.Dense(1)(out)
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

def train_ddpg(gym_name, render_gym=False, total_episodes=100):
    if gym_name not in VIABLE_GYM_NAMES:
        print("{} not a recognized gym".format(gym_name))
        print("Recognized gyms:")
        for viable_gym_name in VIABLE_GYM_NAMES:
            print("{}".format(viable_gym_name))
        return
    env = gym.make(gym_name)

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    
    actor_model = get_actor(num_states, num_actions, upper_bound)
    critic_model = get_critic(num_states, num_actions)

    # Reward buffer to store environment tuples
    buffer = Buffer(num_states, num_actions, buffer_capacity=50000, batch_size=32)

    # Learning rate for actor-critic models
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training hyperparameters
    std_dev = 0.2
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.01

    ddpg_training = DDPG(env, actor_model, critic_model, buffer)
    ddpg_training.compile(actor_optimizer, critic_optimizer)
    ddpg_history = ddpg_training.train(
        total_episodes=total_episodes, 
        gamma=gamma,
        tau=tau,
        action_stddev=std_dev,
        render_env=render_gym
    )

    ddpg_training.save_models(SAVE_MODEL_FORMAT.format(gym_name))
    buffer.save_buffer(SAVE_MODEL_FORMAT.format(gym_name))
    
    # Plotting graph
    # Episodes versus Avg. Rewards
    _, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)
    ax1.plot(ddpg_history['episode'], ddpg_history['reward'])
    ax1.set(ylabel='Episodic Reward')
    ax2.semilogy(ddpg_history['episode'], ddpg_history['critic_loss'])
    ax2.set(ylabel='Critic Loss')
    ax3.semilogy(ddpg_history['episode'], ddpg_history['actor_loss'])
    ax3.set(xlabel='Episode', ylabel='Actor Loss')
    plt.show()

def display_1d_actor_and_critic(actor, critic, env, state_histogram, reward_histogram):
    # data ranges to evaluate over
    xmax, = env.observation_space.high
    xmin, = env.observation_space.low
    xdisp = np.linspace(xmin, xmax, 50)

    umax, = env.action_space.high
    umin, = env.action_space.low
    udisp = np.linspace(umin, umax, 25)

    # evaluate the Q values over X, U grid
    Ugrid, Xgrid = np.meshgrid(udisp, xdisp)
    Qval = critic([Xgrid.flatten(), Ugrid.flatten()])
    Qval = np.reshape(Qval, np.shape(Xgrid))
    
    # extract the optimal u values and corresponding V, A terms from Q
    maxQ_idxs = np.argmax(Qval, axis=1)
    Vval = np.array([Qval[i, max_idx] for i, max_idx in enumerate(maxQ_idxs)])
    uval_critic = udisp[maxQ_idxs]
    Aval = Qval - np.transpose(np.tile(Vval, (np.size(udisp), 1)))

    uval_actor = actor(xdisp)
    
    fig1 = plt.figure()
    ax1_1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1_1.plot_surface(Xgrid, Ugrid, Qval)
    ax1_1.set(xlabel='X', ylabel='U', zlabel='Q', title='Quality')
    ax1_1.grid()
    ax1_2 = fig1.add_subplot(1, 2, 2, projection='3d')
    ax1_2.plot_surface(Xgrid, Ugrid, Aval)
    ax1_2.set(xlabel='X', ylabel='U', zlabel='A', title='Advantage')
    ax1_2.grid()

    fig2 = plt.figure()
    ax2_1 = fig2.add_subplot(2, 1, 1)
    ax2_1.plot(xdisp, uval_actor)
    ax2_1.set(ylabel='U', title='actor')
    ax2_1.grid()

    ax2_2_1 = fig2.add_subplot(2, 1, 2, sharex=ax2_1)
    ax2_2_1.plot(xdisp, uval_critic, color='blue')
    ax2_2_1.set(xlabel='X', title='critic')
    ax2_2_1.set_ylabel('U', color='blue')
    ax2_2_1.tick_params(axis='y', labelcolor='blue')
    ax2_2_1.grid()
    ax2_2_2 = ax2_2_1.twinx()
    ax2_2_2.plot(xdisp, Vval, color='green')
    ax2_2_2.set_ylabel('V', color='green')
    ax2_2_2.tick_params(axis='y', labelcolor='green')
    ax2_2_2.grid()
    fig2.tight_layout()

    observed_state_distribution, sampled_state_distribution, state_bin_edges = state_histogram
    state_bin_centers = [0.5*(edge[1:] + edge[:-1]) for edge in state_bin_edges]
    observed_state_distribution = observed_state_distribution / sum(observed_state_distribution.flatten())
    sampled_state_distribution = sampled_state_distribution / sum(sampled_state_distribution.flatten())

    observed_reward_distribution, sampled_reward_distribution, reward_bin_edges = reward_histogram
    reward_bin_centers = [0.5*(edge[1:] + edge[:-1]) for edge in reward_bin_edges]
    observed_reward_distribution = observed_reward_distribution / sum(observed_reward_distribution)
    sampled_reward_distribution = sampled_reward_distribution / sum(sampled_reward_distribution)

    fig3 = plt.figure()
    ax3_1 = fig3.add_subplot(1, 2, 1)
    ax3_1.bar(state_bin_centers[0], observed_state_distribution, 1.0, alpha=0.4, label='observed')
    ax3_1.bar(state_bin_centers[0], sampled_state_distribution, 1.0, alpha=0.4, label='sampled')
    ax3_1.set(xlabel='X', ylabel='frequency of observance', title='state distribution')
    ax3_1.legend()
    ax3_2 = fig3.add_subplot(1, 2, 2)
    ax3_2.bar(reward_bin_centers[0], observed_reward_distribution, 1.0, alpha=0.4, label='observed')
    ax3_2.bar(reward_bin_centers[0], sampled_reward_distribution, 1.0, alpha=0.4, label='sampled')
    ax3_2.set(xlabel='reward', ylabel='frequency of observance', title='reward distribution')
    ax3_2.legend()

def display_2d_actor_and_critic(actor, critic, env, state_histogram, reward_histogram):
    # data ranges to evaluate over
    xmax, ymax = env.observation_space.high
    xmin, ymin = env.observation_space.low
    xdisp = np.linspace(xmin, xmax, 20)
    ydisp = np.linspace(ymin, ymax, 20)
    Xgrid, Ygrid = np.meshgrid(xdisp, ydisp)
    state_grid_tf = tf.convert_to_tensor(np.transpose([Xgrid.flatten(), Ygrid.flatten()]))

    uxmax, uymax = env.action_space.high
    uthetadisp = np.linspace(0, 2*np.pi, 16)
    umagdisp = np.linspace(0, 1.0, 10)
    Umaggrid, Uthetagrid = np.meshgrid(umagdisp, uthetadisp)
    Uxgrid = uxmax * Umaggrid * np.cos(Uthetagrid)
    Uygrid = uymax * Umaggrid * np.sin(Uthetagrid)
    N_disp_actions = np.size(umagdisp) * np.size(uthetadisp)
    action_grid_tf = tf.convert_to_tensor(np.transpose([Uxgrid.flatten(), Uygrid.flatten()]))

    Qval_critic = np.zeros((np.size(ydisp), np.size(xdisp), np.size(uthetadisp), np.size(umagdisp)))
    Vval_critic = np.zeros((np.size(ydisp), np.size(xdisp)))
    Aval_critic = np.zeros(np.shape(Qval_critic))
    uxval_critic = np.zeros((np.size(ydisp), np.size(xdisp)))
    uyval_critic = np.zeros((np.size(ydisp), np.size(xdisp)))

    # solve for Q-values from critic
    for iy, y in np.ndenumerate(ydisp):
        for ix, x in np.ndenumerate(xdisp):
            state_xy_tf = tf.convert_to_tensor(np.transpose([x * np.ones((N_disp_actions,)), y * np.ones((N_disp_actions,))]))
            
            Qxy = critic([state_xy_tf, action_grid_tf])
            Qxy = np.reshape(Qxy, np.shape(Uxgrid))
            maxQ_idx = np.argmax(Qxy)
            Qval_critic[iy, ix, :, :] = Qxy
            Vval_critic[iy, ix] = Qxy.flatten()[maxQ_idx]
            Aval_critic[iy, ix, :, :] = Qxy - Vval_critic[iy, ix]
            uxval_critic[iy, ix] = Uxgrid.flatten()[maxQ_idx]
            uyval_critic[iy, ix] = Uygrid.flatten()[maxQ_idx]

    # evaluate action values from actor
    uval_actor = actor(state_grid_tf)
    uxval_actor = np.reshape(uval_actor[:, 0].numpy(), np.shape(Xgrid))
    uyval_actor = np.reshape(uval_actor[:, 1].numpy(), np.shape(Ygrid))
    

    fig1 = plt.figure()
    ax1_1 = fig1.add_subplot(1, 2, 1)
    ax1_1.contour(Xgrid, Ygrid, Vval_critic, 10)
    ax1_1.quiver(Xgrid, Ygrid, uxval_critic, uyval_critic, pivot='mid')
    ax1_1.set(xlabel='X', ylabel='Y', title='target critic')
    ax1_1.grid()

    ax1_2 = fig1.add_subplot(1, 2, 2)
    ax1_2.quiver(Xgrid, Ygrid, uxval_actor, uyval_actor, pivot='mid')
    ax1_2.set(xlabel='X', ylabel='Y', title='target actor')
    ax1_2.grid()


    observed_state_distribution, sampled_state_distribution, state_bin_edges = state_histogram
    state_bin_centers = [0.5*(edge[1:] + edge[:-1]) for edge in state_bin_edges]
    observed_state_distribution = observed_state_distribution / sum(observed_state_distribution.flatten())
    sampled_state_distribution = sampled_state_distribution / sum(sampled_state_distribution.flatten())

    observed_reward_distribution, sampled_reward_distribution, reward_bin_edges = reward_histogram
    reward_bin_centers = [0.5*(edge[1:] + edge[:-1]) for edge in reward_bin_edges]
    observed_reward_distribution = observed_reward_distribution / sum(observed_reward_distribution)
    sampled_reward_distribution = sampled_reward_distribution / sum(sampled_reward_distribution)

    fig2 = plt.figure()
    state_bnds = (state_bin_centers[0][0], state_bin_centers[0][-1], state_bin_centers[-1][0], state_bin_centers[0][-1])
    ax2_1 = fig2.add_subplot(1, 3, 1)
    obs_state_dist_imshow = ax2_1.imshow(observed_state_distribution, extent=state_bnds)
    ax2_1.set(xlabel='X', ylabel='Y', title='observed state distribution')
    fig2.colorbar(obs_state_dist_imshow, ax=ax2_1)

    ax2_2 = fig2.add_subplot(1, 3, 2)
    smpl_state_dist_imshow = ax2_2.imshow(sampled_state_distribution, extent=state_bnds)
    ax2_2.set(xlabel='X', ylabel='Y', title='sampled state distribution')
    fig2.colorbar(smpl_state_dist_imshow, ax=ax2_2)
    
    ax2_3 = fig2.add_subplot(1, 3, 3)
    ax2_3.bar(reward_bin_centers[0], observed_reward_distribution, 1.0, alpha=0.4, label='observed')
    ax2_3.bar(reward_bin_centers[0], sampled_reward_distribution, 1.0, alpha=0.4, label='sampled')
    ax2_3.set(xlabel='reward', ylabel='frequency of observance', title='reward distribution')
    ax2_3.legend()

def display_ddpg_results(gym_name):
    if gym_name not in VIABLE_GYM_NAMES:
        print("{} not a recognized gym".format(gym_name))
        print("Recognized gyms:")
        for viable_gym_name in VIABLE_GYM_NAMES:
            print("{}".format(viable_gym_name))
        return
    env = gym.make(gym_name)
    data_folder = SAVE_MODEL_FORMAT.format(gym_name)
    try:
        actor = tf.keras.models.load_model(data_folder + "/actor")
        critic = tf.keras.models.load_model(data_folder + "/critic")
        target_actor = tf.keras.models.load_model(data_folder + "/target_actor")
        target_critic = tf.keras.models.load_model(data_folder + "/target_critic")
    except OSError:
        print("No saved DDPG models found for {}".format(gym_name))
        return
    
    try:
        state_distribution_npzfile = np.load(data_folder + "/state_buffer_distribution.npz")
        observed_state_distribution = state_distribution_npzfile['observed_counts']
        sampled_state_distribution = state_distribution_npzfile['sampled_counts']
        state_distribution_edges = state_distribution_npzfile['edges']
        state_histogram = (observed_state_distribution, sampled_state_distribution, state_distribution_edges)

        reward_distribution_npzfile = np.load(data_folder + "/reward_buffer_distribution.npz")
        observed_reward_distribution = reward_distribution_npzfile['observed_counts']
        sampled_reward_distribution = reward_distribution_npzfile['sampled_counts']
        reward_distribution_edges = reward_distribution_npzfile['edges']
        reward_histogram = (observed_reward_distribution, sampled_reward_distribution, reward_distribution_edges)
    except OSError:
        print("No distribution data found for {}".format(gym_name))
        return

    if gym_name in GYM_NAMES_1D:
        display_1d_actor_and_critic(target_actor, target_critic, env, state_histogram, reward_histogram)
    elif gym_name in GYM_NAMES_2D:
        display_2d_actor_and_critic(target_actor, target_critic, env, state_histogram, reward_histogram)
    else:
        print("Not currently an implemented display for {}".format(gym_name))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Tensorflow DDPG example')
    parser.add_argument('-g', '--gym', help='name of gym to train against')
    parser.add_argument('-d', '--display', help='display results from training on provided gym')
    parser.add_argument('-e', '--episodes', default=100, type=int, help='number of episodes to train over')
    parser.add_argument('--render_gym', action='store_true', help='render gym')
    args = parser.parse_args()

    if args.gym:
        train_ddpg(args.gym, render_gym=args.render_gym, total_episodes=args.episodes)
    elif args.display:
        display_ddpg_results(args.display)
