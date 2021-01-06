import gym
import numpy as np
import tensorflow as tf
import time

# Buffer for storing experience tuples of state, action, reward, next state
class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64, num_reward_bins = 100, num_state_bins = 10):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Number of tuples to train on.
        self.batch_size = batch_size

        # Its tells us number of times record() was called.
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.buffer_weights = np.zeros(self.buffer_capacity)
        
        self.state_distribution_edges = [np.linspace(-1., 1., num_state_bins+1)] * num_states
        self.state_distribution = np.zeros([num_state_bins] * num_states)
        self.reward_distribution_edges = [np.linspace(-1., 1., num_reward_bins)]
        self.reward_distribution = np.zeros(num_reward_bins)
        
        # Record of sampling history
        self.sampling_history = {}

        self.compiled = False
    
    def compile(self, buffer_sampling='forgetful'):
        self.buffer_sampling = buffer_sampling
        if self.buffer_sampling not in ['uniform', 'forgetful', 'equalized']:
            print("Unknown sample distribution type provided: {}, defaulting to forgetful".format(self.buffer_sampling))
            self.buffer_sampling = 'forgetful'
        
        if self.buffer_sampling == 'equalized':
            self.buffer_indices = np.zeros(self.buffer_capacity, dtype=int)
        
        self.compiled = True

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, state, action, reward, next_state):
        # Set index to zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity

        if self.buffer_counter >= self.buffer_capacity:
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
        
        if self.buffer_sampling == 'uniform':
            # sample indices using uniform weight on each index
            if self.buffer_counter < self.buffer_capacity:
                self.buffer_weights[:index+1] = 1/(index+1)
        elif self.buffer_sampling == 'forgetful':
            # sample indices with inverse importance for more recent records
            if self.buffer_counter < self.buffer_capacity:
                self.buffer_weights[:index+1] = np.linspace(1, index+1, index+1) * 2 / ((index + 1) * (index + 2))
            else:
                self.buffer_weights = np.roll(self.buffer_weights, 1)
        elif self.buffer_sampling == 'equalized':
            # sample indices using weight = 1/p_i / sum(1/p_i)
            buffer_range = min(self.buffer_counter, self.buffer_capacity)
            self.buffer_indices[index] = reward_hist_index[0]
            inv_hist_sum = 0.0
            for n_i in np.nditer(self.reward_distribution):
                if n_i > 0:
                    inv_hist_sum += 1.0 / n_i
            for i, dist_idx in np.ndenumerate(self.buffer_indices[:buffer_range+1]):
                # dist_idx = get_histogram_index(np.array([r]), self.reward_distribution_edges)
                # print("checking index of {}: {}".format(i, dist_idx))
                w = 1.0 / (self.reward_distribution[dist_idx] ** 2 * inv_hist_sum) 
                self.buffer_weights[i[0]] = w
        if not np.abs(np.sum(self.buffer_weights) - 1.0) < 1e-3:
            print("recording {} formed using {} with sum {}".format(index, self.buffer_sampling, np.sum(self.buffer_weights)))
            print("buffer weights: {}".format(self.buffer_weights[:index+1]))

        self.buffer_counter += 1
        
    def sample(self):

        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, size=self.batch_size, p=self.buffer_weights[:record_range])
        
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

        # If equalized buffer, reassess sample indices
        if self.buffer_sampling == 'equalized':
            for i, r in np.ndenumerate(self.reward_buffer[:record_range]):
                dist_idx = get_histogram_index(np.array([r]), self.reward_distribution_edges)
                self.buffer_indices[i[0]] = dist_idx[0]
    
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
        self.buffer.compile(buffer_sampling=buffer_sampling)
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

            step_durations = []
            # run episode
            step_start = time.clock()
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
                state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()

                # train actor and critic
                critic_loss, actor_loss = self.train_models(gamma, state_batch, action_batch, reward_batch, next_state_batch)

                # propagate weights into target actor and critic
                self.update_targets(tau)

                # End this episode when `done` is True
                step_durations.append(time.clock() - step_start)
                step_start = time.clock()
                if done:
                    break
                
                # Propagate the state forward
                state = next_state
            
            print("Episode {} * Total Time {:.2f}s * Avg Step {:.2f}s * Reward: {:.2f}".format(episode, sum(step_durations), np.average(step_durations), episodic_reward))
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


# This updates target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

