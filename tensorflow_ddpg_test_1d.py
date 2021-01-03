import argparse
import gym
import tensorflow as tf
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import custom_tf_ddpg

VIABLE_GYM_NAMES = [
    'custom_gym:custom-1d-gym-v0'
]
SAVE_MODEL_FORMAT = "./saved_models/{}"

def format_gym_name(gym_name):
    return gym_name[gym_name.find(':')+1:]

# Generate actor NN
def generate_actor(num_states, num_actions, upper_bound):
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

# Generate critic NN
def generate_critic(num_states, num_actions):
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

def display_1d_actor_and_critic(actor, critic, observation_space, action_space, state_histogram, reward_histogram):
    # data ranges to evaluate over
    xmax, = observation_space.high
    xmin, = observation_space.low
    xdisp = np.linspace(xmin, xmax, 50)

    umax, = action_space.high
    umin, = action_space.low
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

def display_ddpg_results(gym_name):
    if gym_name not in VIABLE_GYM_NAMES:
        print("{} not a recognized gym".format(gym_name))
        print("Recognized gyms:")
        for viable_gym_name in VIABLE_GYM_NAMES:
            print("{}".format(viable_gym_name))
        return
    env = gym.make(gym_name)
    
    actor, critic, target_actor, target_critic = load_ddpg(format_gym_name(gym_name))
    state_histogram = load_histogram(format_gym_name(gym_name), 'state_buffer_distribution')
    reward_histogram = load_histogram(format_gym_name(gym_name), 'reward_buffer_distribution')

    display_1d_actor_and_critic(
        target_actor, 
        target_critic, 
        env.observation_space, 
        env.action_space, 
        state_histogram, 
        reward_histogram
    )
    plt.show()

# Train DDPG 
def train_ddpg(gym_name, 
    render_gym=False, 
    total_episodes=100):

    env = gym.make(gym_name)

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    
    actor_model = generate_actor(num_states, num_actions, upper_bound)
    critic_model = generate_critic(num_states, num_actions)

    # Reward buffer to store environment tuples
    buffer = custom_tf_ddpg.Buffer(num_states, num_actions, buffer_capacity=50000, batch_size=32)

    # Learning rate for actor-critic models
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training hyperparameters
    std_dev = 0.2
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.01

    ddpg_training = custom_tf_ddpg.DDPG(env, actor_model, critic_model, buffer)
    ddpg_training.compile(actor_optimizer, critic_optimizer)
    ddpg_history = ddpg_training.train(
        total_episodes=total_episodes, 
        gamma=gamma,
        tau=tau,
        action_stddev=std_dev,
        render_env=render_gym
    )

    ddpg_training.save_models(SAVE_MODEL_FORMAT.format(format_gym_name(gym_name)))
    buffer.save_buffer(SAVE_MODEL_FORMAT.format(format_gym_name(gym_name)))
    
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

def load_ddpg(gym_name):
    data_folder = SAVE_MODEL_FORMAT.format(gym_name)
    try:
        actor = tf.keras.models.load_model(data_folder + "/actor")
        critic = tf.keras.models.load_model(data_folder + "/critic")
        target_actor = tf.keras.models.load_model(data_folder + "/target_actor")
        target_critic = tf.keras.models.load_model(data_folder + "/target_critic")        
        return actor, critic, target_actor, target_critic
    except OSError:
        print("No saved DDPG models found for {}".format(gym_name))
        return None

def load_histogram(gym_name, distribution_name):
    data_folder = SAVE_MODEL_FORMAT.format(gym_name)
    distribution_npzfile = np.load(data_folder + "/{}.npz".format(distribution_name))
    observed_distribution = distribution_npzfile['observed_counts']
    sampled_distribution = distribution_npzfile['sampled_counts']
    distribution_edges = distribution_npzfile['edges']
    return observed_distribution, sampled_distribution, distribution_edges
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Tensorflow DDPG example')
    parser.add_argument('gym', help='name of gym to train against')
    parser.add_argument('-t', '--train', action='store_true', help='name of gym to train against')
    parser.add_argument('-d', '--display', action='store_true', help='display results from training on provided gym')
    parser.add_argument('-e', '--episodes', default=100, type=int, help='number of episodes to train over')
    parser.add_argument('--render_gym', action='store_true', help='render gym')
    args = parser.parse_args()

    if args.train:
        train_ddpg(args.gym, render_gym=args.render_gym, total_episodes=args.episodes)
    elif args.display:
        display_ddpg_results(args.gym)
