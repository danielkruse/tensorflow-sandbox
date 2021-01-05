import argparse
import gym
import tensorflow as tf
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import custom_tf_ddpg

GYM_NAMES_1D = [
    'custom_gym:custom-1d-gym-v0'
]
GYM_NAMES_2D = [
    'custom_gym:custom-2d-gym-v0', 
    'custom_gym:custom-2d-obstacles-gym-v0'
]
GYM_NAMES_CLASSIC_CONTROL = [
    'Pendulum-v0'
]
VIABLE_GYM_NAMES = GYM_NAMES_1D + GYM_NAMES_2D + GYM_NAMES_CLASSIC_CONTROL
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

def display_1d_actor_and_critic(actor, critic, env, state_histogram, reward_histogram):
    # state data ranges to evaluate over
    xmax, = env.observation_space.high
    xmin, = env.observation_space.low
    xdisp = np.linspace(xmin, xmax, 50)

    # action data ranges to evaluate over
    umax, = env.action_space.high
    umin, = env.action_space.low
    udisp = np.linspace(umin, umax, 25)

    # evaluate the Q values over X, U grid
    Ugrid, Xgrid = np.meshgrid(udisp, xdisp)
    Qval = critic([Xgrid.flatten(), Ugrid.flatten()])
    Qval = np.reshape(Qval, np.shape(Xgrid))
    
    # extract the optimal u values and corresponding V, A terms from Q
    maxQ_idxs = np.argmax(Qval, axis=1)
    # Value is the maximum Q-value for each state
    Vval = np.array([Qval[i, max_idx] for i, max_idx in enumerate(maxQ_idxs)])
    # Extract the action that generates the maximum Q
    uval_critic = udisp[maxQ_idxs]
    # Calculate Advantage as Q - V
    Aval = Qval - np.transpose(np.tile(Vval, (np.size(udisp), 1)))

    # Generate actions according to actor
    uval_actor = actor(xdisp)
    
    # Figure showing resulting critic Q and A surfaces
    fig1 = plt.figure()
    gs1 = fig1.add_gridspec(1, 2)
    # 3D surface showing Q-values as a function of state and action
    f1_ax1 = fig1.add_subplot(gs1[0, 0], projection='3d')
    f1_ax1.plot_surface(Xgrid, Ugrid, Qval)
    f1_ax1.set(xlabel='X', ylabel='U', zlabel='Q', title='Quality')
    f1_ax1.grid()
    # 3D surface showing Advantage as a function of state and action
    f1_ax2 = fig1.add_subplot(gs1[0, 1], projection='3d')
    f1_ax2.plot_surface(Xgrid, Ugrid, Aval)
    f1_ax2.set(xlabel='X', ylabel='U', zlabel='A', title='Advantage')
    f1_ax2.grid()

    # Figure showing actor and critic results
    fig2 = plt.figure()
    gs2 = fig2.add_gridspec(2, 1)

    # Actor with plot of action vs state
    f2_ax1 = fig2.add_subplot(gs2[0, 0])
    f2_ax1.plot(xdisp, uval_actor)
    f2_ax1.set(ylabel='U', title='actor')
    f2_ax1.grid()

    # Critic with y-y plot showing Value and Optimal Action according to the critic
    # set sharex to match actor plot
    f2_ax2_y1 = fig2.add_subplot(gs2[1, 0], sharex=f2_ax1) 
    # left-y set to blue for action that maximizes value
    f2_ax2_y1.plot(xdisp, uval_critic, color='blue')
    f2_ax2_y1.set(xlabel='X', title='critic')
    f2_ax2_y1.set_ylabel('U', color='blue')
    f2_ax2_y1.tick_params(axis='y', labelcolor='blue')
    f2_ax2_y1.grid()
    # right-y set to green for maximum value
    f2_ax2_y2 = f2_ax2_y1.twinx()
    f2_ax2_y2.plot(xdisp, Vval, color='green')
    f2_ax2_y2.set_ylabel('V', color='green')
    f2_ax2_y2.tick_params(axis='y', labelcolor='green')
    f2_ax2_y2.grid()

    # Set to tight layout to avoid clipping right-y axis
    fig2.tight_layout()

    # Convert histogram tuple to PDFs
    observed_state_distribution, sampled_state_distribution, state_bin_centers = histogram_to_distribution(state_histogram)
    observed_reward_distribution, sampled_reward_distribution, reward_bin_centers = histogram_to_distribution(reward_histogram)
    
    # Collect features of PDF for display convenience
    state_bin_width = np.mean(state_bin_centers[0][1:] - state_bin_centers[0][:-1])
    reward_bin_width = np.mean(reward_bin_centers[0][1:] - reward_bin_centers[0][:-1])

    # Figure to show variation between observed and sampled data
    fig3 = plt.figure()
    gs3 = fig1.add_gridspec(2, 1)
    
    # Overlaid histograms to show observed and off-policy sampled states
    f3_ax1 = fig3.add_subplot(gs3[0, 0])
    f3_ax1.bar(state_bin_centers[0], observed_state_distribution, state_bin_width, alpha=0.4, label='observed')
    f3_ax1.bar(state_bin_centers[0], sampled_state_distribution, state_bin_width, alpha=0.4, label='sampled')
    f3_ax1.set(xlabel='X', ylabel='frequency of observance', title='state distribution')
    f3_ax1.legend()

    # Overlaid histograms to show observed and off-policy sampled rewards
    f3_ax2 = fig3.add_subplot(gs3[1, 0])
    f3_ax2.bar(reward_bin_centers[0], observed_reward_distribution, reward_bin_width, alpha=0.4, label='observed')
    f3_ax2.bar(reward_bin_centers[0], sampled_reward_distribution, reward_bin_width, alpha=0.4, label='sampled')
    f3_ax2.set(xlabel='reward', ylabel='frequency of observance', title='reward distribution')
    f3_ax2.legend()

    # Set to tight layout to prevent clipping
    fig3.tight_layout()


def display_2d_actor_and_critic(actor, critic, env, state_histogram, reward_histogram):
    # state data ranges to evaluate over
    xmax, ymax = env.observation_space.high
    xmin, ymin = env.observation_space.low
    xdisp = np.linspace(xmin, xmax, 20)
    ydisp = np.linspace(ymin, ymax, 20)
    # Generate state grid
    Xgrid, Ygrid = np.meshgrid(xdisp, ydisp)
    state_grid_tf = tf.convert_to_tensor(np.transpose([Xgrid.flatten(), Ygrid.flatten()]))

    # action data ranges to evaluate over
    #  (reconstructing x-y actions as direction and magnitude for future quiver usage)
    uxmax, uymax = env.action_space.high
    uxmin, uymin = env.action_space.low
    uthetadisp = np.linspace(0, 2*np.pi, 16)
    umagdisp = np.linspace(0, 1.0, 10)
    Umaggrid, Uthetagrid = np.meshgrid(umagdisp, uthetadisp)
    Uxgrid = 0.5*(uxmax + uxmin) + 0.5*(uxmax - uxmin) * Umaggrid * np.cos(Uthetagrid)
    Uygrid = 0.5*(uymax + uymin) + 0.5*(uymax - uymin) * Umaggrid * np.sin(Uthetagrid)
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
            # evaluate Q-values for state-action pairs
            Qxy = critic([state_xy_tf, action_grid_tf])
            Qxy = np.reshape(Qxy, np.shape(Uxgrid))
            # Identify maximum Q-value to generate A, V, u
            maxQ_idx = np.argmax(Qxy)
            Qval_critic[iy, ix, :, :] = Qxy
            # V is the maximum Q for state s
            Vval_critic[iy, ix] = Qxy.flatten()[maxQ_idx]
            # A is the Q - V for state-action pairs
            Aval_critic[iy, ix, :, :] = Qxy - Vval_critic[iy, ix]
            # u is the action that maximizes Q
            uxval_critic[iy, ix] = Uxgrid.flatten()[maxQ_idx]
            uyval_critic[iy, ix] = Uygrid.flatten()[maxQ_idx]

    # evaluate action values from actor
    uval_actor = actor(state_grid_tf)
    uxval_actor = np.reshape(uval_actor[:, 0].numpy(), np.shape(Xgrid))
    uyval_actor = np.reshape(uval_actor[:, 1].numpy(), np.shape(Ygrid))
    
    # Figure to show generated actor and critic results
    fig1 = plt.figure()
    gs1 = fig1.add_gridspec(1, 2)

    # Critic results displayed as contour plot for value and quiver for max advantage
    f1_ax1 = fig1.add_subplot(gs1[0, 0])
    if env.obstacles:
        for pos, rad in env.obstacles:
            obs_circle = plt.Circle(pos, rad, facecolor='blue', edgecolor='red', alpha=0.4)
            f1_ax1.add_patch(obs_circle)

    f1_ax1.contour(Xgrid, Ygrid, Vval_critic, 10)
    f1_ax1.quiver(Xgrid, Ygrid, uxval_critic, uyval_critic, pivot='mid')
    f1_ax1.set(xlabel='X', ylabel='Y', title='target critic')
    f1_ax1.grid()

    # Actor results display as quiver
    f1_ax2 = fig1.add_subplot(gs1[0, 1])
    if env.obstacles:
        for pos, rad in env.obstacles:
            obs_circle = plt.Circle(pos, rad, facecolor='blue', edgecolor='red', alpha=0.4)
            f1_ax2.add_patch(obs_circle)
    f1_ax2.quiver(Xgrid, Ygrid, uxval_actor, uyval_actor, pivot='mid')
    f1_ax2.set(xlabel='X', ylabel='Y', title='target actor')
    f1_ax2.grid()

    # Convert histograms to PDFs
    observed_state_distribution, sampled_state_distribution, state_bin_centers = histogram_to_distribution(state_histogram)
    observed_reward_distribution, sampled_reward_distribution, reward_bin_centers = histogram_to_distribution(reward_histogram)

    # Collect bounds and widths of data bins that are helpful for plotting and axis labels
    state_bnds = (state_bin_centers[0][0], state_bin_centers[0][-1], state_bin_centers[-1][0], state_bin_centers[0][-1])
    reward_bin_width = np.mean(reward_bin_centers[0][1:] - reward_bin_centers[0][:-1])
    max_state_dist = np.max([observed_state_distribution, sampled_state_distribution])
    min_state_dist = np.min([observed_state_distribution, sampled_state_distribution])

    # Figure showing distribution of sampling in state and reward spaces
    # States are displayed as heatmap images, Reward as a histogram
    fig2 = plt.figure()
    gs2 = fig2.add_gridspec(2, 2)

    # Heatmap image of observed state distribution
    f2_ax1 = fig2.add_subplot(gs2[0, 0])
    # generate heatmap image - use extent to relabel axes
    obs_state_dist_imshow = f2_ax1.imshow(observed_state_distribution, extent=state_bnds, vmin=min_state_dist, vmax=max_state_dist)
    f2_ax1.set(xlabel='X', ylabel='Y', title='observed state distribution')
    fig2.colorbar(obs_state_dist_imshow, ax=f2_ax1, use_gridspec=True)

    # Heatmap image of off-policy sampled state distribution
    f2_ax2 = fig2.add_subplot(gs2[0, 1])
    # generate heatmap image - use extent to relabel axes
    smpl_state_dist_imshow = f2_ax2.imshow(sampled_state_distribution, extent=state_bnds, vmin=min_state_dist, vmax=max_state_dist)
    f2_ax2.set(xlabel='X', ylabel='Y', title='sampled state distribution')
    fig2.colorbar(smpl_state_dist_imshow, ax=f2_ax2, use_gridspec=True)
    
    # Overlaid histograms of observed and sampled reward distributions
    f2_ax3 = fig2.add_subplot(gs2[1, :])
    # Create overlaid histograms using bar charts with alpha = 0.4 for transparency
    f2_ax3.bar(reward_bin_centers[0], observed_reward_distribution, reward_bin_width, alpha=0.4, label='observed')
    f2_ax3.bar(reward_bin_centers[0], sampled_reward_distribution, reward_bin_width, alpha=0.4, label='sampled')
    f2_ax3.set(xlabel='reward', ylabel='frequency of observance', title='reward distribution')
    f2_ax3.legend()
    fig2.tight_layout()

def display_ddpg_results(gym_name):
    if gym_name not in VIABLE_GYM_NAMES:
        print("{} not a recognized gym".format(gym_name))
        print("Recognized gyms:")
        for viable_gym_name in VIABLE_GYM_NAMES:
            print("{}".format(viable_gym_name))
        return
    env = gym.make(gym_name)
    
    # Only really care about target actor/critic
    actor, critic, target_actor, target_critic = load_ddpg(format_gym_name(gym_name))
    state_histogram = load_histogram(format_gym_name(gym_name), 'state_buffer_distribution')
    reward_histogram = load_histogram(format_gym_name(gym_name), 'reward_buffer_distribution')

    if gym_name in GYM_NAMES_1D:
        display_1d_actor_and_critic(
            target_actor, 
            target_critic, 
            env, 
            state_histogram, 
            reward_histogram
        )
    elif gym_name in GYM_NAMES_2D:
        display_2d_actor_and_critic(
            target_actor, 
            target_critic, 
            env, 
            state_histogram, 
            reward_histogram
        )
    else:
        print("Don't currently have a data visualization for the gym {}".format(gym_name))
        return
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

    # Custom buffer to store environment tuples for state, action, reward, next state
    buffer = custom_tf_ddpg.Buffer(num_states, num_actions, buffer_capacity=50000, batch_size=32)

    # Learning rate for actor-critic models
    # Using Adam optimizers (ADAptive Momentum)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Sampling strategy to use.  Can be
    #   'uniform' - all prior episodes equal weight
    #   'forgetful' - newest episodes weighted higher than older samples
    #   'equalized' - episodes inversely weighted by distribution of reward observance
    sampling_strategy = 'forgetful'

    # Training hyperparameters
    std_dev = 0.2
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.01

    ddpg_training = custom_tf_ddpg.DDPG(env, actor_model, critic_model, buffer)
    ddpg_training.compile(actor_optimizer, critic_optimizer, buffer_sampling=sampling_strategy)
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

def load_histogram(gym_name, histogram_name):
    data_folder = SAVE_MODEL_FORMAT.format(gym_name)
    histogram_npzfile = np.load(data_folder + "/{}.npz".format(histogram_name))
    observed_histogram = histogram_npzfile['observed_counts']
    sampled_histogram = histogram_npzfile['sampled_counts']
    histogram_edges = histogram_npzfile['edges']
    return observed_histogram, sampled_histogram, histogram_edges

# Expecting tuple of data (observed bin counts, sampled bin counts, bin edges)
def histogram_to_distribution(histogram_tuple):
    observed_counts, sampled_counts, bin_edges = histogram_tuple
    bin_centers = np.array([0.5*(edge[1:] + edge[:-1]) for edge in bin_edges])
    observed_distribution = observed_counts / sum(observed_counts)
    sampled_distribution = sampled_counts / sum(sampled_counts)

    return observed_distribution, sampled_distribution, bin_centers
    
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
