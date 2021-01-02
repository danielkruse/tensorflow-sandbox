import gym
import numpy as np

class Custom2DGym(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_speed_norm = 2.0
        self.max_speed = np.array([self.max_speed_norm, self.max_speed_norm])
        self.max_position = np.array([5.0, 5.0])
        self.dt = 0.05
        self.viewer = None

        self.action_space = gym.spaces.Box(
            low=-self.max_speed,
            high=self.max_speed,
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-self.max_position,
            high=self.max_position,
            dtype=np.float32
        )

        self.seed()
    
    # random generator initialization
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    # expecting command as 2D np array point value
    def step(self, u):
        u = np.clip(np.squeeze(u), -self.max_speed, self.max_speed)
        xp1 = self.state + u * self.dt
        xp1 = np.clip(xp1, -self.max_position, self.max_position)

        costs = np.dot(self.state, self.state) + 0.01 * np.dot(u, u)
        # for rendering
        self.last_u = u

        self.state = xp1

        # return state, rewards, terminal, info
        return self._get_obs(), -costs, False, {}
    
    def _get_obs(self):
        return self.state
    
    # reset state and return new initial observation
    def reset(self):
        self.state = self.np_random.uniform(low=-self.max_position, high=self.max_position)
        self.last_u = None
        return self._get_obs()
    
    def render(self, mode='human'):
        if self.viewer is None:
            # Initialize gym rendering
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            view_xmax, view_ymax = 1.1 * self.max_position
            self.viewer.set_bounds(-view_xmax, view_xmax, -view_ymax, view_ymax)
            # vertical and horizontal lines for center
            vertical_line = rendering.Line((0., -view_ymax), (0., view_ymax))
            horizontal_line = rendering.Line((-view_xmax, 0.), (view_xmax, 0.))
            self.viewer.add_geom(vertical_line)
            self.viewer.add_geom(horizontal_line)
            # add body to represent state
            body = rendering.make_circle(0.25)
            body.set_color(0.3, 0.8, 0.3)
            self.body_transform = rendering.Transform()
            body.add_attr(self.body_transform)
            self.viewer.add_geom(body)
            # add body to represent action
            line = rendering.make_capsule(1.0, 0.2)
            line.set_color(0.8, 0.3, 0.8)
            self.line_transform = rendering.Transform()
            line.add_attr(self.line_transform)
            self.viewer.add_geom(line)
        
        # Update bodies for current state/action
        x, y = self.state

        # display state for body transform
        self.body_transform.set_translation(x, y)
        # display state for action transform
        if not self.last_u is None:

            u_mag = np.linalg.norm(self.last_u)
            ux, uy = self.last_u
            u_theta = np.arctan2(uy, ux)
            self.line_transform.set_scale(u_mag / self.max_speed_norm, 1.0)
            self.line_transform.set_rotation(u_theta)
            self.line_transform.set_translation(x + 0.25 * np.cos(u_theta), y + 0.25 * np.sin(u_theta))
        else:
            self.line_transform.set_scale(0.0, 0.0)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None