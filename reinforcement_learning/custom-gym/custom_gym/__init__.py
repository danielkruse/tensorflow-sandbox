from gym.envs.registration import register

register(
    id='custom-1d-gym-v0',
    entry_point='custom_gym.envs:Custom1DGym',
    max_episode_steps=200
)

register(
    id='custom-2d-gym-v0',
    entry_point='custom_gym.envs:Custom2DGym',
    max_episode_steps=200
)

register(
    id='custom-2d-obstacles-gym-v0',
    entry_point='custom_gym.envs:Custom2DObstaclesGym',
    max_episode_steps=200
)