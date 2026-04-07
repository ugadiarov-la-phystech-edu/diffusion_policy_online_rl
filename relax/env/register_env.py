import gymnasium as gym


gym.envs.registration.register(
    'LiftMedium-v0', entry_point='relax.env.robosuite_env:RobosuiteEnv', kwargs=dict(
        task='Lift', horizon=125, initialization_noise_magnitude=0.5, use_random_object_position='medium',
        raw_observation=True, obs_size=224)
)


gym.envs.registration.register(
    'ReachingHard-v0', entry_point='relax.env.cw_envs.target:CwTargetEnvGymnasium',
    kwargs=dict(obs_size=64, time_limit=100)
)
