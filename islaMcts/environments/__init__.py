from gym.envs.registration import register

register(
    id='enviroments:CurveEnv-v0',
    entry_point='CurveEnv',
)

register(
    id='CurveEnv-Discrete-v0',
    entry_point='DiscreteCurveEnv',
)
