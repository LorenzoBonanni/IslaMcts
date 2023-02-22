from gymnasium.envs.registration import register

register(
    id='CurveEnv-v0',
    entry_point='islaMcts.environments.curve_env:CurveEnv',
)

register(
    id='CurveEnv-Discrete-v0',
    entry_point='islaMcts.environments.curve_env:DiscreteCurveEnv',
)
