from gym.envs.registration import register

register(
    id='ReverseTruck-v0',
    entry_point='gym_ReverseTruck.envs:ReverseTruckEnv',
)
