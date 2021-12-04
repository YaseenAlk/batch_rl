import numpy as np
from pathlib import Path

import gin
gin.parse_config_file(Path(".", "rem.gin"))
#from dopamine.agents.dqn import dqn_agent
from batch_rl.baselines.replay_memory.logged_replay_buffer import OutOfGraphLoggedReplayBuffer

for heuristic in ['bba', 'bola', 'mpc', 'opt_rate', 'pess_rate', 'rate']:
    traj_file = np.load(Path(f'../../traj/{heuristic}_traj.npy'))
    print(f'loaded {heuristic}_traj.npy')
    print('\tshape: ', traj_file.shape)
    logged_buffer = OutOfGraphLoggedReplayBuffer(Path(".", f'data_{heuristic}_traj'))

    """
    For each chunk, there is an array of length (19 + 1 + 1 + 19 + 1). In order, this array consists of:

    An observation, with 19 dimensions
    - Last 5 throughput measurements
    - Last 5 chunk download times
    - Buffer level
    - Number of chunks left in video
    - Previous action
    - 6 file sizes for 6 possible encoding choices (the actions)
    The chosen action
    The reward in that state
    The next observation, with 19 dimensions
    Whether the session is over or not (always 1 at the end of the 490 chunks and 0 elsewhere)

    """


    #print(traj_file[0][0])
    #chunk_obs = tuple(traj_file[0][0])
    #prev_obs = chunk_obs[:19]
    #action = (chunk_obs[19],)
    #rew = (chunk_obs[20],)
    #next_obs = chunk_obs[21:40]
    #terminal = (chunk_obs[40],)
    #print('chunk_obs, len', chunk_obs, len(chunk_obs))
    #print('prev_obs, len', prev_obs, len(prev_obs))
    #print('action, len', action, len(action))
    #print('rew, len', rew, len(rew))
    #print('next_obs, len', next_obs, len(next_obs))
    #print('terminal, len', terminal, len(terminal))

    for session in range(traj_file.shape[0]):
        for chunk in range(traj_file.shape[1]):
            chunk_obs = tuple(traj_file[session][chunk])

            action = chunk_obs[19]
            rew = chunk_obs[20]
            next_obs = chunk_obs[21:40]
            terminal = chunk_obs[40]
            logged_buffer.add(next_obs, action, rew, terminal)
    logged_buffer.log_final_buffer()



#print(f.shape)
#obs = []
#obs[:3] = f[:3, :19]
#print([obs[i].shape for i in range(len(obs))])
#print('aaa')
#print(np.expand_dims(f[:3, 19], axis=1))
#self._obs[:self._stored_steps] = f[:self._stored_steps, :19]
#self._actions[:self._stored_steps] = np.expand_dims(f[:self._stored_steps, 19], axis=1)
#self._rewards[:self._stored_steps] = np.expand_dims(f[:self._stored_steps, 20], axis=1)
#
#self._mc_rewards[:self._stored_steps] = np.expand_dims(f[:self._stored_steps, 20], axis=1)
#self._terminals[:self._stored_steps] = np.expand_dims(f[:self._stored_steps, 40], axis=1)