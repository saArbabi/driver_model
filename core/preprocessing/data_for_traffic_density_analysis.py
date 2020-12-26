import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# %%
spec_col = ['episode_id', 'scenario', 'lc_frm', 'm_id', 'y_id', 'fadj_id', 'f_id',
       'frm_n']
validation_episodes = np.loadtxt('./datasets/validation_episodes.csv', delimiter=',')
all_state_arr = np.loadtxt('./datasets/states_arr.csv', delimiter=',')
all_target_arr = np.loadtxt('./datasets/targets_arr.csv', delimiter=',')
spec = pd.read_csv('./datasets/episode_spec.txt', delimiter=' ',
                                                        header=None, names=spec_col)

validation_set = all_state_arr[np.isin(all_state_arr[:, 0], validation_episodes)]
speeds = validation_set[:,0:2]
# %%
def pickup_episodes(all_speeds, max_speed, min_speed, episode_n):
    potential_episodes = speeds[np.where((speeds[:, 1]<max_speed) & (speeds[:, 1]>min_speed))[0]][:,0]
    possible_episodes  = spec.loc[spec['frm_n']>40]['episode_id'].values
    episodes = possible_episodes[np.isin(possible_episodes, np.unique(potential_episodes))]

    return np.random.choice(episodes, episode_n, replace=False)

def data_saver(data, data_name):
    file_name = './datasets/' + data_name + '.csv'
    if data.dtype == 'int64':
        np.savetxt(file_name, data, fmt='%i', delimiter=',')
    else:
        np.savetxt(file_name, data, fmt='%10.3f', delimiter=',')


# %%
high_density_episodes = pickup_episodes(speeds, max_speed=25, min_speed=14, episode_n=50)
medium_density_episodes = pickup_episodes(speeds, max_speed=16, min_speed=8, episode_n=50)
low_density_episodes = pickup_episodes(speeds, max_speed=8, min_speed=0, episode_n=50)

data_saver(high_density_episodes, 'high_density_test_episodes')
data_saver(medium_density_episodes, 'medium_density_test_episodes')
data_saver(low_density_episodes, 'low_density_test_episodes')
# %%
file_name = './datasets/' + 'high_density_states_test'
with open(file_name, "wb") as f:
    _arr = all_state_arr[np.isin(all_state_arr[:, 0], high_density_episodes)]
    pickle.dump(_arr, f)
file_name = './datasets/' + 'high_density_targets_test'
with open(file_name, "wb") as f:
    _arr = all_target_arr[np.isin(all_target_arr[:, 0], high_density_episodes)]
    pickle.dump(_arr, f)

file_name = './datasets/' + 'medium_density_states_test'
with open(file_name, "wb") as f:
    _arr = all_state_arr[np.isin(all_state_arr[:, 0], medium_density_episodes)]
    pickle.dump(_arr, f)
file_name = './datasets/' + 'medium_density_targets_test'
with open(file_name, "wb") as f:
    _arr = all_target_arr[np.isin(all_target_arr[:, 0], medium_density_episodes)]
    pickle.dump(_arr, f)

file_name = './datasets/' + 'low_density_states_test'
with open(file_name, "wb") as f:
    _arr = all_state_arr[np.isin(all_state_arr[:, 0], low_density_episodes)]
    pickle.dump(_arr, f)
file_name = './datasets/' + 'low_density_targets_test'
with open(file_name, "wb") as f:
    _arr = all_target_arr[np.isin(all_target_arr[:, 0], low_density_episodes)]
    pickle.dump(_arr, f)
# %%
