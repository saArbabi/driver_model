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
low_density_episodes = pickup_episodes(speeds, max_speed=25, min_speed=14, episode_n=50)
medium_density_episodes = pickup_episodes(speeds, max_speed=16, min_speed=8, episode_n=50)
high_density_episodes = pickup_episodes(speeds, max_speed=8, min_speed=0, episode_n=50)

data_saver(medium_density_episodes, 'medium_density_test_episodes')
data_saver(low_density_episodes, 'low_density_test_episodes')
data_saver(high_density_episodes, 'high_density_test_episodes')
894,  800, 2714, 2636, 1694, 2872, 1281, 1793, 2577, 2173, 2840,
       1935, 1115, 2553, 1585, 2895, 1933,  895, 2546, 2521, 1669, 2026,
       2800, 2772, 2126, 1563, 1601, 2815, 2047, 1732,  640, 1952,  835,
       2870, 1251, 1618, 2625, 1515, 1841, 2584, 2000, 2691, 1962, 2344,
       2753, 1462,  839, 2891, 2423, 2805
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
hist
plt.hist(_arr[:,1], bins=50)
plt.hist(_arr[:,1], bins=50)
_arr
