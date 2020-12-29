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
# %%
def pickup_episodes(validation_set, min_speed, max_speed, episode_n):
    # traffic density is roughly assumed to be indicated by the average vehicle speeds
    potential_episodes = validation_set[np.where(
                                        (validation_set[:, 1]<max_speed) &
                                        (validation_set[:, 1]>min_speed) &
                                        (validation_set[:, 5]<max_speed) &
                                        (validation_set[:, 5]>min_speed) &
                                        (validation_set[:, 11]<max_speed) &
                                        (validation_set[:, 11]>min_speed)
                                        )[0]][:,0]

    possible_episodes  = spec.loc[(spec['frm_n']>40) &
                                    (spec['y_id']!=0) &
                                    (spec['f_id']!=0) &
                                    (spec['fadj_id']!=0)]['episode_id'].values


    episodes = possible_episodes[np.isin(possible_episodes, np.unique(potential_episodes))]
    print(episodes.shape)

    return np.random.choice(episodes, episode_n, replace=False)

def data_saver(data, data_name):
    file_name = './datasets/' + data_name + '.csv'
    if data.dtype == 'int64':
        np.savetxt(file_name, data, fmt='%i', delimiter=',')
    else:
        np.savetxt(file_name, data, fmt='%10.3f', delimiter=',')

# %%
low_density_episodes = pickup_episodes(validation_set, min_speed=12, max_speed=25, episode_n=50)
medium_density_episodes = pickup_episodes(validation_set, min_speed=7, max_speed=14, episode_n=50)
high_density_episodes = pickup_episodes(validation_set, min_speed=0, max_speed=7, episode_n=50)

#
#
_arr = all_state_arr[np.isin(all_state_arr[:, 0], medium_density_episodes)]
plt.hist(_arr[:, 1])
_arr = all_state_arr[np.isin(all_state_arr[:, 0], low_density_episodes)]
plt.hist(_arr[:, 1])
_arr = all_state_arr[np.isin(all_state_arr[:, 0], high_density_episodes)]
plt.hist(_arr[:, 11])
# %%
data_saver(medium_density_episodes, 'medium_density_test_episodes')
data_saver(low_density_episodes, 'low_density_test_episodes')
data_saver(high_density_episodes, 'high_density_test_episodes')

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
