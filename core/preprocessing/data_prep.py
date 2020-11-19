"""
- Outputs training set and validation set for a particular experiment prior to
start of training.
"""
from collections import deque
import numpy as np
from sklearn.utils import shuffle
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle
from scipy.interpolate import CubicSpline
import dill
# %%
def read_data():
    global all_state_arr, all_target_arr
    global training_episodes, validation_episodes, test_episodes

    all_state_arr = np.loadtxt('./datasets/states_arr.csv', delimiter=',')
    all_target_arr = np.loadtxt('./datasets/targets_arr.csv', delimiter=',')

    training_episodes = np.loadtxt('./datasets/training_episodes.csv', delimiter=',')
    validation_episodes = np.loadtxt('./datasets/validation_episodes.csv', delimiter=',')
    test_episodes = np.loadtxt('./datasets/test_episodes.csv', delimiter=',')


read_data()
# %%
class DataPrep():
    random.seed(2020)

    def __init__(self, config, dirName):
        self.config = config['data_config']
        self.obs_n = self.config["obs_n"]
        self.pred_h = self.config["pred_h"]
        self.setScalers() # will set the scaler attributes
        self.dirName = dirName
        self.coef_scaler = {}
        os.mkdir(dirName)

    def obsSequence(self, state_arr, target_arr):
        actions = [target_arr[:, n] for n in range(5)]
        traj_len = len(state_arr)
        snip_n = 5
        if len(state_arr) > 20:

            for n in range(5):
                coefs = np.empty([traj_len-snip_n, 4]) # number of splines = knots_n - 1
                coefs[:] = np.nan

                for i in range(snip_n):
                    indx = []
                    indx.extend(np.arange(i, traj_len, snip_n))
                    traj_snippets = actions[n][indx]
                    f = CubicSpline(indx, traj_snippets)
                    coefs[indx[:-1], :] = np.stack(f.c, axis=1)[:,:] # number of splines = knots_n - 1

                coefs = coefs.tolist()
                for i in range(traj_len):
                    end_indx = i + self.obs_n - 1
                    targ_indx = [end_indx+(snip_n)*n for n in range(self.pred_h)]
                    targ_indx = [num for num in targ_indx if num < len(coefs)]

                    if targ_indx:
                        seq_len = len(targ_indx)
                        if seq_len not in self.targs:
                            self.targs[seq_len] = [[],[],[],[],[]]
                            self.conds[seq_len] = [[],[],[],[],[]]

                        cond_indx = [end_indx-snip_n]
                        cond_indx.extend(targ_indx[:-1])
                        self.targs[seq_len][n].append([coefs[num][0:1] for num in targ_indx])
                        self.conds[seq_len][n].append([coefs[num] for num in cond_indx])

                        if n == 0:
                            if seq_len not in self.states:
                                self.states[seq_len] = [state_arr[i:(i + self.obs_n), :].tolist()]
                            self.states[seq_len].append(state_arr[i:(i + self.obs_n), :].tolist())
                    else:
                        break

    def get_episode_arr(self, episode_id):
        state_arr = all_state_arr[all_state_arr[:, 0] == episode_id]
        target_arr = all_target_arr[all_target_arr[:, 0] == episode_id]
        return state_arr[:, 1:], target_arr[:, 1:]

    def applystateScaler(self, _arr):
        _arr[:, :-4] = self.state_scaler.transform(_arr[:, :-4])
        return _arr

    def setScalers(self):
        self.state_scaler = StandardScaler().fit(all_state_arr[:, 1:-4])

    def setCoefScalers(self, coefs):
        for n in range(2):
            scaler = []
            for c in range(4):
                min, max = np.quantile(coefs[4][n][:,0,c], [0.005, 0.995])
                scaler.append([min, max])
            if n == 0:
                self.coef_scaler['long'] = scaler
            else:
                self.coef_scaler['lat'] = scaler

    def concat_lat_long_action(self, data_dict):
        data_dict=dict(data_dict)
        for seq_n in range(1, 5):
            new_coefs = []
            coefs = data_dict[seq_n]
            new_coefs.append(np.concatenate([coefs[0], coefs[1]], axis=2))
            new_coefs.extend(coefs[2:])
            data_dict[seq_n] = new_coefs
        return data_dict

    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        state_arr, target_arr = self.get_episode_arr(episode_id)
        state_arr = self.applystateScaler(state_arr)
        self.obsSequence(state_arr, target_arr)

    def shuffle(self, data_dict):
        for seq_n in range(1, self.pred_h+1):
            if len(data_dict[seq_n])>1:
                # targets and conditionals
                data_dict[seq_n] = [np.array(shuffle(data_dict[seq_n][n], \
                                            random_state=2020)) for n in range(5)]
            else:
                # states
                data_dict[seq_n] = np.array(shuffle(data_dict[seq_n], random_state=2020))
        return data_dict

    def trim_scale_coefs(self, data_dict):

        for seq_n in range(1, self.pred_h+1):
            coefs = data_dict[seq_n].copy()
            if coefs[0].shape[-1]==4:
                for n in range(5):
                    # 5 actions
                    for c in range(4):
                        # 4 coefficients for cubic spline - conditionals
                        if n == 1:
                            min, max = self.coef_scaler['lat'][c]
                        else:
                            min, max = self.coef_scaler['long'][c]

                        coefs[n][:,:,c][coefs[n][:,:,c]<min]=min
                        coefs[n][:,:,c][coefs[n][:,:,c]>max]=max
                        coefs[n][:,:,c] = coefs[n][:,:,c]/max
            else:
                for n in range(5):
                    # - targets
                    if n == 1:
                        min, max = self.coef_scaler['lat'][0]
                    else:
                        min, max = self.coef_scaler['long'][0]

                    coefs[n][:,:,0:1][coefs[n][:,:,0:1]<min]=min
                    coefs[n][:,:,0:1][coefs[n][:,:,0:1]>max]=max
                    coefs[n][:,:,0:1] = coefs[n][:,:,0:1]/max

            data_dict[seq_n] = coefs
        return data_dict

    # def concat_lat_long_motion

    def pickler(self, episode_type):

        self.states = self.shuffle(self.states)
        self.targs = self.shuffle(self.targs)
        self.conds = self.shuffle(self.conds)

        if not self.coef_scaler:
            self.setCoefScalers(self.conds)

        self.conds = self.trim_scale_coefs(self.conds)
        self.targs = self.trim_scale_coefs(self.targs)
        self.conds = self.concat_lat_long_action(self.conds)
        self.targs = self.concat_lat_long_action(self.targs)


        if episode_type == 'validation_episodes':
            with open(self.dirName+'/states_val', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_val', "wb") as f:
                pickle.dump(self.targs, f)

            with open(self.dirName+'/conditions_val', "wb") as f:
                pickle.dump(self.conds, f)

            delattr(self, 'states')
            delattr(self, 'targs')
            delattr(self, 'conds')

        elif episode_type == 'training_episodes':
            with open(self.dirName+'/states_train', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_train', "wb") as f:
                pickle.dump(self.targs, f)

            with open(self.dirName+'/conditions_train', "wb") as f:
                pickle.dump(self.conds, f)

            delattr(self, 'states')
            delattr(self, 'targs')
            delattr(self, 'conds')

            # also you want to save validation arr for later use
            with open(self.dirName+'/data_obj', "wb") as f:
                dill.dump(self, f)

            with open(self.dirName+'/states_test', "wb") as f:
                _arr = all_state_arr[np.isin(all_state_arr[:, 0], test_episodes)]
                pickle.dump(_arr, f)

            with open(self.dirName+'/targets_test', "wb") as f:
                _arr = all_target_arr[np.isin(all_target_arr[:, 0], test_episodes)]
                pickle.dump(_arr, f)

    def data_prep(self, episode_type=None):
        if not episode_type:
            raise ValueError("Choose training_episodes or validation_episodes")
        self.states = {}
        self.targs = {}
        self.conds = {}

        if episode_type == 'training_episodes':
            for episode_id in training_episodes:
                self.episode_prep(episode_id)

        elif episode_type == 'validation_episodes':
            for episode_id in validation_episodes:
                self.episode_prep(episode_id)

        self.pickler(episode_type)
