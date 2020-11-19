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
                        cond_indx = [end_indx-snip_n]
                        cond_indx.extend(targ_indx[:-1])
                        seq_len = len(targ_indx)
                        if seq_len in self.targs[n]:
                            self.targs[n][seq_len].append([coefs[num] for num in targ_indx])
                            self.conds[n][seq_len].append([coefs[num] for num in cond_indx])
                        else:
                            self.targs[n][seq_len] = [[coefs[num] for num in targ_indx]]
                            self.conds[n][seq_len] = [[coefs[num] for num in cond_indx]]

                        if n == 0:
                            if seq_len in self.states:
                                self.states[seq_len].append(state_arr[i:(i + self.obs_n), :].tolist())
                            else:
                                self.states[seq_len] = [state_arr[i:(i + self.obs_n), :].tolist()]
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

    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        state_arr, target_arr = self.get_episode_arr(episode_id)
        state_arr = self.applystateScaler(state_arr)
        self.obsSequence(state_arr, target_arr)

    def shuffle(self, data_list):
        for item in data_list:
            data_list[item] = np.array(shuffle(data_list[item], random_state=2020))
        return data_list

    def trim_scale_coefs(self, coefs):
        """
        Scale the spline coefficients
        """
        for item in coefs:
            for c in range(4): #cubic polynomial
                min, max = np.quantile(coefs[item][:,0,c], [0.005, 0.995])
                coefs[item][:,:,c][coefs[item][:,:,c]<min]=min
                coefs[item][:,:,c][coefs[item][:,:,c]>max]=max
                coefs[item][:,:,c] = coefs[item][:,:,c]/max
        return coefs

    def pickler(self, episode_type):
        self.states = self.shuffle(self.states)
        self.targs = [self.shuffle(self.targs[n]) for n in range(5)]
        self.conds = [self.shuffle(self.conds[n]) for n in range(5)]
        self.targs = [self.trim_scale_coefs(self.targs[n]) for n in range(5)]
        self.conds = [self.trim_scale_coefs(self.conds[n]) for n in range(5)]

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
        self.targs = [{}, {}, {}, {}, {}] # for dict for each vehicle aciton
        self.conds = [{}, {}, {}, {}, {}]

        if episode_type == 'training_episodes':
            for episode_id in training_episodes:
                self.episode_prep(episode_id)

        elif episode_type == 'validation_episodes':
            for episode_id in validation_episodes:
                self.episode_prep(episode_id)

        self.pickler(episode_type)
