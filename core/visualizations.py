import pandas as pd
import os
import matplotlib.pyplot as plt

col = ['id', 'frm', 'scenario', 'pc', 'lc_bool', 'vel', 'act_long', 'act_lat',
       'ff_id', 'ff_vel', 'ff_long', 'ff_v', 'ff_bb_v', 'bb_id',
       'bb_vel', 'bb_long', 'bb_v']

data = pd.read_csv('./driver_model/dataset/interaction_episodes.txt', delimiter=' ',header=None, names=col)
datasets = {
                "i101_1": "trajdata_i101_trajectories-0750am-0805am.txt",
                "i101_2": "trajdata_i101_trajectories-0805am-0820am.txt",
                "i101_3": "trajdata_i101_trajectories-0820am-0835am.txt",
                "i80_1": "trajdata_i80_trajectories-0400-0415.txt",
                "i80_2": "trajdata_i80_trajectories-0500-0515.txt",
                "i80_3": "trajdata_i80_trajectories-0515-0530.txt"
        }

len(data.loc[data['lc_bool'] == 1])
len(data.loc[data['lc_bool'] == 0])
data['id'].unique()
data['pc'].min()

os.getcwd()


def draw_scene(ax):
        ax.hlines(3.7 * 2, self.road_start,
                                    self.road_end, colors='k', linestyles='solid')

        ax.hlines(3.7, road_start, road_end,
                                                        colors='k', linestyles='--')

        ax.hlines(0, road_start, road_end, colors='k', linestyles='solid')
        ax.axes.set_ylim(0, 3.7 * 2)

        ax.set_yticks([])


def draw_profile(ax, car_id, scenario, profile=['vel', 'ff_vel', 'bb_vel']):
    # adds a vehicle's trajecotry to scene

    df_id = data.loc[(data['scenario'] == scenario) & (data['id'] == car_id)]
    ax.grid()
    for item in profile:
        ax.plot(range(len(df_id)), df_id[item].values)

    ax.legend(profile)

# %%
id = 816
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

draw_profile(ax1, id, 'i101_1', profile=['vel', 'ff_vel', 'bb_vel'])
draw_profile(ax2, id, 'i101_1', profile=['pc'])
draw_profile(ax2, id, 'i101_1', profile=['lc_bool'])




# %%

ids = data['id'].unique()[0:30]
for id in ids:
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    draw_profile(ax1, id, 'i101_1', profile=['vel', 'ff_vel', 'bb_vel'])
    draw_profile(ax2, id, 'i101_1', profile=['pc'])
