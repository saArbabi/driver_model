"""
- clip outliers/unrealistic feature values
- Randomly sample episodes for validation and training
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'pc',
       'gap_size', 'dx', 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']

yveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'act_long_p', 'act_long']

mveh_df = pd.read_csv('./driver_model/datasets/mveh_df.txt', delimiter=' ',
                        header=None, names=mveh_col)
yveh_df = pd.read_csv('./driver_model/datasets/yveh_df.txt', delimiter=' ',
                        header=None, names=yveh_col)

mveh_df.loc[mveh_df['act_long']<-3, 'act_long'] = -3
mveh_df.loc[mveh_df['act_long']>3, 'act_long'] = 3
mveh_df.loc[mveh_df['act_long_p']<-3, 'act_long_p'] = -3
mveh_df.loc[mveh_df['act_long_p']>3, 'act_long_p'] = 3

mveh_df.loc[mveh_df['act_lat']<-1.5, 'act_lat'] = -1.5
mveh_df.loc[mveh_df['act_lat']>1.5, 'act_lat'] = 1.5
mveh_df.loc[mveh_df['act_lat_p']<-1.5, 'act_lat_p'] = -1.5
mveh_df.loc[mveh_df['act_lat_p']>1.5, 'act_lat_p'] = 1.5

mveh_df.loc[mveh_df['gap_size']==70, 'gap_size'] = 100
mveh_df.loc[mveh_df['gap_size']>100, 'gap_size'] = 100
mveh_df.loc[mveh_df['dx']>70, 'dx'] = 70
mveh_df.loc[mveh_df['pc']<-2.5, 'pc'] = -2.5
mveh_df.loc[mveh_df['pc']>2.5, 'pc'] = 2.5

yveh_df.loc[yveh_df['act_long']<-3, 'act_long'] = -3
yveh_df.loc[yveh_df['act_long']>3, 'act_long'] = 3
yveh_df.loc[yveh_df['act_long_p']<-3, 'act_long_p'] = -3
yveh_df.loc[yveh_df['act_long_p']>3, 'act_long_p'] = 3


feature_col = ['vel', 'act_long', 'act_lat', 'act_long_p', 'act_lat_p', 'gap_size', 'pc', 'dx']

mveh_df['act_lat'].plot.hist(bins=125)
for feature in feature_col:
    plt.figure()
    mveh_df[feature].plot.hist(bins=125)
    plt.title(feature)

all_episodes = mveh_df['episode_id'].unique()
training_episodes = list(np.random.choice(all_episodes, int(0.8*len(all_episodes))))
validation_episodes = list(set(all_episodes).symmetric_difference(set(training_episodes)))


def save_list(my_list, name):
    file_name = './driver_model/datasets/'+name+'.txt'

    with open(file_name, "w") as f:
        for item in my_list:
            f.write("%s\n"%item)

def data_saver(mveh_df, yveh_df):

    mveh_df.to_csv('./driver_model/datasets/mveh_df0.txt',
                                    header=None, index=None, sep=' ', mode='a')
    yveh_df.to_csv('./driver_model/datasets/yveh_df0.txt',
                                    header=None, index=None, sep=' ', mode='a')

data_saver(mveh_df, yveh_df)
save_list(training_episodes, 'training_episodes')
save_list(validation_episodes, 'validation_episodes')
