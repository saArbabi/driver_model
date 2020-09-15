"""
- clip outliers/unrealistic feature values
- Randomly sample episodes for validation and training
- Add some additional features which may be useful
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

m_col = ['episode_id', 'id', 'frm', 'vel', 'pc', 'lc_type', 'act_long_p',
                                            'act_lat_p', 'act_long', 'act_lat']

y_col = ['episode_id', 'frm','vel', 'dv', 'dx', 'da', 'a_ratio', 'act_long_p', 'act_long']
o_col = ['episode_id', 'frm','dv', 'dx', 'da', 'a_ratio', 'act_long_p']

spec_col = ['episode_id', 'scenario', 'lc_frm', 'm_id', 'y_id', 'fadj_id', 'f_id',
       'frm_n']

m_df = pd.read_csv('./datasets/m_df.txt', delimiter=' ',
                                                        header=None, names=m_col)
y_df = pd.read_csv('./datasets/y_df.txt', delimiter=' ',
                                                        header=None, names=y_col)
f_df = pd.read_csv('./datasets/f_df.txt', delimiter=' ',
                                                        header=None, names=o_col)
fadj_df = pd.read_csv('./datasets/fadj_df.txt', delimiter=' ',
                                                        header=None, names=o_col)

spec = pd.read_csv('./datasets/episode_spec.txt', delimiter=' ',
                                                        header=None, names=spec_col)
# %%
spec.loc[spec['episode_id']==1]
spec.loc[spec['frm_n']>300]

plt.plot(y_df.loc[y_df['episode_id']==2]['vel'])
plt.plot(m_df.loc[m_df['episode_id']==2930]['act_lat'])
plt.grid()

plt.plot(m_df.loc[m_df['episode_id']==2930]['pc'])

f_df['dx'].min()
f_df['dx'].min()
y_df.loc[y_df['episode_id']==1]['vel']
m_df.loc[m_df['episode_id']==22]
f_df.loc[f_df['episode_id']==4]
f_df['dv'].plot.hist(bins=125)

# %%
# def trimFeatureVals(veh_df)

m_df.loc[m_df['act_long']<-3, 'act_long'] = -3
m_df.loc[m_df['act_long']>3, 'act_long'] = 3
m_df.loc[m_df['act_long_p']<-3, 'act_long_p'] = -3
m_df.loc[m_df['act_long_p']>3, 'act_long_p'] = 3

m_df.loc[m_df['act_lat']<-1.5, 'act_lat'] = -1.5
m_df.loc[m_df['act_lat']>1.5, 'act_lat'] = 1.5
m_df.loc[m_df['act_lat_p']<-1.5, 'act_lat_p'] = -1.5
m_df.loc[m_df['act_lat_p']>1.5, 'act_lat_p'] = 1.5

m_df.loc[m_df['dx']>70, 'dx'] = 70
m_df.loc[m_df['pc']<-2.5, 'pc'] = -2.5
m_df.loc[m_df['pc']>2.5, 'pc'] = 2.5

y_df.loc[y_df['act_long']<-3, 'act_long'] = -3
y_df.loc[y_df['act_long']>3, 'act_long'] = 3
y_df.loc[y_df['act_long_p']<-3, 'act_long_p'] = -3
y_df.loc[y_df['act_long_p']>3, 'act_long_p'] = 3


feature_col = ['vel', 'act_long', 'act_lat', 'act_long_p', 'pc', 'dx']

for feature in feature_col:
    plt.figure()
    m_df[feature].plot.hist(bins=125)
    plt.title(feature)

all_episodes = m_df['episode_id'].unique()
training_episodes = list(np.random.choice(all_episodes, int(0.8*len(all_episodes))))
validation_episodes = list(set(all_episodes).symmetric_difference(set(training_episodes)))

# %%
def draw_traj(all_dfs, features, episode_id):
    for item in features:
        fig = plt.figure()
        for df in all_dfs:
            plt.plot(df[item])
        plt.grid()
        # plt.legend([ 'y', 'f', 'fadj', 'm'])
        plt.legend(['y', 'f', 'fadj'])
        plt.title([item, episode_id])


def get_episode_df(veh_df, episode_id):
    return veh_df.loc[veh_df['episode_id'] == episode_id].reset_index(drop = True)



for episode_id in all_episodes[0:5]:
    all_dfs = []
    all_dfs.append(get_episode_df(y_df, episode_id))
    all_dfs.append(get_episode_df(f_df, episode_id))
    all_dfs.append(get_episode_df(fadj_df, episode_id))
    # all_dfs.append(get_episode_df(m_df, episode_id))

    # draw_traj(all_dfs, ['act_long_p'], episode_id)
    draw_traj(all_dfs, ['da', 'dv'], episode_id)


# %%


def save_list(my_list, name):
    file_name = '/datasets/'+name+'.txt'

    with open(file_name, "w") as f:
        for item in my_list:
            f.write("%s\n"%item)

def data_saver(m_df, y_df):

    m_df.to_csv('/datasets/m_df0.txt',
                                header=None, index=None, sep=' ', mode='a')
    y_df.to_csv('/datasets/y_df0.txt',
                                    header=None, index=None, sep=' ', mode='a')

data_saver(m_df, y_df)
save_list(training_episodes, 'training_episodes')
save_list(validation_episodes, 'validation_episodes')
