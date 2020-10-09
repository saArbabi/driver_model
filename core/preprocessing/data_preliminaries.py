"""
- Randomly sample episodes for validation and training
- clip outliers/unrealistic feature values from the training set
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

spec['frm_n'].plot.hist(bins=125)
m_df.loc[m_df['pc']>2]['pc'].plot.hist(bins=125)

# %%
# def trimFeatureVals(veh_df)
def trimStatevals(_df, state_names):
    df = _df.copy() #only training set

    for state_name in state_names:
        if state_name == 'dx':
            df.loc[df['dx']>70, 'dx'] = 70

        else:
            min, max = df[state_name].quantile([0.005, 0.995])
            df.loc[df[state_name]<min, state_name] = min
            df.loc[df[state_name]>max, state_name] = max

    return df

def save_list(my_list, name):
    file_name = './datasets/'+name+'.txt'
    with open(file_name, "w") as f:
        for item in my_list:
            f.write("%s\n"%item)

def data_saver(veh_df, o_name):
    file_name = './datasets/' + o_name + '.txt'
    veh_df.to_csv(file_name, header=None, index=None, sep=' ', mode='a')

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

def vis_trajs(n_traj, episodes, lc_type):
    plt.figure()

    for episode_id in episodes[0:n_traj]:
        df = get_episode_df(m_df, episode_id)
        if df.iloc[0]['lc_type'] == lc_type:
            x=[0]
            y=[0]
            for i in range(len(df)):
                x.append(x[-1]+df.iloc[i]['vel']*0.1)
                y.append(y[-1]+df.iloc[i]['act_lat']*0.1)
            plt.plot(x, y)

def vis_dataDistribution(_df, state_names):
    for state_name in state_names:
        plt.figure()
        _df[state_name].plot.hist(bins=125)
        plt.title(state_name)

def get_ffadjstate_df(f_df, fadj_df):
    """These remain ffadj during state propagation.
    """
    # f_df = f_df[['episode_id', 'dv', 'dx', 'da', 'a_ratio']]
    # fadj_df = fadj_df[['dv', 'dx', 'da', 'a_ratio']]
    f_df = f_df[['episode_id', 'dv', 'dx', 'act_long_p']]
    fadj_df = fadj_df[['dv', 'dx', 'act_long_p']]
    return pd.concat([f_df, fadj_df], axis=1)

# %%
o_trim_col = ['dv', 'dx', 'da', 'a_ratio', 'act_long_p']
y_trim_col = ['dv', 'dx', 'da', 'a_ratio', 'act_long_p', 'act_long']
m_trim_col = ['act_long_p', 'act_lat_p', 'act_long', 'act_lat']

_f_df = trimStatevals(f_df, o_trim_col)
_fadj_df = trimStatevals(fadj_df, o_trim_col)
_y_df = trimStatevals(y_df, y_trim_col)
_m_df = trimStatevals(m_df, m_trim_col)
ffadj_df = get_ffadjstate_df(_f_df, _fadj_df)
len(_m_df)/len(m_df)
ffadj_df.shape
# %%
vis_dataDistribution(_f_df, o_trim_col)
vis_dataDistribution(_fadj_df, o_trim_col)
vis_dataDistribution(_fadj_df, o_trim_col)
vis_dataDistribution(ffadj_df, ['act_long_p'])

#%%
all_episodes = list(spec['episode_id'].values)
validation_episodes = list(np.random.choice(all_episodes, int(0.1*len(all_episodes))))
test_episodes = spec.loc[(spec['episode_id'].isin(validation_episodes)) &
                                        (spec['frm_n']>60) &
                                        (spec['f_id']>0) &
                                        (spec['fadj_id']>0)]['episode_id'].sample(50).values

len(test_episodes)
training_episodes = list(set(spec['episode_id']).symmetric_difference(set(validation_episodes)))
len(validation_episodes)/len(training_episodes)
# %%


vis_trajs(80, training_episodes, -1)

# %%



# %%
for episode_id in all_episodes[0:5]:
    all_dfs = []
    all_dfs.append(get_episode_df(y_df, episode_id))
    all_dfs.append(get_episode_df(f_df, episode_id))
    all_dfs.append(get_episode_df(fadj_df, episode_id))
    # all_dfs.append(get_episode_df(m_df, episode_id))

    # draw_traj(all_dfs, ['act_long_p'], episode_id)
    draw_traj(all_dfs, ['da', 'dv'], episode_id)


# %%
data_saver(_m_df, 'm_df0')
data_saver(_y_df, 'y_df0')
data_saver(ffadj_df, 'ffadj_df0')

save_list(training_episodes, 'training_episodes')
save_list(validation_episodes, 'validation_episodes')
save_list(test_episodes, 'test_episodes')
