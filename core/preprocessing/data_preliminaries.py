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

o_col = ['episode_id','frm', 'exists', 'vel', 'dx', 'act_long_p', 'act_long']
spec_col = ['episode_id', 'scenario', 'lc_frm', 'm_id', 'y_id', 'fadj_id', 'f_id',
       'frm_n']

m_df = pd.read_csv('./datasets/m_df.txt', delimiter=' ',
                                                        header=None, names=m_col)
y_df = pd.read_csv('./datasets/y_df.txt', delimiter=' ',
                                                        header=None, names=o_col)
f_df = pd.read_csv('./datasets/f_df.txt', delimiter=' ',
                                                        header=None, names=o_col)
fadj_df = pd.read_csv('./datasets/fadj_df.txt', delimiter=' ',
                                                        header=None, names=o_col)

spec = pd.read_csv('./datasets/episode_spec.txt', delimiter=' ',
                                                        header=None, names=spec_col)

# %%
y_df['exists'].plot.hist(bins=125)
m_df.loc[m_df['pc']>2]['pc'].plot.hist(bins=125)

# %%
# def trimFeatureVals(veh_df)
def trimStatevals(_df, names):
    df = _df.copy() #only training set

    for name in names:
        if name == 'dx':
            df.loc[df['dx']>70, 'dx'] = 70

        else:
            min, max = df[name].quantile([0.005, 0.995])
            df.loc[df[name]<min, name] = min
            df.loc[df[name]>max, name] = max

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

def vis_dataDistribution(_arr, names):
    for i in range(len(names)):
        plt.figure()
        pd.DataFrame(_arr[:, i]).plot.hist(bins=125)
        plt.title(names[i])

def get_stateBool_arr(m_df, y_df, f_df, fadj_df):
    m_arr = m_df[['lc_type']].values
    y_arr = y_df[['exists']].values
    f_arr = f_df[['exists']].values
    fadj_arr = fadj_df[['exists']].values
    return np.concatenate([m_arr, y_arr, f_arr, fadj_arr], axis=1)

def get_stateReal_arr(m_df, y_df, f_df, fadj_df):
    m_arr = m_df[['episode_id', 'vel', 'pc', 'act_long_p','act_lat_p']].values
    col_o = ['vel', 'dx', 'act_long_p']
    y_arr = y_df[col_o].values
    f_arr = f_df[col_o].values
    fadj_arr = fadj_df[col_o].values
    return np.concatenate([m_arr, y_arr, f_arr, fadj_arr], axis=1)

def get_target_arr(m_df, y_df, f_df, fadj_df):
    m_arr = m_df[['episode_id', 'act_long','act_lat']].values
    y_arr = y_df[['act_long']].values
    f_arr = f_df[['act_long']].values
    fadj_arr = fadj_df[['act_long']].values
    return np.concatenate([m_arr, y_arr, f_arr, fadj_arr], axis=1)

def get_condition_arr(m_df, y_df, f_df, fadj_df):
    m_arr = m_df[['episode_id', 'act_long_p','act_lat_p']].values
    y_arr = y_df[['act_long_p']].values
    f_arr = f_df[['act_long_p']].values
    fadj_arr = fadj_df[['act_long_p']].values
    return np.concatenate([m_arr, y_arr, f_arr, fadj_arr], axis=1)

# %%
o_trim_col = ['dx', 'act_long_p', 'act_long']
m_trim_col = ['act_long_p', 'act_lat_p', 'act_long', 'act_lat']

_f_df = trimStatevals(f_df, o_trim_col)
_fadj_df = trimStatevals(fadj_df, o_trim_col)
_y_df = trimStatevals(y_df, o_trim_col)
_m_df = trimStatevals(m_df, m_trim_col)
state_bool_arr = get_stateBool_arr(_m_df, _y_df, _f_df, _fadj_df)
state_real_arr = get_stateReal_arr(_m_df, _y_df, _f_df, _fadj_df)
target_arr = get_target_arr(_m_df, _y_df, _f_df, _fadj_df)
condition_arr = get_condition_arr(_m_df, _y_df, _f_df, _fadj_df)
state_arr =  np.concatenate([state_real_arr, state_bool_arr], axis=1)
state_arr.shape
target_arr[1000]
state_arr[1000]
condition_arr[1000]
# %%
state_col = ['episode_id', 'vel', 'pc', 'act_long_p','act_lat_p',
                                     'vel', 'dx', 'act_long_p',
                                     'vel', 'dx', 'act_long_p',
                                     'vel', 'dx', 'act_long_p',
                                     'lc_type', 'exists', 'exists', 'exists']

target_col = ['episode_id', 'act_long','act_lat',
                                            'act_long', 'act_long', 'act_long']

vis_dataDistribution(state_arr, state_col)
vis_dataDistribution(target_arr, target_col)


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

data_saver(state_arr, 'states_arr')
data_saver(target_arr, 'targets_arr')

save_list(training_episodes, 'training_episodes')
save_list(validation_episodes, 'validation_episodes')
save_list(test_episodes, 'test_episodes')
