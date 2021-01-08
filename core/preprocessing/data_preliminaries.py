import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

m_col = ['episode_id', 'id', 'frm', 'vel', 'pc', 'lc_type', 'act_long_z',
                                            'act_lat_z', 'act_long', 'act_lat']

o_col = ['episode_id','frm', 'exists', 'vel', 'dx', 'act_long_z', 'act_long']
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

m_df['act_long'].quantile(0.995)
m_df['act_lat'].quantile(0.995)


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

def data_saver(data, data_name):
    file_name = './datasets/' + data_name + '.csv'
    if data.dtype == 'int64':
        np.savetxt(file_name, data, fmt='%i', delimiter=',')
    else:
        np.savetxt(file_name, data, fmt='%10.3f', delimiter=',')

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
    m_arr = m_df[['episode_id', 'vel', 'pc', 'act_long','act_lat']].values
    col_o = ['vel', 'dx', 'act_long']
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
    m_arr = m_df[['episode_id', 'act_long','act_lat']].values
    y_arr = y_df[['act_long']].values
    f_arr = f_df[['act_long']].values
    fadj_arr = fadj_df[['act_long']].values
    return np.concatenate([m_arr, y_arr, f_arr, fadj_arr], axis=1)

# %%
o_trim_col = ['dx', 'act_long', 'act_long']
m_trim_col = ['act_long', 'act_lat', 'act_long', 'act_lat']

_f_df = trimStatevals(f_df, o_trim_col)
_fadj_df = trimStatevals(fadj_df, o_trim_col)
_y_df = trimStatevals(y_df, o_trim_col)
_m_df = trimStatevals(m_df, m_trim_col)
state_bool_arr = get_stateBool_arr(_m_df, _y_df, _f_df, _fadj_df)
state_real_arr = get_stateReal_arr(_m_df, _y_df, _f_df, _fadj_df)
target_arr = get_target_arr(_m_df, _y_df, _f_df, _fadj_df)
# condition_arr = get_condition_arr(_m_df, _y_df, _f_df, _fadj_df)
state_arr =  np.concatenate([state_real_arr, state_bool_arr], axis=1)
state_arr.shape
target_arr[1000]
state_arr[1000]

 # %%
state_col = ['episode_id', 'vel', 'pc', 'act_long','act_lat',
                                     'vel', 'dx', 'act_long',
                                     'vel', 'dx', 'act_long',
                                     'vel', 'dx', 'act_long',
                                     'lc_type', 'exists', 'exists', 'exists']

target_col = ['episode_id', 'act_long','act_lat',
                                            'act_long', 'act_long', 'act_long']

vis_dataDistribution(state_arr, state_col)
vis_dataDistribution(target_arr, target_col)


#%%
len(all_episodes)
spec['frm_n'].plot.hist(bins=125)
all_episodes = spec['episode_id'].values
test_episodes = spec.loc[(spec['frm_n']>40) &
                        (spec['frm_n']<50) &
                        (spec['f_id']>0) &
                        (spec['fadj_id']>0)]['episode_id'].sample(50, replace=False).values
validation_n = int(0.08*len(all_episodes))-int(len(test_episodes))
validation_episodes = spec[~spec['episode_id'].isin(test_episodes)]['episode_id'].sample(  \
                                    validation_n, replace=False).values
validation_episodes = np.append(validation_episodes, test_episodes)
validation_episodes = np.append(validation_episodes, [2895, 1289, 1037, 2870, 2400, 1344, 2872, 2266, 2765, 2215])
training_episodes = np.setdiff1d(all_episodes, validation_episodes)
len(validation_episodes)/len(training_episodes)

# %%

np.random.choice(all_episodes,), replace=False)


# %%

all_episodes[all_episodes == 2895]
test_episodes[test_episodes == 2895]
training_episodes[training_episodes == 2895]
validation_episodes[validation_episodes == 2895]

# %%


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

    # draw_traj(all_dfs, ['act_long'], episode_id)
    draw_traj(all_dfs, ['da', 'dv'], episode_id)


# %%

data_saver(state_arr, 'states_arr')
data_saver(target_arr, 'targets_arr')
# data_saver(condition_arr, 'conditions_arr')

data_saver(training_episodes, 'training_episodes')
data_saver(validation_episodes, 'validation_episodes')
data_saver(test_episodes, 'test_episodes')
