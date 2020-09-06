import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.getcwd()

# os.chdir('../')
# %%
def prep_rwse_data(df, step_size):
    df = df.values.round(3)
    y = []
    t = []
    i = 0
    while i < len(df):
        y.append(df[i])
        t.append(i/10)
        i += 1
    return y[0::step_size], t[0::step_size]

def figure_append(_error, t, ax):
    ax.plot(t, _error, color=)
    return ax

# %%
"""
Test plot
"""
fig = plt.figure(figsize=(8, 8))


ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

experiments_considered = [
'lag_seq_35',
'lag_seq_45',
'lag_seq_53',
'lag_seq_54',
'cov_drp_50_30_std',
'cov_drp_40_30_std',
'cov_drp_80_30_std',
'fully_connected'
]

for exp in experiments_considered:
    error_cat = 'y_err_df.txt'
    file_name = './experiments/' + exp + '/plt_data/' + error_cat
    _er = pd.read_csv(file_name, delimiter=' ', names=['None'], index_col=None)
    _error, t = prep_rwse_data(_er, 3)
    figure_append(_error,t,ax1)
# %%
error_cat = 'x_err_df.txt'
file_name = './experiments/' + 'fully_connected' + '/plt_data/' + error_cat
_er = pd.read_csv(file_name, delimiter=' ', names=['None'], index_col=None)
plt.plot(_er)

# %%




# %%
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

experiments_considered = [
# 'lag_seq_35',
# 'lag_seq_45',
# 'lag_seq_53',
# 'lag_seq_54',
'cov_drp_40_30_std',
'lag_seq_55',
'fully_connected_2'
]

file_names = [
'GRU with state masking',
'GRU no state masking',
'FF'
]
file = 'np'
""" x error """
for exp in experiments_considered:
    error_cat = 'x_err_df.txt'
    file_name = './experiments/' + exp + '/plt_data/' + error_cat
    _er = pd.read_csv(file_name, delimiter=' ', names=['None'], index_col=None)
    if exp == 'csc':
        start = _er.iloc[70][0]
        _er.iloc[-1][0]

        _er = _er.values.tolist()[0:70] + np.arange(start,end,(end-start)/9).reshape(-1,1).tolist()
        _er = pd.DataFrame(_er)
    _error, t = prep_rwse_data(_er, 3)
    figure_append(_error,t,ax1)
#################################################
yticks = np.arange(0, 8.1, 2)
xticks = np.arange(0, 7.1, 1)

ax1.set_xticklabels([])
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_ylabel('Longitudinal position $(m)$')

""" y error """
for exp in experiments_considered:
    error_cat = 'y_err_df.txt'
    file_name = './experiments/' + exp + '/plt_data/' + error_cat
    _er = pd.read_csv(file_name, delimiter=' ', names=['None'], index_col=None)
    _error, t = prep_rwse_data(_er, 3)
    figure_append(_error,t,ax2)
#################################################
yticks = np.arange(0, 2, 0.5)
ax2.set_xticklabels([])
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_ylabel('Lateral position $(m)$')

""" v error """
for exp in experiments_considered:
    error_cat = 'v_err_df.txt'
    file_name = './experiments/' + exp + '/plt_data/' + error_cat
    _er = pd.read_csv(file_name, delimiter=' ', names=['None'], index_col=None)

    if file == 'es-6_is-5':
        start = _er.iloc[70][0]
        end = _er.iloc[-1][0]

        _er = _er.values.tolist()[0:70] + np.arange(start,end,(end-start)/9).reshape(-1,1).tolist()
        _er = pd.DataFrame(_er)
    _error, t = prep_rwse_data(_er, 3)
    figure_append(_error,t,ax3)
#################################################
ax3.set_xticklabels(xticks)
ax3.set_xticks(xticks)
yticks = np.arange(0, 2.1, 0.5)
ax3.set_yticks(yticks)
ax3.set_ylabel('Longitudinal speed $(m/s)$')
ax3.set_xlabel('Horizon $(s)$')



ax1.legend(file_names)
fig.subplots_adjust(wspace=0, hspace=0.1)

# fig.savefig("rwse_time_step.png", bbox_inches='tight', dpi = 200)
