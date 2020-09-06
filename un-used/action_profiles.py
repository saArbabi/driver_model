import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ast
os.getcwd()
# os.chdir('./driver_model2/lstm/last_try/rwse_results')
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
    ax.plot(t, _error)
    return ax

# %%
col_true = ['a_long_c', 'a_long_p', 'bb_a', 'bb_id', 'bb_long', 'bb_v', 'bb_v_long',
       'ff_a', 'ff_bb_v', 'ff_id', 'ff_long', 'ff_v', 'ff_v_long', 'frm', 'id',
       'lc_bool', 'pc', 'scenario', 'v_lat_c', 'v_lat_p', 'v_long']

# pred_ = './acc_/10%_dropouts2_hist_5_seq_5/'+ 'state_df' + '.txt'
df_true = pd.read_csv(true_, delimiter=' ', names=col_true)
df_pred = pd.read_csv(pred_, delimiter=' ', names=col_true)

test_col = [ 'pc', 'lc_bool', 'v_long', 'a_long', 'v_lat',
       'ff_v_long', 'ff_long','bb_v_long', 'bb_long']
# for item in test_col:
#     plt.figure()
#     plt.plot(df_true[item].values)
#     plt.plot(df_pred[item].values)
#     plt.legend(['true','prediction'])
#     plt.title(item)
    # plt.savefig(item, bbox_inches='tight', dpi = 200)

df_pred

# %%
experiments_considered = ['exp_hist_5', 'exp_hist_7','exp_hist_3','fully_connected']
fig = plt.figure(figsize=(9, 6))
t = np.arange(0,6.9,0.1)
ax1 = fig.add_subplot(221)

for exp in experiments_considered:
    error_cat = 'y_err_df.txt'
    file_name = './experiments/' + exp + '/plt_data/' + error_cat
    rwse = pd.read_csv(file_name, delimiter=' ')
    ax1.plot(t, rwse)



# %%
################################################################
################################################################
################################################################
################################################################

t = np.arange(0,7.7,0.1)
fig = plt.figure(figsize=(9, 6))
##################################
""" a_long_no_past"""
ax1 = fig.add_subplot(221)

ax1.plot(t, df_true['a_long'], color='darkorange')
ax1.plot(t, df_pred['a_long'], color='grey')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticklabels([])
ax1.set_ylabel('longitudinal acceleration ($m/s^2$)')
ax1.legend(['Ground Truth','Predicted motion'])
ax1.set_title('With $a_{t0}$')

##################################
""" a_long_with_past"""

ax2 = fig.add_subplot(222)
ax2.plot(t, df_true['a_long'], color='darkorange')
ax2.plot(t, df_pred['a_long'], color='grey')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticklabels([])
ax2.set_title('Without $a_{t0}$')

##################################
""" v_long_no_past"""

ax3 = fig.add_subplot(223)
ax3.plot(t, df_true['v_lat_c'], color='darkorange')
ax3.plot(t, df_pred['v_lat_c'], color='grey')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
xticks = np.arange(0, 9, 2)
ax3.set_xticks(xticks)
ax3.set_ylabel('lateral speed (m/s)')
ax3.set_xlabel('Horizon (s)')

##################################
""" v_long_with_past"""

ax4 = fig.add_subplot(224)
ax4.plot(t, df_true['v_lat_c'], color='darkorange')
ax4.plot(t, df_pred['v_lat_c'], color='grey')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
xticks = np.arange(0, 9, 2)
ax4.set_xticks(xticks)
ax4.set_xlabel('Horizon (s)')

# fig
# %%
################################################################
################################################################
################################################################
################################################################
""" lateral and speed profiles """

traces_ = './experiments/' + 'cov_drp_40_30_std' + '/state_traces.txt'
df_traces = pd.read_csv(traces_, delimiter=' ', names=None).values
df_traces.shape

# %%
state_traces = []

v_long_preds = []
v_lat_preds = []
with open(traces_, 'r') as infile:
    traces = infile.readlines()
    for trace in traces:
        trace = ast.literal_eval(trace)
        v_long_pred = []
        v_lat_pred = []
        for state in trace:
            v_long_pred.append(state[1])
            v_lat_pred.append(state[2])
        v_long_preds.append(v_long_pred)
        v_lat_preds.append(v_lat_pred)

# %%
##################################
##################################
##################################
##################################
##################################

t = np.arange(0,7.2,0.1)
fig = plt.figure(figsize=(11, 8))
##################################
""" v_long_traces"""
ax1 = fig.add_subplot(221)
for trace in v_long_preds:
    ax1.plot(t, trace, color='grey',lw=1)
ax1.plot(t, true_speed_prof['v_long'], color='darkorange',lw=3)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticklabels([])
ax1.set_ylabel('longitudinal speed ($ms^{-1}$)')
ax1.set_title('Propagated traces')
yticks = np.arange(5, 9, 1)
ax1.set_yticks(yticks)
# ax1.legend(['Ground Truth','Predicted motion'])
##################################

""" v_long_avg"""

ax2 = fig.add_subplot(222)
v_avg = []
err = []
for i in range(len(v_long_preds[0])):
    set = []
    for trace in v_long_preds:
        set.append(trace[i])
    v_avg.append(np.mean(set))
    err.append(np.std(set))
err = np.array(err)
v_avg = np.array(v_avg)

ax2.plot(t, v_avg, color='grey',lw=3)
ax2.fill_between(t, v_avg-err, v_avg+err, alpha=0.2)

ax2.plot(t, true_speed_prof['v_long'], color='darkorange',lw=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticklabels([])
# ax2.set_ylabel('longitudinal speed ($ms^{-1}$)')
ax2.set_title('Average')
ax2.set_yticks(yticks)

##################################
##################################

""" v_lat_traces"""
ax3 = fig.add_subplot(223)
for trace in v_lat_preds:
    ax3.plot(t, trace, color='grey',lw=1)

ax3.plot(t, true_speed_prof['v_lat_c'], color='darkorange',lw=3)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
# ax3.set_xticklabels([])
ax3.set_ylabel('lateral speed ($ms^{-1}$)')
ax3.set_xlabel('Horizon (s)')

# ax3.legend(['Ground Truth','Predicted motion'])
##################################
""" v_lat_avg"""

ax4 = fig.add_subplot(224)
v_avg = []
err = []
for i in range(len(v_lat_preds[0])):
    set = []
    for trace in v_lat_preds:
        set.append(trace[i])
    v_avg.append(np.mean(set))
    err.append(np.std(set))
err = np.array(err)
v_avg = np.array(v_avg)

ax4.plot(t, v_avg, color='grey',lw=3)
ax4.fill_between(t, v_avg-err, v_avg+err, alpha=0.2)

ax4.plot(t, true_speed_prof['v_lat_c'], color='darkorange',lw=3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
# ax4.set_xticklabels([])
ax4.set_xlabel('Horizon (s)')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("speed_profiles.png", bbox_inches='tight', dpi = 200)






















# %%
ax2 = fig.add_subplot(222)
ax2.plot(t, df_true['a_long'], color='darkorange')
ax2.plot(t, df_pred['a_long'], color='grey')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticklabels([])
ax2.set_title('Without $a_{t0}$')

##################################
############"""" Avg """"###############
##################################

""" v_long_no_past"""

ax3 = fig.add_subplot(223)
ax3.plot(t, df_true['v_lat_c'], color='darkorange')
ax3.plot(t, df_pred['v_lat_c'], color='grey')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
xticks = np.arange(0, 9, 2)
ax3.set_xticks(xticks)
ax3.set_ylabel('lateral speed (m/s)')
ax3.set_xlabel('Horizon (s)')

##################################
""" v_long_with_past"""

ax4 = fig.add_subplot(224)
ax4.plot(t, df_true['v_lat_c'], color='darkorange')
ax4.plot(t, df_pred['v_lat_c'], color='grey')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
xticks = np.arange(0, 9, 2)
ax4.set_xticks(xticks)
ax4.set_xlabel('Horizon (s)')

# fig
