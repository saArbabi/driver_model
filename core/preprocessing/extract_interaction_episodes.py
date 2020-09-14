import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from models.core.preprocessing import utils
from math import hypot
from importlib import reload

reload(utils)
cwd = os.getcwd()

# %%
"""
m_df - merge_veh_df
y_df - yield_veh_df
"""
col = ['id','frm','scenario','lane_id',
                                'bool_r','bool_l','pc','vel',
                                'a_long','act_lat','a_lat','e_class',
                                'ff_id','ff_long','ff_lat','ff_v',
                                'fl_id','fl_long','fl_lat','fl_v',
                                'bl_id','bl_long','bl_lat','bl_v',
                                'fr_id','fr_long','fr_lat','fr_v',
                                'br_id','br_long','br_lat','br_v',
                                'bb_id','bb_long','bb_lat','bb_v']

col_drop = ['bool_r','bool_l','a_lat','a_long',
                    'fr_id','fr_long','fr_lat','fr_v',
                    'fl_id','fl_long','fl_lat','fl_v']
datasets = {
        "i101_1": "trajdata_i101_trajectories-0750am-0805am.txt",
        "i101_2": "trajdata_i101_trajectories-0805am-0820am.txt",
        "i101_3": "trajdata_i101_trajectories-0820am-0835am.txt",
        "i80_1": "trajdata_i80_trajectories-0400-0415.txt",
        "i80_2": "trajdata_i80_trajectories-0500-0515.txt",
        "i80_3": "trajdata_i80_trajectories-0515-0530.txt"}

col_df_all = ['id','frm','scenario','lane_id','length','x_front','y_front','class']

# %%

feature_set = pd.read_csv('./datasets/feature_set.txt', delimiter=' ',
                        header=None, names=col).drop(col_drop,axis=1)

df_all = pd.read_csv('./datasets/df_all.txt', delimiter=' ',
                                                            header=None, names=col_df_all)

os.chdir('../NGSIM_data_and_visualisations')
import road_geometry
reload(road_geometry)

xc_80, yc_80 = road_geometry.get_centerlines('./NGSIM DATA/centerlines80.txt')
xc_101, yc_101 = road_geometry.get_centerlines('./NGSIM DATA/centerlines101.txt')

os.chdir(cwd)

# %%
def draw_traj(m_df, y_df, case_info):
    # for some vis
    fig = plt.figure()
    item = 'pc'
    plt.plot(m_df[item])
    # plt.plot(y_df[item])

    # plt.plot(y_df[item])
    indx = m_df.loc[m_df['frm'] == case_info['lc_frm']].index[0]
    plt.scatter(indx, m_df[item].iloc[indx])
    plt.title([case_info['id'], case_info['lc_frm'], case_info['scenario']])
    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])


    fig = plt.figure()
    item = 'act_lat'
    plt.plot(m_df[item])

    # plt.plot(y_df[item])
    # plt.plot(y_df[item])
    indx = m_df.loc[m_df['frm'] == case_info['lc_frm']].index[0]
    plt.scatter(indx, m_df[item].iloc[indx])

    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])

def get_glob_df(case_info):
    """
    :return: global pose of interacting cars
    Note: start_frm and end_frm are not included here. They are dropped later when
    calculating acceleations.
    """

    glob_pos = df_all.loc[(df_all['scenario'] == case_info['scenario']) &
                            (df_all['frm'] > case_info['start_frm']) &
                            (df_all['frm'] <  case_info['end_frm'])]

    return glob_pos[['id','frm','x_front','y_front', 'length']]

def get_lane_cor(scenario, lane_id):
    if scenario in ['i101_1', 'i101_2', 'i101_3']:
        xc = np.array(xc_101[int(lane_id-1)])
        yc = np.array(yc_101[int(lane_id-1)])
    else:
        xc = np.array(xc_80[int(lane_id-1)])
        yc = np.array(yc_80[int(lane_id-1)])

    return [xc, yc]

# def get_act_lat():
reload(utils)

counter = 0
for scenario in datasets:
    feat_df = feature_set.loc[(feature_set['scenario'] == scenario) &
                                        (feature_set['lane_id'] < 7)] # feat_set_scene
    ids = feat_df['id'].unique().astype('int')

    for id in ids[0:20]:

        veh_df = feat_df.loc[(feat_df['id'] == id)].reset_index(drop = True)
        lc_frms = utils.lc_entrance(veh_df)

        for lane_change in lc_frms:
            if lc_frms[lane_change]:
                for lc_frm, lane_id in lc_frms[lane_change]:
                    y_id, y_class = utils.get_vehInfo(veh_df, lc_frm, 'bb_id')
                    m_class = veh_df['e_class'].iloc[0]

                    end_frm = utils.lc_completion(veh_df, lc_frm, y_id, lane_change, lane_id)
                    start_frm = utils.lc_initation(veh_df, lc_frm-1, y_id, lane_change, lane_id)
                    frms_n = int(end_frm-start_frm)
                    m_df = utils.frmTrim(veh_df, end_frm, start_frm)
                    v_min = m_df['vel'].min()

                    if y_id and y_class == m_class == 2 and frms_n > 20 and \
                                        start_frm != 0 and end_frm != 0 and v_min > 0:

                        case_info = {
                                    'scenario':scenario,
                                    'id':id,
                                    'frms_n':frms_n,
                                    'y_id':y_id,
                                    'lc_frm':lc_frm,
                                    'start_frm':start_frm,
                                    'end_frm':end_frm,
                                    'episode_id': counter,
                                    'lc_type': -1
                                    }

                        fadj_id, _ = utils.get_vehInfo(veh_df, lc_frm, 'ff_id')
                        f_id, _ = utils.get_vehInfo(veh_df, lc_frm-1, 'ff_id')
                        lane_cor = get_lane_cor(scenario, lane_id)

                        m_df = utils.get_m_features(m_df, case_info)
                        o_df = utils.frmTrim(feat_df, end_frm, start_frm) # other vehicles' df

                        glob_pos = get_glob_df(case_info)

                        y_df = utils.get_o_df(o_df, y_id, case_info['episode_id'])
                        m_df, y_df = utils.get_dxdv(glob_pos, m_df, y_df, lane_cor, 'front')

                        if fadj_id:
                            fadj_df = utils.get_o_df(o_df, fadj_id, case_info['episode_id'])
                            m_df, fadj_df = utils.get_dxdv(glob_pos, m_df, fadj_df, lane_cor, 'behind')

                        if f_id:
                            f_df = utils.get_o_df(o_df, f_id, case_info['episode_id'])
                            m_df, f_df = utils.get_dxdv(glob_pos, m_df, f_df, lane_cor, 'behind')

                        mveh_size = len(m_df)

                        if not fadj_id:
                            fadj_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                        else:
                            fadj_df = utils.applyCorrections(m_df, fadj_df, 'na', mveh_size)

                        if not f_id:
                            f_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                        else:
                            f_df = utils.applyCorrections(m_df, f_df, 'na', mveh_size)

                        y_df = utils.applyCorrections(m_df, y_df, 'yveh', mveh_size)

                        utils.data_saver(m_df, 'm_df')
                        utils.data_saver(y_df, 'y_df')
                        utils.data_saver(f_df, 'f_df')
                        utils.data_saver(fadj_df, 'fadj_df')

                        counter += 1
                        print(counter, ' ### lane change extracted ###')
                        draw_traj(m_df, y_df, case_info)


# %%
m_df = feature_set.loc[(feature_set['scenario'] == 'i80_1') &
                                    (feature_set['id'] == 103) &
                                    (feature_set['frm'] >= 1077) &
                                    (feature_set['frm'] <= 1162) &
                                    (feature_set['lane_id'] < 7)] # feat_set_scene

plt.plot(m_df['pc'])
plt.plot(m_df['bl_id'])
plt.plot(y_df['dx'])

# %%



case_info
y_df.columns

m_df.columns
