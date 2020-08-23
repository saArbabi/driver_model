import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from driver_model.preprocessing import utils
from importlib import reload

reload(utils)

os.getcwd()
"""
mveh_df - merge_vehicle_df
yveh_df - yield_vehicle_df
"""

def get_glob_pos(initiation_frm, completion_frm, id, scenario):
    """
    :return: global pose of interacting cars
    """
    glob_pos = df_all.loc[(df_all['scenario'] == scenario) &
                            (df_all['frm'] >= initiation_frm) &
                            (df_all['frm'] <= completion_frm)]

    return glob_pos.loc[(glob_pos['id'] == id)][['x_front','y_front','length']].values.tolist()

def get_dxpc(frm_range, mveh_glob_pos, yveh_glob_pos):

    if scenario in ['i101_1', 'i101_2', 'i101_3']:
        xc = np.array(xc_101[int(lane_id_f-1)])
        yc = np.array(yc_101[int(lane_id_f-1)])
    else:
        xc = np.array(xc_80[int(lane_id_f-1)])
        yc = np.array(yc_80[int(lane_id_f-1)])

    dx = []
    pc_ = []

    for i in range(frm_range):
        mveh_c_x = mveh_glob_pos[i][0]
        mveh_c_y = mveh_glob_pos[i][1]
        yveh_c_x = yveh_glob_pos[i][0]
        yveh_c_y = yveh_glob_pos[i][1]
        mveh_length = mveh_glob_pos[i][2]

        mveh_p = get_p([mveh_c_x,mveh_c_y], xc, yc)
        yveh_p = get_p([yveh_c_x,yveh_c_y], xc, yc)

        yveh_long = np.hypot(mveh_p[0]-yveh_p[0],mveh_p[1]-yveh_p[1])-ff_length
        if yveh_long < 0:
            yveh_long = 0

        pc_.append(get_pc(mveh_p, mveh_c_x,mveh_c_y))
        dx.append(yveh_long)

    return dx

# def get_act_lat():


for scenario in datasets:

    feat_df = feature_set.loc[(feature_set['scenario'] == scenario) &
                                        (feature_set['lane_id'] < 7)] # feat_set_scene
    ids = fss['id'].unique().astype('int')

    for id in ids:
        mveh_df = feat_df.loc[(feat_df['id'] == id)].reset_index(drop = True)
        lc_frms = lc_entrance(mveh_df)

        for lc_frm in lc_frms['right']:
            completion_frm, yveh_id = lc_completion(vehicle_df, lc_frm, yveh_name='yveh_id')
            initiation_frm = lc_initation(vehicle_df, lc_frm-1, yveh_name='br_id')

            mveh_glob_pos = get_glob_pos(initiation_frm, completion_frm, id, scenario)
            yveh_glob_pos = get_glob_pos(initiation_frm, completion_frm, yveh_id, scenario)
            dx, pc = get_dxpc(frm_range, mveh_glob_pos, yveh_glob_pos)





'frm'
'mveh_id',
'mveh_vel',
'mveh_act_long',
'mveh_act_lat',
'gap_size',
'dx',
'dv',
'mveh_pc',
'yveh_id',
'yveh_vel',
'yveh_act_long',
