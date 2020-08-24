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
col = ['id','frm','scenario','lane_id',
                                'bool_r','bool_l','pc','v_long','e_class',
                                            'ff_id','ff_long','ff_lat','ff_v',
                                            'fl_id','fl_long','fl_lat','fl_v',
                                            'bl_id','bl_long','bl_lat','bl_v',
                                            'fr_id','fr_long','fr_lat','fr_v',
                                            'br_id','br_long','br_lat','br_v',
                                            'bb_id','bb_long','bb_lat','bb_v',
                                            'a_long','v_lat','a_long_f','v_lat_f']


datasets = {
        "i101_1": "trajdata_i101_trajectories-0750am-0805am.txt",
        "i101_2": "trajdata_i101_trajectories-0805am-0820am.txt",
        "i101_3": "trajdata_i101_trajectories-0820am-0835am.txt",
        "i80_1": "trajdata_i80_trajectories-0400-0415.txt",
        "i80_2": "trajdata_i80_trajectories-0500-0515.txt",
        "i80_3": "trajdata_i80_trajectories-0515-0530.txt"}

all_states = ['id','frm','scenario','lane_id','pc','fr_id','br_id','fl_id','bl_id',
                                    'fl_long','bl_long','fr_long','br_long',
                                        'v_long','a_long','v_lat','a_lat']



drop_col = ['a_long_f','v_lat_f']
feature_set = pd.read_csv('./Driver_model/feature_extraction/feature_set2.txt', delimiter=' ',header=None, names=col).drop(drop_col,axis=1)


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
            if ():
                any missing frms, raise error.
                raise
            dx, pc = get_dxpc(frm_range, mveh_glob_pos, yveh_glob_pos)




be vary of model cheating
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
