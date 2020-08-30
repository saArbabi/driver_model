import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from driver_model.preprocessing import utils
from math import hypot
from importlib import reload

reload(utils)
cwd = os.getcwd()

# %%
"""
mveh_df - merge_vehicle_df
yveh_df - yield_vehicle_df
"""
col = ['id','frm','scenario','lane_id',
                                'bool_r','bool_l','pc','v_long',
                                'a_long','v_lat','a_lat','e_class',
                                'ff_id','ff_long','ff_lat','ff_v',
                                'fl_id','fl_long','fl_lat','fl_v',
                                'bl_id','bl_long','bl_lat','bl_v',
                                'fr_id','fr_long','fr_lat','fr_v',
                                'br_id','br_long','br_lat','br_v',
                                'bb_id','bb_long','bb_lat','bb_v']

col_drop = ['bool_r','bool_l','a_lat',
                    'fr_id','fr_long','fr_lat','fr_v',
                    'fl_id','fl_long','fl_lat','fl_v']
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

col_df_all = ['id','frm','scenario','lane_id','length','x_front','y_front','class']

# %%

feature_set = pd.read_csv('./driver_model/datasets/feature_set.txt', delimiter=' ',
                        header=None, names=col).drop(col_drop,axis=1)

df_all = pd.read_csv('./driver_model/datasets/df_all.txt', delimiter=' ',
                                                            header=None, names=col_df_all)

os.chdir('../NGSIM_data_and_visualisations')
import road_geometry
reload(road_geometry)

xc_80, yc_80 = road_geometry.get_centerlines('./NGSIM DATA/centerlines80.txt')
xc_101, yc_101 = road_geometry.get_centerlines('./NGSIM DATA/centerlines101.txt')

os.chdir(cwd)
df_all['length'].mean()
# %%

def get_glob_df(case_info):
    """
    :return: global pose of interacting cars
    """

    glob_pos = df_all.loc[(df_all['scenario'] == case_info['scenario']) &
                            (df_all['frm'] >= case_info['initiation_frm']) &
                            (df_all['frm'] <=  case_info['completion_frm'])]

    return glob_pos[['id','frm','x_front','y_front','length']]

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

test_list = []
yveh_names = {'v_long':'vel','a_long':'act_long'}
counter = 0
for scenario in datasets:

    feat_df = feature_set.loc[(feature_set['scenario'] == scenario) &
                                        (feature_set['lane_id'] < 7)] # feat_set_scene
    ids = feat_df['id'].unique().astype('int')


    for id in ids:

        veh_df = feat_df.loc[(feat_df['id'] == id)].reset_index(drop = True)
        lc_frms = utils.lc_entrance(veh_df)

        for lc_frm, lane_id in lc_frms['right']:

            yveh_id, yveh_class = utils.get_yveh(veh_df, lc_frm)
            lane_cor = get_lane_cor(scenario, lane_id)
            veh_class = veh_df['e_class'].iloc[0]

            if yveh_id and yveh_class == veh_class == 2:
                completion_frm = utils.lc_completion(veh_df, lc_frm, yveh_id, lane_id)
                initiation_frm = utils.lc_initation(veh_df, lc_frm-1, yveh_id, 'right', lane_id)

                mveh_df = veh_df.loc[(veh_df['frm'] >= initiation_frm) &
                            (veh_df['frm'] <= completion_frm)].reset_index(drop = True)

                yveh_df = feat_df.loc[(feat_df['id'] == yveh_id) &
                            (feat_df['frm'] >= initiation_frm) &
                    (feat_df['frm'] <= completion_frm)].reset_index(drop = True)

                frm_range = int(completion_frm-initiation_frm)
                if frm_range > 20 and initiation_frm != 0 and completion_frm != 0:

                    case_info = {
                    'scenario':scenario,
                    'id':id,
                    'frm_range':frm_range,
                    'yveh_id':yveh_id,
                    'lc_frm':lc_frm,
                    'initiation_frm':initiation_frm,
                    'completion_frm':completion_frm,
                    'episode_id':'r' + str(counter),

                    }

                    if all(mveh_df['frm'].diff().dropna() != 1):
                        raise ValueError("There are missing frames", case_info)

                    yveh_df = yveh_df[['scenario','frm','id','v_long','a_long']].rename(columns=yveh_names)

                    glob_pos = get_glob_df(case_info)
                    mveh_glob_pos = utils.get_vehglob_pos(glob_pos, id)
                    yveh_glob_pos = utils.get_vehglob_pos(glob_pos, yveh_id)

                    dx = utils.get_dx(mveh_glob_pos, yveh_glob_pos, case_info, lane_cor)
                    gap_size = utils.get_gap_size(mveh_df, case_info, glob_pos, lane_cor)

                    mveh_df, yveh_df = utils.get_veh_feats(mveh_df, yveh_df, gap_size, dx, case_info['episode_id'])

                    test_list.append([case_info, gap_size])
                    counter += 1
                    case_info['gap_size'] = gap_size
                    print(counter, ' ### lane change extracted ###')

                    # draw_traj(mveh_df, yveh_df, case_info)
                    utils.data_saver(mveh_df, yveh_df)

        for lc_frm, lane_id in lc_frms['left']:

            yveh_id, yveh_class = utils.get_yveh(veh_df, lc_frm)
            lane_cor = get_lane_cor(scenario, lane_id)
            veh_class = veh_df['e_class'].iloc[0]

            if yveh_id and yveh_class == veh_class == 2:
                completion_frm = utils.lc_completion(veh_df, lc_frm, yveh_id, lane_id)
                initiation_frm = utils.lc_initation(veh_df, lc_frm-1, yveh_id, 'left', lane_id)

                mveh_df = veh_df.loc[(veh_df['frm'] >= initiation_frm) &
                            (veh_df['frm'] <= completion_frm)].reset_index(drop = True)

                yveh_df = feat_df.loc[(feat_df['id'] == yveh_id) &
                            (feat_df['frm'] >= initiation_frm) &
                    (feat_df['frm'] <= completion_frm)].reset_index(drop = True)

                frm_range = int(completion_frm-initiation_frm)
                if frm_range > 20 and initiation_frm != 0 and completion_frm != 0:

                    case_info = {
                    'scenario':scenario,
                    'id':id,
                    'frm_range':frm_range,
                    'yveh_id':yveh_id,
                    'lc_frm':lc_frm,
                    'initiation_frm':initiation_frm,
                    'completion_frm':completion_frm,
                    'episode_id':'l' + str(counter),

                    }

                    if all(mveh_df['frm'].diff().dropna() != 1):
                        raise ValueError("There are missing frames", case_info)

                    yveh_df = yveh_df[['scenario','frm','id','v_long','a_long']].rename(columns=yveh_names)

                    glob_pos = get_glob_df(case_info)
                    mveh_glob_pos = utils.get_vehglob_pos(glob_pos, id)
                    yveh_glob_pos = utils.get_vehglob_pos(glob_pos, yveh_id)

                    dx = utils.get_dx(mveh_glob_pos, yveh_glob_pos, case_info, lane_cor)
                    gap_size = utils.get_gap_size(mveh_df, case_info, glob_pos, lane_cor)

                    mveh_df, yveh_df = utils.get_veh_feats(mveh_df, yveh_df, gap_size, dx, case_info['episode_id'])

                    test_list.append([case_info, gap_size])
                    counter += 1
                    case_info['gap_size'] = gap_size
                    print(counter, ' ### lane change extracted ###')

                    # draw_traj(mveh_df, yveh_df, case_info)
                    utils.data_saver(mveh_df, yveh_df)

# %%
case_info
mveh_df.columns
feat_df = feature_set.loc[(feature_set['scenario'] == 'i101_1') &
                                    (feature_set['lane_id'] < 7)] # feat_set_scene

car_df = feat_df.loc[(feat_df['id'] == 695)].reset_index(drop = True)

plt.plot(mveh_df['v_lat'])
car_df
car_df
plt.plot(car_df['pc'])
plt.plot(veh_df['bb_id'].iloc[25:])

vehicle_df = veh_df

vehicle_df.loc[(vehicle_df['frm'] > lc_frm) &
                            # (vehicle_df['bb_id'] == 699) &
                            (vehicle_df['lane_id'] == lane_id) &
                            ((vehicle_df['pc'].abs() < 1) |
                            (vehicle_df['v_lat'].abs() < 0.1))]['frm']
plt.plot(veh_df['lane_id'].iloc[25:])

case_info

vehicle_df = veh_df

initiation_frms = vehicle_df.loc[(vehicle_df['frm'] < lc_frm) &
                            # (vehicle_df['bl_id'] == yveh_id) &
                            (vehicle_df['lane_id'] == 3) &
                            (vehicle_df['v_lat'].abs() < 0.1)]

completion_frm
vehicle_df.loc[(vehicle_df['frm'] > lc_frm) &
                            (vehicle_df['bb_id'] == yveh_id) &
                            (vehicle_df['lane_id'] == lane_id) &
                            ((vehicle_df['pc'].abs() < 1) |
                            (vehicle_df['v_lat'].abs() < 0.1))]['frm'].min()
mveh_df.columns




# %%
def draw_traj(mveh_df, yveh_df, case_info):
    # for some vis
    fig = plt.figure()
    item = 'pc'
    plt.plot(mveh_df[item])
    # plt.plot(yveh_df[item])

    # plt.plot(yveh_df[item])
    indx = mveh_df.loc[mveh_df['frm'] == case_info['lc_frm']].index[0]
    plt.scatter(indx, mveh_df[item].iloc[indx])
    plt.title([case_info['id'], case_info['lc_frm'],  case_info['gap_size']])
    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])


    fig = plt.figure()
    item = 'act_lat'
    plt.plot(mveh_df[item])

    # plt.plot(yveh_df[item])
    # plt.plot(yveh_df[item])
    indx = mveh_df.loc[mveh_df['frm'] == case_info['lc_frm']].index[0]
    plt.scatter(indx, mveh_df[item].iloc[indx])

    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])

# %%
