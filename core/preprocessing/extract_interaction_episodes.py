import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from models.core.preprocessing import utils
from math import hypot
import json
from importlib import reload

cwd = os.getcwd()

# %%
"""
m_df - merge_mveh_df
y_df - yield_mveh_df
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
feature_set['fr_long'].mean()
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
    plt.plot(m_df['frm'], m_df[item])
    # plt.plot(y_df[item])

    # plt.plot(y_df[item])
    indx = m_df.loc[m_df['frm'] == case_info['lc_frm']].index[0]

    plt.scatter(case_info['lc_frm'], m_df[item].iloc[indx])
    plt.title([case_info['id'], case_info['lc_frm'], case_info['scenario']])
    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])


    fig = plt.figure()
    item = 'act_lat'
    plt.plot(m_df['frm'], m_df[item])

    # plt.plot(y_df[item])
    # plt.plot(y_df[item])
    plt.scatter(case_info['lc_frm'], m_df[item].iloc[indx])

    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])

def get_glob_df(case_info):
    """
    :return: global pose of interacting cars
    Note: start_frm and end_frm are not included here. They are dropped later when
    calculating acceleations.
    """

    glob_pos = df_all.loc[(df_all['scenario'] == case_info['scenario']) &
                            (df_all['frm'] >= case_info['start_frm']) &
                            (df_all['frm'] <=  case_info['end_frm'])]

    return glob_pos[['id','frm','x_front','y_front', 'length']]

def get_lane_cor(scenario, lane_id):
    if scenario in ['i101_1', 'i101_2', 'i101_3']:
        xc = np.array(xc_101[int(lane_id-1)])
        yc = np.array(yc_101[int(lane_id-1)])
    else:
        xc = np.array(xc_80[int(lane_id-1)])
        yc = np.array(yc_80[int(lane_id-1)])

    return [xc, yc]

# %%
# def get_act_lat():
reload(utils)

counter = 0
episode_spec = {}

for scenario in datasets:
    feat_df = feature_set.loc[(feature_set['scenario'] == scenario) &
                                        (feature_set['lane_id'] < 7)] # feat_set_scene
    ids = feat_df['id'].unique().astype('int')

    for id in ids:
        mveh_df = feat_df.loc[(feat_df['id'] == id)].reset_index(drop = True)
        lc_frms = utils.lc_entrance(mveh_df)

        for lane_change in lc_frms:
            if lane_change == 'right':
                lc_type = -1
            else:
                lc_type = 1

            if lc_frms[lane_change]:
                for lc_frm, lane_id in lc_frms[lane_change]:
                    m_class = mveh_df['e_class'].iloc[0]
                    end_frm = utils.lc_completion(mveh_df, lc_frm, lane_change, lane_id)
                    start_frm = utils.lc_initation(mveh_df, lc_frm-1, lane_change, lane_id)
                    frms_n = int(end_frm-start_frm)
                    m_df = utils.frmTrim(mveh_df, end_frm, start_frm)
                    v_min = m_df['vel'].min()

                    if m_class == 2 and frms_n > 20 and \
                                        start_frm != 0 and end_frm != 0 and v_min > 0:

                        try:
                            y_id, _ = utils.get_vehInfo(mveh_df, lc_frm, 'bb_id')
                            fadj_id, _ = utils.get_vehInfo(mveh_df, lc_frm, 'ff_id')
                            f_id, _ = utils.get_vehInfo(mveh_df, lc_frm-1, 'ff_id')

                            case_info = {
                                        'scenario':scenario,
                                        'id':id,
                                        'frms_n':frms_n,
                                        'y_id':y_id,
                                        'lc_frm':lc_frm,
                                        'start_frm':start_frm,
                                        'end_frm':end_frm,
                                        'episode_id': counter,
                                        'lc_type': lc_type
                                        }
                            lane_cor = get_lane_cor(scenario, lane_id)
                            m_df = utils.get_m_features(m_df, case_info)
                            o_df = utils.frmTrim(feat_df, end_frm, start_frm) # other vehicles' df
                            glob_pos = get_glob_df(case_info)

                            if y_id:
                                y_df = utils.get_o_df(o_df, y_id, case_info['episode_id'])
                                m_df, y_df = utils.get_dxdv(glob_pos, m_df, y_df, lane_cor, 'front')
                                y_df['exists'] = 1

                            if fadj_id:
                                fadj_df = utils.get_o_df(o_df, fadj_id, case_info['episode_id'])
                                m_df, fadj_df = utils.get_dxdv(glob_pos, m_df, fadj_df, lane_cor, 'behind')
                                fadj_df['exists'] = 1

                            if f_id:
                                f_df = utils.get_o_df(o_df, f_id, case_info['episode_id'])
                                m_df, f_df = utils.get_dxdv(glob_pos, m_df, f_df, lane_cor, 'behind')
                                f_df['exists'] = 1

                            mveh_size = len(m_df)

                            if not y_id:
                                y_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                            else:
                                y_df = utils.applyCorrections(m_df, y_df, mveh_size)

                            if not fadj_id:
                                fadj_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                            else:
                                fadj_df = utils.applyCorrections(m_df, fadj_df, mveh_size)

                            if not f_id:
                                f_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                            else:
                                f_df = utils.applyCorrections(m_df, f_df, mveh_size)

                            utils.data_saver(m_df, 'm_df')
                            utils.data_saver(y_df, 'y_df')
                            utils.data_saver(f_df, 'f_df')
                            utils.data_saver(fadj_df, 'fadj_df')

                            episode_spec[counter] = {'episode_id':counter,'scenario':scenario ,
                                        'lc_frm':lc_frm, 'm_id':int(id), 'y_id':y_id,'fadj_id':fadj_id,
                                        'f_id':f_id, 'frm_n':mveh_size}

                            counter += 1
                            print(counter, ' ### lane change extracted ###')
                            # draw_traj(m_df, y_df, case_info)
                        except:
                            print('This episode has a vehicle with missing frame - episode is ignored')
                            continue

pd.DataFrame.from_dict(episode_spec, orient='index').to_csv('./datasets/episode_spec.txt',
                                header=None, index=None, sep=' ', mode='a')

# %%
counter = 0
reload(utils)
episode_spec = {}

scenario = 'i80_3'
id = 1741
feat_df = feature_set.loc[(feature_set['scenario'] == scenario) &
                                        (feature_set['lane_id'] < 7)] # feat_set_scene
mveh_df = feat_df.loc[(feat_df['id'] == id)].reset_index(drop = True)
lc_frms = utils.lc_entrance(mveh_df)

for lane_change in lc_frms:
    if lane_change == 'right':
        lc_type = -1
    else:
        lc_type = 1

    if lc_frms[lane_change]:
        for lc_frm, lane_id in lc_frms[lane_change]:
            m_class = mveh_df['e_class'].iloc[0]
            end_frm = utils.lc_completion(mveh_df, lc_frm, lane_change, lane_id)
            start_frm = utils.lc_initation(mveh_df, lc_frm-1, lane_change, lane_id)
            frms_n = int(end_frm-start_frm)
            m_df = utils.frmTrim(mveh_df, end_frm, start_frm)
            v_min = m_df['vel'].min()
            if m_class == 2 and frms_n > 20 and \
                                start_frm != 0 and end_frm != 0 and v_min > 0:

                y_id, _ = utils.get_vehInfo(mveh_df, lc_frm, 'bb_id')
                fadj_id, _ = utils.get_vehInfo(mveh_df, lc_frm, 'ff_id')
                f_id, _ = utils.get_vehInfo(mveh_df, lc_frm-1, 'ff_id')

                case_info = {
                            'scenario':scenario,
                            'id':id,
                            'frms_n':frms_n,
                            'y_id':y_id,
                            'lc_frm':lc_frm,
                            'start_frm':start_frm,
                            'end_frm':end_frm,
                            'episode_id': counter,
                            'lc_type': lc_type
                            }

                lane_cor = get_lane_cor(scenario, lane_id)

                m_df = utils.get_m_features(m_df, case_info)
                o_df = utils.frmTrim(feat_df, end_frm, start_frm) # other vehicles' df
                glob_pos = get_glob_df(case_info)

                if y_id:
                    y_df = utils.get_o_df(o_df, y_id, case_info['episode_id'])
                    m_df, y_df = utils.get_dxdv(glob_pos, m_df, y_df, lane_cor, 'front')
                    y_df['exists'] = 1

                if fadj_id:
                    fadj_df = utils.get_o_df(o_df, fadj_id, case_info['episode_id'])
                    m_df, fadj_df = utils.get_dxdv(glob_pos, m_df, fadj_df, lane_cor, 'behind')
                    fadj_df['exists'] = 1

                if f_id:
                    f_df = utils.get_o_df(o_df, f_id, case_info['episode_id'])
                    m_df, f_df = utils.get_dxdv(glob_pos, m_df, f_df, lane_cor, 'behind')
                    f_df['exists'] = 1

                mveh_size = len(m_df)

                if not y_id:
                    y_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                else:
                    y_df = utils.applyCorrections(m_df, y_df, mveh_size)

                if not fadj_id:
                    fadj_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                else:
                    fadj_df = utils.applyCorrections(m_df, fadj_df, mveh_size)

                if not f_id:
                    f_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                else:
                    f_df = utils.applyCorrections(m_df, f_df, mveh_size)

                # utils.data_saver(m_df, 'm_df')
                # utils.data_saver(y_df, 'y_df')
                # utils.data_saver(f_df, 'f_df')
                # utils.data_saver(fadj_df, 'fadj_df')
                episode_spec[counter] = {'episode_id':counter,'scenario':scenario ,
                            'lc_frm':lc_frm, 'm_id':int(id), 'y_id':y_id,'fadj_id':fadj_id,
                            'f_id':f_id, 'frm_n':mveh_size}

                counter += 1
                print(counter, ' ### lane change extracted ###')
                        # draw_traj(m_df, y_df, case_info)
                draw_traj(m_df, y_df, case_info)


# %%
episode_spec[2]
case_info
test = feature_set.loc[(feature_set['scenario'] == case_info['scenario']) &
                        (feature_set['frm'] >= case_info['start_frm']) &
                        (feature_set['frm'] <= case_info['end_frm']) &
                        (feature_set['lane_id'] < 7) & # feat_set_scene
                                    (feature_set['id'] == 2174)]
plt.plot(test['br_id'])

# %%
plt.plot(frms)


plt.plot(m_df['pc'])
plt.plot(m_df['bl_id'])
plt.plot(y_df['dx'])

# %%

f_df

case_info
y_df.columns

f_df.columns
