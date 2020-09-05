import pandas as pd
import numpy as np
def get_pc(p, Ex, Ey):
    """ a car's distance from the shadow point of target lane (lane the merging car
    merges into)
    """
    delta_x = Ex-p[0]
    delta_y = Ey-p[1]
    pc = np.hypot(delta_x,delta_y) #|EP|, distance of car to ego's centerline
    return pc

def get_p(car_glob_pos, xc, yc):
    Ex_o = car_glob_pos[0]
    Ey_o = car_glob_pos[1]

    # to find closest datums to car's centre
    delta_x = xc - Ex_o
    delta_y = yc - Ey_o
    idx = np.argsort(np.hypot(delta_x,delta_y))
    ###
    n1 = idx[0]
    n2 = idx[1]
    Ax, Bx = xc[n1], xc[n2]
    Ay, By = yc[n1], yc[n2]
    A = np.array([Ax, Ay])
    B = np.array([Bx, By])
    mag = np.hypot(Ax-Bx,Ay-By)
    U = (B - A)/mag
    Ux = U[0]
    Uy = U[1]
    g = (Ux*Ax-Ex_o*Ux+Uy*Ay-Uy*Ey_o)/-(Uy**2+Ux**2) # solving for g (equation by from solving dot product)
    p = A + g*U # shaddow point on centreline/current
    return p

def correct_glitch(vehicle_df, indexes):
    if len(indexes) > 1:
        for i in range(len(indexes)-1):
            lc_i = indexes[i]
            lc_ii = indexes[i+1]
            min_pc = vehicle_df['pc'].iloc[lc_i:lc_ii].abs().min()
            if min_pc > 1.25 or lc_ii-lc_i < 20:
                # not really a lane change!
                keep_fixed = ['ff_id', 'bb_id', 'bl_id', 'br_id', 'lane_id']

                vehicle_df.loc[lc_i:lc_ii, keep_fixed] = \
                                    vehicle_df.iloc[lc_i - 1][keep_fixed].values

                for indx in range(lc_i,lc_ii):
                    vehicle_df.at[indx, 'pc'] = vehicle_df['pc'].iloc[indx-1] + \
                                                            0.1*vehicle_df['v_lat'].iloc[indx-1]

def detect_glitch(vehicle_df):
    """
    Detects some glitch in lane_id and pc values. Glitch cause not investigated.
    """
    indexes = vehicle_df[vehicle_df['lane_id'].diff().abs() == 1].index
    correct_glitch(vehicle_df, indexes)

def lc_entrance(vehicle_df):
    """
    :return: lane change frames for a given car
    """
    detect_glitch(vehicle_df)

    lc_frms = {}
    lc_frms['left'] = vehicle_df[vehicle_df['lane_id'].diff() == -1
                                            ][['frm', 'lane_id']].values.tolist()
    lc_frms['right'] = vehicle_df[vehicle_df['lane_id'].diff() == 1
                                            ][['frm', 'lane_id']].values.tolist()

    return lc_frms

def get_yveh(vehicle_df, lc_frm):
    yveh_id = vehicle_df.loc[vehicle_df['frm'] == lc_frm
                                                ][['bb_id', 'e_class']]

    return yveh_id.iloc[0].tolist()

def get_fveh(vehicle_df, lc_frm):
    fveh_id = vehicle_df.loc[vehicle_df['frm'] == lc_frm
                                                ]['ff_id']

    return fveh_id.iloc[0]

def lc_completion(vehicle_df, lc_frm, yveh_id, lane_id):
    """
    :return: lane change completion frame
    """

    completion_frm = vehicle_df.loc[(vehicle_df['frm'] > lc_frm) &
                                (vehicle_df['bb_id'] == yveh_id) &
                                (vehicle_df['lane_id'] == lane_id) &
                                (vehicle_df['v_lat'].abs() < 0.1)]['frm']

    if not completion_frm.empty:
        return completion_frm.iloc[0]
    else:
        return 0


def lc_initation(vehicle_df, lc_frm, yveh_id, lc_direction, lane_id):
    if lc_direction == 'right':
        yveh_name = 'br_id'
        lane_id -= 1

    else:
        yveh_name = 'bl_id'
        lane_id += 1

    initiation_frms = vehicle_df.loc[(vehicle_df['frm'] < lc_frm) &
                                (vehicle_df[yveh_name] == yveh_id) &
                                (vehicle_df['lane_id'] == lane_id) &
                                (vehicle_df['v_lat'].abs() < 0.1)]

    if not initiation_frms.empty:
        initiation_frm = initiation_frms['frm'].iloc[-1]
        if not initiation_frms.loc[initiation_frms['frm'] < initiation_frm - 20].empty:
            initiation_frm -= 20
        return initiation_frm
    else:
        return 0

def get_vehglob_pos(glob_pos, vehicle_id):
    return glob_pos.loc[glob_pos['id'] == vehicle_id].drop(['id','frm'],axis=1).values.tolist()

def get_dx(mveh_glob_pos, yveh_glob_pos, case_info, lane_cor):
    dx = []
    mveh_size = len(mveh_glob_pos)
    yveh_size = len(yveh_glob_pos)

    if yveh_size != mveh_size:
        raise Exception("mveh and yveh have different lengths: {} vs {}".format(
                                                        mveh_size, yveh_size))

    for i in range(case_info['frm_range']+1):
        mveh_c_x = mveh_glob_pos[i][0]
        mveh_c_y = mveh_glob_pos[i][1]
        yveh_c_x = yveh_glob_pos[i][0]
        yveh_c_y = yveh_glob_pos[i][1]
        mveh_length = mveh_glob_pos[i][2]

        mveh_p = get_p([mveh_c_x,mveh_c_y], lane_cor[0], lane_cor[1] )
        yveh_p = get_p([yveh_c_x,yveh_c_y], lane_cor[0], lane_cor[1] )

        yveh_long = np.hypot(mveh_p[0]-yveh_p[0],mveh_p[1]-yveh_p[1])-mveh_length
        dx.append(yveh_long)

    return dx

def get_gap_size(vehicle_df, case_info, glob_pos, lane_cor):
    fveh_id = get_fveh(vehicle_df, case_info['lc_frm'])

    if fveh_id != 0:
        glob_pos = glob_pos.loc[(glob_pos['frm'] == case_info['lc_frm'])]
        fveh_glob_pos = get_vehglob_pos(glob_pos, fveh_id)
        yveh_glob_pos = get_vehglob_pos(glob_pos, case_info['yveh_id'])
        yveh_c_x = yveh_glob_pos[0][0]
        yveh_c_y = yveh_glob_pos[0][1]
        fveh_c_x = fveh_glob_pos[0][0]
        fveh_c_y = fveh_glob_pos[0][1]
        fveh_length = fveh_glob_pos[0][2]

        fveh_p = get_p([fveh_c_x,fveh_c_y], lane_cor[0], lane_cor[1] )
        yveh_p = get_p([yveh_c_x,yveh_c_y], lane_cor[0], lane_cor[1] )

        return np.hypot(fveh_p[0]-yveh_p[0],fveh_p[1]-yveh_p[1])-fveh_length

    else:
        return 70

def get_veh_feats(mveh_df, yveh_df, gap_size, dx, case_info):
    mveh_df = mveh_df[['id', 'frm', 'scenario', 'v_long',
                                'v_lat','pc']]

    mveh_df = mveh_df.rename(columns={'v_lat':'act_lat', 'v_long':'vel'})
    mveh_df.insert(loc=6, column='gap_size', value=gap_size)
    mveh_df.insert(loc=1, column='episode_id', value=case_info['episode_id'])
    mveh_df.insert(loc=2, column='name', value='mveh')
    mveh_df.insert(loc=3, column='lc_type', value=case_info['lc_type'])

    mveh_df.loc[:,'dx'] = pd.Series(dx)
    mveh_df.loc[:,'act_long'] = pd.Series(dx)
    get_act_long(mveh_df)
    get_past_action(mveh_df, 'mveh')
    get_act_long(yveh_df)
    get_past_action(yveh_df, 'yveh')
    yveh_df.insert(loc=1, column='episode_id', value=case_info['episode_id'])
    yveh_df.insert(loc=2, column='name', value='yveh')
    yveh_df.insert(loc=3, column='lc_type', value=case_info['lc_type'])

    yveh_df = yveh_df[['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'act_long_p', 'act_long']]
    mveh_df = mveh_df[['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'pc',
           'gap_size', 'dx', 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']]

    return mveh_df, yveh_df

def data_saver(mveh_df, yveh_df):
    check = episode_checker(mveh_df, yveh_df)

    if check == 1:
        mveh_df.to_csv('/datasets/mveh_df.txt',
                                        header=None, index=None, sep=' ', mode='a')
        yveh_df.to_csv('/datasets/yveh_df.txt',
                                        header=None, index=None, sep=' ', mode='a')

def get_act_long(vehicle_df):
    acc = (vehicle_df['vel'].iloc[1:].values - vehicle_df['vel'].iloc[:-1].values)/0.1
    vehicle_df.drop(vehicle_df.index[-1],  inplace=True)
    vehicle_df.reset_index(drop=True,  inplace=True)

    vehicle_df.loc[:,'act_long'] = acc

def get_past_action(vehicle_df, name):
    if name ==  'mveh':
        action_names = ['act_long', 'act_lat']
    else:
        action_names = ['act_long']

    action_p = vehicle_df[action_names].iloc[:-1].values
    action_names_p = [name +'_p' for name in action_names]
    vehicle_df.drop(vehicle_df.index[0],  inplace=True)
    vehicle_df.reset_index(drop=True,  inplace=True)
    vehicle_df[action_names_p] = pd.DataFrame(action_p)

def episode_checker(mveh_df, yveh_df):
    """
    Exclusion of some cars from training.
    If return 1, accept the car
    """
    mveh_size = len(mveh_df)
    yveh_size = len(yveh_df)
    vel_min = mveh_df['vel'].min()
    gap_size = mveh_df['gap_size'].iloc[0]

    if yveh_size != mveh_size or vel_min < 0 or gap_size < 5:
        check = 0
    else:
        check = 1

    return check
