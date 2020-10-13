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

def correct_glitch(veh_df, indexes):
    if len(indexes) > 1:
        for i in range(len(indexes)-1):
            lc_i = indexes[i]
            lc_ii = indexes[i+1]
            min_pc = veh_df['pc'].iloc[lc_i:lc_ii].abs().min()
            if min_pc > 1.25 or lc_ii-lc_i < 20:
                # not really a lane change!
                keep_fixed = ['ff_id', 'bb_id', 'bl_id', 'br_id', 'lane_id']

                veh_df.loc[lc_i:lc_ii, keep_fixed] = \
                                    veh_df.iloc[lc_i - 1][keep_fixed].values

                for indx in range(lc_i,lc_ii):
                    veh_df.at[indx, 'pc'] = veh_df['pc'].iloc[indx-1] + \
                                                            0.1*veh_df['act_lat'].iloc[indx-1]

def detect_glitch(veh_df):
    """
    Detects some glitch in lane_id and pc values. Glitch cause not investigated.
    """
    indexes = veh_df[veh_df['lane_id'].diff().abs() == 1].index
    correct_glitch(veh_df, indexes)

def lc_entrance(veh_df):
    """
    :return: lane change frames for a given car
    """
    detect_glitch(veh_df)
    lc_frms = {}
    lc_frms['left'] = veh_df[veh_df['lane_id'].diff() == -1
                                            ][['frm', 'lane_id']].values.tolist()
    lc_frms['right'] = veh_df[veh_df['lane_id'].diff() == 1
                                            ][['frm', 'lane_id']].values.tolist()

    return lc_frms

def get_vehInfo(veh_df, lc_frm, veh_type):
    veh_id = veh_df.loc[veh_df['frm'] == lc_frm][[veh_type, 'e_class']]
    return veh_id.iloc[0].tolist()

def lc_completion(veh_df, lc_frm, lc_direction, lane_id):
    """
    :return: lane change completion frame
    """
    if lc_direction == 'right':
        end_frm = veh_df.loc[(veh_df['frm'] > lc_frm) &
                                    (veh_df['lane_id'] == lane_id) &
                                    ((veh_df['act_lat'].abs() < 0.1) |
                                    (veh_df['pc'] < 0))]['frm']

        if not end_frm.empty:
            return end_frm.iloc[0]
        else:
            return 0

    else:
        end_frm = veh_df.loc[(veh_df['frm'] > lc_frm) &
                                    (veh_df['lane_id'] == lane_id) &
                                    ((veh_df['act_lat'].abs() < 0.1) |
                                    (veh_df['pc'] > 0))]['frm']

        if not end_frm.empty:
            return end_frm.iloc[0]
        else:
            return 0

def lc_initation(veh_df, lc_frm, lc_direction, lane_id):
    if lc_direction == 'right':
        lane_id -= 1
    else:
        lane_id += 1

    if lc_direction == 'right':
        start_frms = veh_df.loc[(veh_df['frm'] < lc_frm) &
                                    (veh_df['lane_id'] == lane_id) &
                                    ((veh_df['act_lat'].abs() < 0.1) |
                                    ((veh_df['act_lat'].abs() > 0.1) &
                                    (veh_df['pc'] > 0)))]


        if not start_frms.empty:
            start_frm = start_frms['frm'].iloc[-1]
            if not start_frms.loc[start_frms['frm'] < start_frm - 20].empty:
                start_frm -= 20
            return start_frm
        else:
            return 0
    else:
        start_frms = veh_df.loc[(veh_df['frm'] < lc_frm) &
                                    (veh_df['lane_id'] == lane_id) &
                                    ((veh_df['act_lat'].abs() < 0.1) |
                                    ((veh_df['act_lat'].abs() > 0.1) &
                                    (veh_df['pc'] < 0)))]


        if not start_frms.empty:
            start_frm = start_frms['frm'].iloc[-1]
            if not start_frms.loc[start_frms['frm'] < start_frm - 20].empty:
                start_frm -= 20
            return start_frm
        else:
            return 0

def get_globPos(glob_pos, vehicle_id):
    return glob_pos.loc[glob_pos['id'] == vehicle_id].drop(['id'],axis=1).values

def get_m_features(m_df, case_info):
    get_act_long(m_df)
    get_past_action(m_df, 'mveh')
    m_df.loc[:, ['episode_id', 'lc_type']] = [case_info['episode_id'], case_info['lc_type']]
    col = ['episode_id', 'id', 'frm', 'vel', 'pc', 'lc_type',
                                        'act_long_p', 'act_lat_p', 'act_long', 'act_lat']
    return m_df[col]

def applyCorrections(m_df, veh_df, mveh_size):
    """All dfs must be the same size.
    """
    if len(veh_df) != mveh_size:
        frm_max =  m_df['frm'].iloc[-1]
        frm_min =  m_df['frm'].iloc[0]
        veh_df = frmTrim(veh_df, frm_max, frm_min)
    return remove_redundants(veh_df)

def remove_redundants(veh_df):
    return veh_df[['episode_id','frm', 'exists', 'vel', 'dx', 'act_long_p', 'act_long']]

def get_o_df(o_df, veh_id, episode_id):
    veh_df = o_df.loc[o_df['id'] == veh_id].reset_index(drop = True)
    get_act_long(veh_df)
    get_past_action(veh_df, 'o')
    veh_df['episode_id'] = episode_id
    return veh_df

def get_dummyVals(episode_id, df_size):
    """
    Note: dummy values are dataset averages. This is done avoid distorting normalization
    """
    dummy_df = pd.DataFrame(np.repeat([[episode_id, 0, 0, 7.5, 18, 0, 0]], df_size, axis=0),
            columns=['episode_id', 'frm', 'exists', 'vel', 'dx', 'act_long_p', 'act_long'])

    return dummy_df

def get_dxdv(glob_pos, m_df, veh_df, lane_cor, mveh_orientation):
    """
    Finds longitudinal distance between two vehicles.
    """
    dx = []
    m_glob_pos = get_globPos(glob_pos, m_df['id'].iloc[0])
    o_glob_pos = get_globPos(glob_pos, veh_df['id'].iloc[0])

    if not (len(m_glob_pos) == len(o_glob_pos) == len(m_df) == len(veh_df)):

        mins = [o_glob_pos[0,0], m_glob_pos[0,0], m_df['frm'].iloc[0],
                                                            veh_df['frm'].iloc[0]]

        maxes = [o_glob_pos[-1,0], m_glob_pos[-1,0], m_df['frm'].iloc[-1],
                                                            veh_df['frm'].iloc[-1]]

        frm_max = min(maxes)
        frm_min = max(mins)
        o_glob_pos = o_glob_pos[o_glob_pos[:,0] >= frm_min]
        o_glob_pos = o_glob_pos[o_glob_pos[:,0] <= frm_max]
        m_glob_pos = m_glob_pos[m_glob_pos[:,0] >= frm_min]
        m_glob_pos = m_glob_pos[m_glob_pos[:,0] <= frm_max]

        m_df = frmTrim(m_df, frm_max, frm_min)
        veh_df = frmTrim(veh_df, frm_max, frm_min)

    vehm_size = len(m_glob_pos)
    for i in range(vehm_size):
        vehm_c_x = m_glob_pos[i,1]
        vehm_c_y = m_glob_pos[i,2]
        veho_c_x = o_glob_pos[i,1]
        veho_c_y = o_glob_pos[i,2]

        if mveh_orientation == 'front':
            veh_length = m_glob_pos[i,3]
        else:
            veh_length = o_glob_pos[i,3]

        vehm_p = get_p([vehm_c_x,vehm_c_y], lane_cor[0], lane_cor[1])
        veho_p = get_p([veho_c_x,veho_c_y], lane_cor[0], lane_cor[1])

        dx_i = np.hypot(vehm_p[0]-veho_p[0],vehm_p[1]-veho_p[1])-veh_length
        dx.append(dx_i)

    veh_df.loc[:, 'dx'] = dx
    veh_df['episode_id'] = m_df['episode_id'].values
    return m_df, veh_df

def frmTrim(veh_df, frm_max, frm_min):
    veh_df = veh_df.loc[(veh_df['frm'] >= frm_min) &
                        (veh_df['frm'] <= frm_max)]
    return veh_df.reset_index(drop = True)

def get_veh_df(veh_df, veh_id, episode_id):
    veh_df = veh_df.loc[veh_df['id'] == veh_id].reset_index(drop = True)
    get_act_long(veh_df)
    get_past_action(veh_df, 'o')
    veh_df['episode_id'] = episode_id
    return veh_df

def data_saver(veh_df, o_name):
    file_name = './datasets/' + o_name + '.txt'
    veh_df.to_csv(file_name, header=None, index=None, sep=' ', mode='a')

def get_act_long(veh_df):
    acc = (veh_df['vel'].iloc[1:].values - veh_df['vel'].iloc[:-1].values)/0.1
    veh_df.drop(veh_df.index[-1],  inplace=True)
    veh_df.loc[:,'act_long'] = acc

def get_past_action(veh_df, name):
    if name ==  'mveh':
        action_names = ['act_long', 'act_lat']
    else:
        action_names = ['act_long']

    action_p = veh_df[action_names].iloc[:-1].values
    action_names_p = [name +'_p' for name in action_names]
    veh_df.drop(veh_df.index[0],  inplace=True)
    veh_df.reset_index(drop=True,  inplace=True)
    veh_df[action_names_p] = pd.DataFrame(action_p)
