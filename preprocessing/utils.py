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

def lc_entrance(vehicle_df):
    """
    :return: lane change frames for a given car
    """
    lc_frms = {}
    lc_frms['left'] = vehicle_df[vehicle_df['lane_id'].diff() == -1]['frm'].values.tolist()
    lc_frms['right'] = vehicle_df[vehicle_df['lane_id'].diff() == 1]['frm'].values.tolist()

    return lc_frms

def get_yveh(vehicle_df, lc_frm):
    yveh_id = vehicle_df.loc[vehicle_df['frm'] == lc_frm
                                                ][['bb_id', 'e_class']]

    if yveh_id.empty:
        return 0

    return yveh_id.iloc[0].tolist()

def get_fveh(vehicle_df, lc_frm):
    fveh_id = vehicle_df.loc[vehicle_df['frm'] == lc_frm
                                                ]['ff_id']

    return fveh_id.iloc[0]

def lc_completion(vehicle_df, lc_frm, yveh_id):
    """
    :return: lane change completion frame
    """

    completion_frm = vehicle_df.loc[(vehicle_df['frm'] > lc_frm) &
                                (vehicle_df['bb_id'] == yveh_id) &
                                (vehicle_df['pc'].abs() < 1)]['frm']

    if not completion_frm.empty:
        return completion_frm.iloc[0], yveh_id
    else:
        return vehicle_df.iloc[-1]['frm'], yveh_id


def lc_initation(vehicle_df, lc_frm, yveh_id, lc_direction):
    if lc_direction == 'right':
        yveh_name = 'br_id'
    else:
        yveh_name = 'bl_id'

    initiation_frm = vehicle_df.loc[(vehicle_df['frm'] < lc_frm) &
                                (vehicle_df[yveh_name] == yveh_id) &
                                (vehicle_df['v_lat'].abs() < 0.1)]['frm']


    if not initiation_frm.empty:
        initiation_frm = initiation_frm.iloc[0]
        if not vehicle_df.loc[vehicle_df['frm'] == initiation_frm - 20].empty:
            initiation_frm -= 20
        return initiation_frm
    else:
        return vehicle_df.iloc[0]['frm']

def get_vehglob_pos(glob_pos, vehicle_id):
    return glob_pos.loc[glob_pos['id'] == vehicle_id].drop(['id','frm'],axis=1).values.tolist()

def get_dxpc(mveh_glob_pos, yveh_glob_pos, case_info, lane_cor):
    dx = []

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

def rolling_average(values, N):
    values_ = []

    for n in range(len(values)):
        if n == len(values)-N:
            c = values[n:]
        else:
            c = values[n:n+N]

        values_.append(np.mean(c))
    return values_

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

def get_var_deltarate(var):
    # rate of change of variable
    var_deltarate = []
    for n in range(len(var)):
        if n == len(var)-1:
            var_c = var[n-1]
            var_ff = var[n]
        else:
            var_c = var[n]
            var_ff = var[n+1]

        var_deltarate.append((var_ff - var_c)/0.1)
    return var_deltarate


def get_veh_feats(mveh_df, yveh_df, gap_size, dx, pc):
    mveh_df = mveh_df[['id', 'frm', 'scenario', 'v_long', 'a_long',
                                'v_lat','pc']]

    mveh_df['dx'] = pd.DataFrame(dx)

    mveh_df.insert(loc=6, column='gap_size', value=gap_size)
    veh_feats = pd.concat([mveh_df,yveh_df], axis=1)
    veh_feats = veh_feats.loc[veh_feats['dx']>=0].reset_index(drop = True)

    return veh_feats
