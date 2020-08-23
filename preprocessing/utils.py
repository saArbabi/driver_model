import pandas as pd

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
    mag = hypot(Ax-Bx,Ay-By)
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

def lc_completion(vehicle_df, lc_frm, yveh_name):
    """
    :return: lane change completion frame
    """
    yveh_id, lane_id = vehicle_df.loc[vehicle_df['frm'] == lc_frm][[yveh_name, lane_id]]

    completion_frm = vehicle_df.loc[(vehicle_df['frm'] > lc_frm) &
                                (vehicle_df[yveh_name] == yveh_id) &
                                (vehicle_df['lane_id'] == lane_id) &
                                (vehicle_df['v_lat'].abs() < 0.1)].iloc[0]['frm']


    return completion_frm, yveh_id
    

def lc_initation(vehicle_df, lc_frm, yveh_name):
    yveh_id, lane_id = vehicle_df.loc[vehicle_df['frm'] == lc_frm][[yveh_name, lane_id]]

    initiation_frm = vehicle_df.loc[(vehicle_df['frm'] < lc_frm) &
                                (vehicle_df[yveh_name] == yveh_id) &
                                (vehicle_df['lane_id'] == lane_id) &
                                (vehicle_df['v_lat'].abs() < 0.1)].iloc[0]['frm']

    gap_size = []

    if not vehicle_df.loc[vehicle_df['frm'] == initiation_frm - 20].empty:
        initiation_frm -= 20

    return initiation_frm
