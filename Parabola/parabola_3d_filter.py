import numpy as np
import math


def loc_dis(loc1, loc2):
    x1, y1, z1 = loc1
    x2, y2, z2 = loc2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def ground_check(traj):
    for loc in traj:
        x, y, z = loc
        if z > 120:
            return False
    return True


"""Predict new ball loc.

Assume ball trajectory, then estimation parameter form previous ball 3d locations,
then estimate new ball 3d location.
Note that # of previous loc should at least be 3.

Args:
    locs: An list of loc.loc is [x,y,z], both float.
    
Returns:
    a new loc , same format as the loc in locs, [x,y,z]
"""


def predict_new_pose(locs):
    try:
        if len(locs) < 3:
            raise NameError('history loc info is too little')
        x = [loc[0] for loc in locs]
        y = [loc[1] for loc in locs]
        z = [loc[1] for loc in locs]

        time_indx = np.array(list(range(len(locs))))

        z1 = np.polyfit(time_indx, x, 1)
        new_x = np.poly1d(z1)(len(locs))
        z2 = np.polyfit(time_indx, y, 1)
        new_y = np.poly1d(z2)(len(locs))
        z3 = np.polyfit(time_indx, z, 2)
        new_z = np.poly1d(z3)(len(locs))
        return new_x, new_y, new_z
    except NameError as e:
        print(e)



def parabola_3d_locs(init_3d_locs):
    res_traj = {}
    i = 0
    while i < len(init_3d_locs)-2:
        init_3d_traj = init_3d_locs[i:i+3]
        for k,loc in enumerate(init_3d_traj):
            res_traj[i+k] = loc
        if ground_check(init_3d_traj):
            z = [loc[2] for loc in init_3d_traj]
            g, _, _ = np.polyfit([0, 1, 2], z, 2)
            traj = init_3d_traj
            if abs(g) > 0.5:
                for k, loc in enumerate(init_3d_traj):
                    res_traj.pop(i+k)
                continue
            else:
                for j in range(i+3,len(init_3d_locs)):
                    pred_new_loc = predict_new_pose(traj)
                    if loc_dis(pred_new_loc, init_3d_locs[j]) < 100 and loc_dis(init_3d_locs[j-1], init_3d_locs[j]) < 200:
                        traj.append(init_3d_locs[j])
                        res_traj[j] = init_3d_locs[j]
                    else:
                        i = i + len(traj) + 1
        else:
            z = [loc[2] for loc in init_3d_traj]
            g, _, _ = np.polyfit([0, 1, 2], z, 2)
            traj = init_3d_traj
            if abs(g - 4.905) > 0.2:
                for k, loc in enumerate(init_3d_traj):
                    res_traj.pop(i+k)
                continue
            else:
                for j in range(i+3,len(init_3d_locs)):
                    pred_new_loc = predict_new_pose(traj)
                    if loc_dis(pred_new_loc, init_3d_locs[j]) < 100 and loc_dis(init_3d_locs[j-1], init_3d_locs[j]) < 600:
                        traj.append(init_3d_locs[j])
                        res_traj[j] = init_3d_locs[j]
                    else:
                        i = i + len(traj) + 1
    return res_traj