from Reconstruct import Camera
from utils.path_parser import get_camera_info_dir
import json
from Reconstruct.reconstruct import transto3d, find_fundamental_matrix
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


cam0 = Camera(0, 'cam0', None, None, None)
cam0.pack_json_camera()

cam1 = Camera(1, 'cam1', None, None, None)
cam1.pack_json_camera()

F = find_fundamental_matrix(cam0, cam1)


with open('final_loc.json', 'r') as f:
    final_locs = json.load(f)

res1 = final_locs['res_1_s']
res2 = final_locs['res_2_s']

X, Y, Z = [], [], []
for r_1, r_2 in zip(res1, res2):
    res_3d = transto3d(r_1, r_2, cam0, cam1, F)
    X.append(res_3d[0][0])
    Y.append(res_3d[0][1])
    Z.append(res_3d[0][2])


pd_x = pandas.DataFrame(X)
pd_y = pandas.DataFrame(Y)
pd_z = pandas.DataFrame(Z)

plt.figure()
pd_x.plot(title='x')
pd_y.plot(title='y')
pd_z.plot(title='z')


filter_x = savgol_filter(np.array(X), window_length=61, polyorder=1, mode='interp')
filter_y = savgol_filter(np.array(Y), window_length=61, polyorder=1, mode='interp')
filter_z = savgol_filter(np.array(Z), window_length=61, polyorder=2, mode='interp')


pandas.DataFrame(filter_x).plot(title='filter_x')
pandas.DataFrame(filter_y).plot(title='filter_y')
pandas.DataFrame(filter_z).plot(title='filter_z')

diff_x = pandas.DataFrame(pd_x - pandas.DataFrame(filter_x))
diff_y = pandas.DataFrame(pd_y - pandas.DataFrame(filter_y))
diff_z = pandas.DataFrame(pd_z - pandas.DataFrame(filter_z))

diff_x.plot(title='dif_x')
diff_y.plot(title='dif_y')
diff_z.plot(title='dif_z')

plt.show()
