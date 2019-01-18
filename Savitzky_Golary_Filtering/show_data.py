import pandas
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

pd = pandas.read_csv('3d.gt', sep=' ', names=['order','x','y','z','p','s'])

print(type(pd['x'].values))
print(pd['x'].values)

filter_x = savgol_filter(pd['x'].values, window_length=5, polyorder=1, mode='interp')
filter_y = savgol_filter(pd['y'].values, window_length=5, polyorder=1, mode='interp')
filter_z = savgol_filter(pd['z'].values, window_length=5, polyorder=1, mode='interp')

diff_x = pandas.DataFrame(pd['x'] - filter_x)
diff_y = pandas.DataFrame(pd['y'] - filter_y)
diff_z = pandas.DataFrame(pd['z'] - filter_z)

plt.figure()
pd['x'].plot()
pd['y'].plot()
pd['z'].plot()

diff_x.plot()
diff_y.plot()
diff_z.plot()

plt.show()

#
# with open('3d.gt', 'r') as f:
#     contents = f.readlines()
#
#
#
# for i, content in enumerate(contents):
#     x = content.split(' ')[1]