import pandas
import matplotlib.pyplot as plt

pd = pandas.read_csv('3d.gt', sep=' ', names=['order','x','y','z','p','s'])

plt.figure()
pd['x'].plot()
pd['y'].plot()
pd['z'].plot()
plt.show()

#
# with open('3d.gt', 'r') as f:
#     contents = f.readlines()
#
#
#
# for i, content in enumerate(contents):
#     x = content.split(' ')[1]