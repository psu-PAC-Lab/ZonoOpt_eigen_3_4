import zonoopt as zono
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# make a 2D conzono from vertices
V = np.array([[5.0, 15.6],
               [10.0, 15.6 + 5.0*0.33],
               [4.0, 10.6],
               [15.0, 12.1],
               [13.0, 6.6],
               [7.5, 8.1],
               [7.2, 16.8]])
Z = zono.vrep_2_conzono(V)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
zono.plot(Z, ax=ax, color='b', alpha=0.5)
ax.axis('equal')
plt.show()

# make a 3D zonotope
Z = zono.Zono(sp.eye(3), np.zeros(3))

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zono.plot(Z, ax=ax, color='b', alpha=0.5, edgecolor=None)
ax.axis('equal')
plt.show()