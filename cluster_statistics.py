import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

def plot_spheres(f, subPos, x, y, z, r, color, L, xb, yb, zb, rb, colorb):
    # Get the current axes, creating one if necessary
    
    ax = f.add_subplot(subPos, projection='3d')
    
    # Make sure the axes are held so we append spheres
    Npts = len(x)
    assert Npts == len(y) == len(z), "Dimensions of x, y, and z must match"
    
    # Preallocate sphere plotting
    for i in range(Npts):
        # Create a unit sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        X = r * np.cos(u) * np.sin(v) + x[i]
        Y = r * np.sin(u) * np.sin(v) + y[i]
        Z = r * np.cos(v) + z[i]
        
        # Plot the sphere
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, linewidth=0, shade=True)

    Nptsb = len(xb)
    assert Nptsb == len(yb) == len(zb), "Dimensions of x, y, and z must match"
    
    # Preallocate sphere plotting
    for i in range(Nptsb):
        # Create a unit sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        X = rb * np.cos(u) * np.sin(v) + xb[i]
        Y = rb * np.sin(u) * np.sin(v) + yb[i]
        Z = rb * np.cos(v) + zb[i]
        
        # Plot the sphere
        ax.plot_surface(X, Y, Z, color=colorb, rstride=1, cstride=1, linewidth=0, shade=True)
    
    # Set visualization parameters
    ax.set_box_aspect([1, 1, 1])  # Make axes equal
    ax.set_xlim([-L/2, L/2])
    ax.set_ylim([-L/2, L/2])
    ax.set_zlim([-L/2, L/2])
    ax.grid(True)
    
    # Set view angle
    ax.view_init(elev=45, azim=45)

R = 1
L = ((4/3)*np.pi*1000/0.04) ** (1/3)

pos = np.loadtxt('posData.out')
posB = np.loadtxt('posBData.out')
energy = np.loadtxt('energy.out')

#clustering = DBSCAN(eps=2.1, min_samples=2).fit(pos)

#plt.plot(np.diff(energy))

#plt.hist(clustering.labels_)
#plt.show()

fig = plt.figure()
plot_spheres(fig, 121, pos[:, 0], pos[:, 1], pos[:, 2], R, 'silver', L, posB[:, 0], posB[:, 1], posB[:, 2], 0.25*R, 'k')

#plot_spheres(fig, 122, pos[:, 0], pos[:, 1], pos[:, 2], R, 'silver', L, posB[:, 0], posB[:, 1], posB[:, 2], 0.25*R, 'k')

#plt.tight_layout()
plt.show()

