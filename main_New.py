import numpy as np
import matplotlib.pyplot as plt
import energy_module as em
import geometry_module as gm
from prob_dist import discrete_dist

from mpl_toolkits.mplot3d import Axes3D


numTrials = int(1e8) # number of Monte-Carlo trials

N = 1000  # number of beads
p = 0.5  # probability for binomial distribution
meanPatches = 7  # maximum (or mean) number of patches on each bead

#n_patches = np.random.binomial(meanPatches/p, p, N)
# drawn from custom discrete distribution with mean meanPatches (meanPatches >= 2 for the moment)
n_patches = discrete_dist(N, meanPatches, p, 0.5)


# indices to identify which bead each patch belongs to
patch_id = np.repeat(np.arange(0, N), n_patches)

totalPatches = np.cumsum(n_patches)

L = ((4/3)*np.pi*N/0.04) ** (1/3)  # side-length of computational domain in particle radii, volume fraction 4% consistent with experiments

# Maximum rotation and displacement for MC simulations
maxDisp = 0.1
maxRot = 0.1

# Bead parameters
R = 1  # dimensionless bead radius
sigma = 2*R  # dimensionless bead diameter

# depth (epsilon) and decay (mu) of Morse potential
epsilon = 10
mu = 1

cutoff = 3*mu  # cutoff distance to determine nearest neighbours

# initial bead location
pos = gm.initialBeads(N, L)

#initial bacteria positions
theta = np.pi * np.random.rand(np.sum(n_patches), 1)
phi = -np.pi + 2 * np.pi * np.random.rand(np.sum(n_patches), 1)
posB_local = R * np.hstack((np.sin(theta) * np.cos(phi), 
                          np.sin(theta) * np.sin(phi), 
                          np.cos(theta)))
posB = gm.periodicLocation(np.repeat(pos, n_patches, axis=0) + posB_local, L)

np.savetxt('posData_init_mean7.out', pos, delimiter=' ')
np.savetxt('posBData_init_mean7.out', posB, delimiter=' ')

#fig = plt.figure()
##plot_spheres(fig, 121, pos[:, 0], pos[:, 1], pos[:, 2], R, 'silver', L, posB[:, 0], posB[:, 1], posB[:, 2], 0.25*R, 'k')

#fig = plt.figure()
#plot_spheres(fig, 121, posInit[:, 0], posInit[:, 1], posInit[:, 2], R, 'silver', L, posBInit[:, 0], posBInit[:, 1], posBInit[:, 2], 0.25*R, 'k')
#plt.show()

pd_beads, pd_patchGhost = gm.computeDistance(pos, posB, R, N, L)

# nearest neighbour matrix
neighbourDist = pd_beads < cutoff

energyOut = np.zeros(10000)
k = 0
energyOut[k] = em.vHS(pd_beads, sigma) + em.vMorse(pd_patchGhost, epsilon, mu)

for i in range(1, numTrials):
    
    if N > 1:
       selectParticle = np.random.randint(0, N)
    else:
        selectParticle = 0

    pos_Trial, posB_Trial, posB_local_Trial = gm.updateLocations(pos, posB_local, selectParticle, patch_id, n_patches, maxDisp, maxRot, L)

 
    deltaE = em.updateEnergy(pos, posB, pos_Trial, posB_Trial, neighbourDist, selectParticle, patch_id, n_patches, sigma, epsilon, mu, R, L)
    #energy += deltaE

    if (deltaE <= 0) or (np.exp(-deltaE)) >= np.random.rand():
        
        pos = pos_Trial
        posB = posB_Trial
        posB_local = posB_local_Trial
        

    # update nearest neighbour matrix
    if np.mod(i, 10000) == 0:
        k += 1
        pd_beads, pd_patchGhost = gm.computeDistance(pos, posB, R, N, L)
        neighbourDist = pd_beads < cutoff
        energyOut[k] = em.vHS(pd_beads, sigma) + em.vMorse(pd_patchGhost, epsilon, mu)

    if i % (numTrials // 10) == 0:
        print(f"Simulation is {i*100/numTrials}% complete")
        #print(energy)

np.savetxt('posData_mean7.out', pos)
np.savetxt('posBData_mean7.out', posB)
np.savetxt('energy_mean7.out', energyOut)
#plot_spheres(fig, 122, pos[:, 0], pos[:, 1], pos[:, 2], R, 'silver', L, posB[:, 0], posB[:, 1], posB[:, 2], 0.25*R, 'k')

#plt.tight_layout()
#plt.show()



