import numpy as np
import geometry_module as gm

# hard-sphere potential
def vHS(pd, sigma):
    V = np.zeros(np.shape(pd))
    V[pd < sigma] = 1e8
    V[pd == 0] = 0
    return 0.5*np.sum(V)


# Morse potential
def vMorse(pd, epsilon, mu):
    return np.sum(epsilon*(np.exp(-2*pd/mu) - 2*np.exp(-pd/mu)))

def updateEnergy(pos, posB, pos_Trial, posB_Trial, neighbourDist, selectParticle, patch_id, n_patches, sigma, epsilon, mu, R, L):

    neighbours = np.where(neighbourDist[selectParticle, :] == True)[0]

    # if the particle has only itself as a neighbour, then energy change is zero
    if len(neighbours) == 1:
        return 0
    
    else:
        
        #Old energy
        pos_SelectOld = pos[neighbours, :]
        posB_SelectOld = posB[np.isin(patch_id, neighbours), :]
        
        pd_beadsOld, pd_patchGhostOld = gm.computeDistance(pos_SelectOld, posB_SelectOld, R, len(neighbours), L)
        energyOld = vHS(pd_beadsOld, sigma) + vMorse(pd_patchGhostOld, epsilon, mu)

        #New energy
        pos_SelectNew = pos_Trial[neighbours, :]
        posB_SelectNew = posB_Trial[np.isin(patch_id, neighbours), :]
        pd_beadsNew, pd_patchGhostNew= gm.computeDistance(pos_SelectNew, posB_SelectNew, R, len(neighbours), L)
        energyNew = vHS(pd_beadsNew, sigma) + vMorse(pd_patchGhostNew, epsilon, mu)

        return energyNew - energyOld