import numpy as np

# function to initialize position of beads and map onto periodic domain
def initialBeads(n, L):
    if n == 1:
        return np.zeros((n, 3))
    else:
        return -0.5*L + L*np.random.rand(n, 3)


# map positions onto periodic domain
def periodicLocation(x, L):
    
    x[x < -0.5*L] += L
    x[x >= 0.5*L] -= L
    return x


def periodicDifferences(dX, L):
 
    dX[dX > 0.5*L] -= L
    dX[dX <= -0.5*L] += L
    return dX

def computeDistance(X, Xb, R, N, L):

    #Pairwise difference vectors between beads, then compute norm to get distances
    X_diff = periodicDifferences(X[:, np.newaxis, :] - X[np.newaxis, :, :], L)
    pd_beads = np.sqrt(np.sum(X_diff ** 2, axis=2))

    #Pairwise difference vectors between bacteria and beads, then compute norm to get distances
    X_diffb = periodicDifferences(X[:, np.newaxis, :] - Xb[np.newaxis, :, :], L)
    pd_beadsPatchDist = np.sqrt(np.sum(X_diffb ** 2, axis=2))

    return pd_beads, pd_beadsPatchDist - R


def normalize_q(q):
    
    return q/np.sqrt(np.sum(q*q))

def quat_rot(qr, qi, qj, qk):
    return np.array([[1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr)],[2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr)],[2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2)]])

def rotate_posB(pB_local, maxRot):

    # random vector about which to rotate
    th_vec = np.pi*np.random.rand()
    phi_vec = -np.pi + 2*np.pi*np.random.rand()
    u = np.array([np.sin(th_vec)*np.cos(phi_vec), np.sin(th_vec)*np.sin(phi_vec), np.cos(th_vec)])
    u = u/np.sqrt(np.sum(u*u))

    th_rotate = -maxRot + 2*maxRot*np.random.rand()
    
    #rotation quaternion
    q_rot = normalize_q(np.array([np.cos(0.5*th_rotate), u[0]*np.sin(0.5*th_rotate), u[1]*np.sin(0.5*th_rotate), u[2]*np.sin(0.5*th_rotate)]))
    
    return np.dot(quat_rot(q_rot[0], q_rot[1], q_rot[2], q_rot[3]), pB_local.T).T


def updateLocations(pos, posB_local, selectParticle, patch_id, n_patches, maxDisp, maxRot, L):
    
    pos_temp = np.copy(pos)
    posB_local_temp = np.copy(posB_local)
    
    rand_r = np.random.rand()
    rand_th = np.pi*np.random.rand()
    rand_phi = -np.pi + 2*np.pi*np.random.rand()

    pos_temp[selectParticle, :] += maxDisp*(rand_r ** (1/3))*np.array([np.sin(rand_th)*np.cos(rand_phi), np.sin(rand_th)*np.sin(rand_phi), np.cos(rand_th)])
    pos_temp[selectParticle, :] = periodicLocation(pos_temp[selectParticle, :], L)
    
    posB_local_temp[patch_id == selectParticle, :] = rotate_posB(posB_local_temp[patch_id == selectParticle, :], maxRot)


    posB_temp = periodicLocation(np.repeat(pos_temp, n_patches, axis=0) + posB_local_temp, L)

    return pos_temp, posB_temp, posB_local_temp

