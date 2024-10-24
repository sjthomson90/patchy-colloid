import numpy as np

# Create a discrete distribution to define the number of patches per bead
# The probability distribution is symmetric: p = [alpha*p1, p1, P, p1, alpha*p1] where 0 < alpha < 1
# The possible choices are X = [N - 2, N - 1, N, N + 1, N + 2]

# N is the mean number of patches, P is the probability of N, alpha is the ratio p1/p2
def discrete_dist(N_beads, N_bac, P, alpha):
    p1 = 0.5*((1 - P)/(1 + alpha))
    probs = np.array([alpha*p1, p1, P, p1, alpha*p1])
    X = np.array([N_bac - 2, N_bac - 1, N_bac, N_bac + 1, N_bac + 2])
    return np.random.choice(X, size = N_beads, p = probs)
