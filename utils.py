import numpy as np
import scipy
import networkx as nx

try:
    import cupy as cp
    has_cuda = True
except ImportError:
    has_cuda = False


def generate_sparse_matrix(n, m, device='gpu'):
    if device == 'gpu' and has_cuda:
        M = cp.zeros((n,n))
    else:
        M = np.zeros((n,n))

    # Generate positions 
    positions = set()

    while len(positions) < m:
        i = np.random.randint(0, n-1)
        j = np.random.randint(0, n-1)
        positions.add((i, j))

    # Assign values
    if device == 'gpu' and has_cuda:
        for (i,j) in positions:
            M[i,j] = cp.random.rand()
    else:
        for (i,j) in positions:
            M[i,j] = np.random.rand()
    return M


def generate_matrix_params(n, m, kappa, device='gpu'):
    density = m/(n**2)
    matrix = scipy.sparse.random(n, n, density=density).toarray()

    #####
    # we solve for x : (sum(K) + m * x)/(K_min + x) = kappa
    # to get = kappa * (K_min + x) = (sum(K) + m * x)
    # so : x = (kappa * K_min - sum(K)) / (m - kappa)
    #####
    additive_constant = (kappa * np.min(matrix[matrix != 0]) - np.sum(matrix)) / (m - kappa)
    matrix[matrix != 0] += additive_constant

    if device == 'gpu' and has_cuda:
        return cp.array(matrix)
    else:
        return matrix

def diameter_of_matrix(K):
    # Convert the adjacency matrix to a NetworkX graph  
    Gk = nx.from_numpy_array(K)

    # Compute the diameter of the graph
    if nx.is_connected(Gk):
        return nx.diameter(Gk)
    else:
        return np.inf