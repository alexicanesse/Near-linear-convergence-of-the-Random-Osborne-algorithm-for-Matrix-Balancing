import numpy as np
import scipy
import networkx as nx

try:
    import cupy as cp
    has_cuda = True
except ImportError:
    has_cuda = False


def generate_sparse_matrix(n, m, device='gpu'):
    density = m/(n**2)

    # We need to generate matrix that is balanceable
    while True:
        matrix = scipy.sparse.random(n, n, density=density, format="csr")
        matrix.data = np.random.exponential(scale=1.0, size=matrix.data.shape)

        if is_balanceable(matrix.toarray()):
            break

    return matrix


def generate_matrix_params(n, m, kappa, device='gpu'):
    # m < kappa is mandatory otherwise it cannot work due to the definition of kappa
    if m > kappa:
        raise ValueError("Error: kappa should be greater than or equal to m.")
    

    density = m/(n**2)

    # We need to generate a non negative matrix that is balanceable
    while True:
        matrix = scipy.sparse.random(n, n, density=density, format="csr")
        matrix.data = np.random.exponential(scale=1.0, size=matrix.data.shape)

        #####
        # we solve for x : (sum(K) + m * x)/(K_min + x) = kappa
        # to get = kappa * (K_min + x) = (sum(K) + m * x)
        # so : x = (kappa * K_min - sum(K)) / (m - kappa)
        #####
        additive_constant = (kappa * np.min(matrix.data) - matrix.sum()) / (m - kappa)
        matrix.data += additive_constant
        if np.all(matrix.data >= 0) and is_balanceable(matrix.toarray()):
            break

    if device == 'gpu' and has_cuda:
        return cp.array(matrix)
    else:
        return matrix


def diameter_of_matrix(K):
    # Convert the adjacency matrix to a NetworkX graph  
    Gk = nx.DiGraph(K)

    # Compute the diameter of the graph
    if nx.is_strongly_connected(Gk):
        return nx.diameter(Gk)
    else:
        return np.inf
    
def is_balanceable(K):
    # Convert the adjacency matrix to a NetworkX graph  
    Gk = nx.DiGraph(K)
    return nx.is_strongly_connected(Gk)

