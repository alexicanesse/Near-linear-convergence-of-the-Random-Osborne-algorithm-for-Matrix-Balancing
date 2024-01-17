import numpy as np
import scipy
from utils import *

try:
    import cupy as cp
    has_cuda = True
except ImportError:
    has_cuda = False

class MatrixBalancer:
    def __init__(self, matrix, epsilon=1e-6, device='cpu'):
        if not self._is_square(matrix):
            raise ValueError("Matrix must be square for matrix balancing.")

        # We only support square matrices
        self._n, _ = matrix.shape
        self._device = device
        self._epsilon = epsilon
        self._next_updates = []
        self._update_method = ...

        self._matrix = matrix.asformat("csr")
        self._x = scipy.sparse.csr_array(np.eye(self._n))
        self._x_inv = scipy.sparse.csr_array(np.eye(self._n))

        self._update_balanced()

    @property
    def _rowsum(self):
        return np.array(self._balanced_matrix.sum(axis=1))

    @property
    def _colsum(self):
        return np.array(self._balanced_matrix.sum(axis=0))


    def _is_square(self, matrix):
        return matrix.shape[0] == matrix.shape[1]
    
    def _balancing_criterion(self):
        criterion = abs((self._balanced_matrix - self._balanced_matrix.T).sum(axis=0)).sum() / self._balanced_matrix.sum()
        return criterion
    
    def _update_x(self):
        k = self._update()

        r_k = self._balanced_matrix[[k]].sum()
        c_k = self._balanced_matrix[:,[k]].sum()

        self._x.data[k] *= np.sqrt(c_k/r_k)
        self._x_inv.data[k] *= np.sqrt(r_k/c_k)

        self._balanced_matrix[[k]] *= np.sqrt(c_k/r_k)
        self._balanced_matrix[:, [k]] *= np.sqrt(r_k/c_k)

    
    def _update_balanced(self):
        self._balanced_matrix = self._x @ self._matrix @ self._x_inv

    def _update(self):
        if not(self._next_updates):
            self._update_method()

        return self._next_updates.pop()
        
    def _cyclic_update(self):
        if not(self._next_updates):
            self._next_updates = list(range(self._n))
    
    def _random_cyclic_update(self):
        if not(self._next_updates):
            self._next_updates = list(np.arange(start=0, stop=self._n, step=1))
            np.random.shuffle(self._next_updates)
    
    def _greedy_update(self):
        self._next_updates = [np.argmax(np.abs(np.sqrt(self._rowsum) - np.sqrt(self._colsum)))]
    
    def _random_update(self):
        if not(self._next_updates):
            self._next_updates = list(np.random.randint(low=0, high=self._n, size=self._n))
    
    def balance(self, method='cyclic'):
        criterion = self._balancing_criterion()

        if method == 'cyclic':
            self._update_method = self._cyclic_update
        elif method == 'random_cyclic':
            self._update_method = self._random_cyclic_update     
        elif method == 'greedy':
            self._update_method = self._greedy_update
        elif method == 'random':
            self._update_method = self._random_update
        else:
            raise ValueError("Invalid method. Please choose from 'cyclic', 'random_cyclic', 'greedy', or 'random'.")
        
        it = 0
        criterions = []
        while criterion >= self._epsilon:
            self._update_x()
            criterion = self._balancing_criterion()
            criterions.append(criterion)            
            it += 1

        return self._balanced_matrix.toarray(), it, criterions
