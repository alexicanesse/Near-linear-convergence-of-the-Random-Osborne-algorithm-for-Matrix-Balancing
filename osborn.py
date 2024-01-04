import numpy as np
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

        if self._device == 'gpu' and has_cuda:
            self._original_matrix = cp.asarray(matrix.copy())
            self._balanced_matrix = cp.asarray(matrix.copy())
            self._x = cp.zeros(self._n)
        else:
            self._original_matrix = matrix.copy()
            self._balanced_matrix = matrix.copy()
            self._x = np.zeros(self._n)

    def _is_square(self, matrix):
        return matrix.shape[0] == matrix.shape[1]
    
    def _balancing_criterion(self, order=1):
        row_sums = self._rowsum

        if self._device == 'gpu' and has_cuda:
            criterion = cp.linalg.norm(row_sums - self._colsum, order) / cp.sum(row_sums)
        else:
            criterion = np.linalg.norm(row_sums - self._colsum, order) / np.sum(row_sums)

        return criterion

    @property
    def _rowsum(self):
        if self._device == 'gpu' and has_cuda:
            return cp.sum(self._balanced_matrix, axis=1)
        else:
            return np.sum(self._balanced_matrix, axis=1)

    @property
    def _colsum(self):
        if self._device == 'gpu' and has_cuda:
            return cp.sum(self._balanced_matrix, axis=0)
        else:
            return np.sum(self._balanced_matrix, axis=0)
        
    def _update_x(self):
        k = self._update()
        if self._device == 'gpu' and has_cuda:
            c_k = cp.sum(self._balanced_matrix[:, k])
            r_k = cp.sum(self._balanced_matrix[k, :])
            self._x[k] += .5 * (cp.log(c_k) - cp.log(r_k))
        else:
            c_k = np.sum(self._balanced_matrix[:, k])
            r_k = np.sum(self._balanced_matrix[k, :])
            self._x[k] += .5 * (np.log(c_k) - np.log(r_k))
    
    def _update_balanced(self):
        if self._device == 'gpu' and has_cuda:
            self._balanced_matrix = cp.diag(cp.exp(self._x)) @ self._original_matrix @ cp.diag(cp.exp(- self._x))
        else:
            self._balanced_matrix = np.diag(np.exp(self._x)) @ self._original_matrix @ np.diag(np.exp(- self._x))

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
        while criterion >= self._epsilon:
            criterion = self._balancing_criterion()
            self._update_x()
            self._update_balanced()

        return self._balanced_matrix

