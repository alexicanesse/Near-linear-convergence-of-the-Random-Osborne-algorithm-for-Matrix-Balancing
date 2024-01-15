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
        while criterion >= self._epsilon:
            self._update_x()
            #self._update_balanced()
            criterion = self._balancing_criterion()
            it += 1
            #print(criterion)

        #self._update_balanced()
        return self._balanced_matrix.toarray(), it

class MatrixBalancerSparse:
    def __init__(self, matrix, epsilon=1e-6, device='cpu'):
        if not self._is_square(matrix):
            raise ValueError("Matrix must be square for matrix balancing.")

        # We only support square matrices
        self._n, _ = matrix.shape
        self._device = device
        self._epsilon = epsilon
        self._next_updates = []
        self._update_method = ...

        self._original_matrix = matrix.copy()
        self._original_matrix_csc = matrix.asformat("csc")
        self._original_matrix_csr = matrix.asformat("csr")
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
    
    def _balancing_criterion(self, order=1):
        criterion = scipy.linalg.norm((self._balanced_matrix - self._balanced_matrix.T).sum(axis=0), order) / self._balanced_matrix.sum()
        return criterion
    def _update_x(self):
        k = self._update()

        r_k = self._balanced_matrix[[k]].sum()
        c_k = self._balanced_matrix[:,[k]].sum()

        self._x.data[k] *= np.sqrt(c_k/r_k)
        self._x_inv.data[k] *= np.sqrt(r_k/c_k)
    
    def _update_balanced(self):
        
        self._balanced_matrix = self._x @ self._original_matrix_csr @ self._x_inv

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
        while criterion >= self._epsilon:
            self._update_x()
         #   self._update_balanced()
            criterion = self._balancing_criterion()
            it += 1
            #print(criterion)

        #self._update_balanced()
        return self._balanced_matrix.toarray(), it


class MatrixBalancerExp:
    def __init__(self, matrix, epsilon=1e-6, device='cpu'):
        if not self._is_square(matrix):
            raise ValueError("Matrix must be square for matrix balancing.")

        # We only support square matrices
        self._n, _ = matrix.shape
        self._device = device
        self._epsilon = epsilon
        self._next_updates = []
        self._update_method = ...

        self._original_matrix = matrix.copy()
        self._original_matrix_csc = matrix.asformat("csc")
        self._original_matrix_csr = matrix.asformat("csr")
        self._x = np.zeros((self._n, 1))

        # self._original_matrix.setdiag(0)
        # self._original_matrix_csc.setdiag(0)
        # self._original_matrix_csr.setdiag(0)

        self._update_balanced()

        # self._rowsum = np.sum(self._original_matrix, axis=1, keepdims=True)
        # self._colsum = np.sum(self._original_matrix, axis=0, keepdims=True).T

    @property
    def _rowsum(self):
        return np.array(self._balanced_matrix.sum(axis=1))[: , 0]

    @property
    def _colsum(self):
        return np.array(self._balanced_matrix.sum(axis=0))[0]


    def _is_square(self, matrix):
        return matrix.shape[0] == matrix.shape[1]
    
    def _balancing_criterion(self, order=1):
        # print(self._rowsum.shape, self._colsum.T.shape, (self._rowsum - self._colsum.T).data, self._rowsum, self._colsum.T)
        criterion = np.linalg.norm((self._rowsum - self._colsum), order) / self._balanced_matrix.sum()
        return criterion

    # def _cliped_exp(self, x):
    #     return np.exp(np.clip(x, -1.e1, 1.e1))

    def _update_x(self):
        k = self._update()

        # self._rowsum -= np.exp(-self._x[k]) * (self._original_matrix_csc[:, [k]] * np.exp(self._x))
        # self._colsum -= np.exp(self._x[k]) * (self._original_matrix_csr[[k]].T * np.exp(-self._x))

        #print(self._rowsum, self._colsum)
        r_k = self._rowsum[k]
        c_k = self._colsum[k]

        if c_k and r_k:
            self._x[k] += .5 * (np.log(c_k) - np.log(r_k))
        elif c_k == 0:
            self._x[k] = -np.inf 
        else: # r_k == 0
            self._x[k] = np.inf

        # self._rowsum += np.exp(-self._x[k]) * (self._original_matrix_csc[:, [k]] * np.exp(self._x))
        # self._colsum += np.exp(self._x[k]) * (self._original_matrix_csr[[k]].T * np.exp(-self._x))

        # self._rowsum[k] = np.exp(self._x[k]) * self._original_matrix_csr[[k]].dot(np.exp(-self._x))
        # self._colsum[k] = np.exp(-self._x[k]) * self._original_matrix_csc[:, [k]].T.dot(np.exp(self._x))

    
    def _update_balanced(self):
        #print(self._balanced_matrix.shape, self._rowsum, self._colsum)
        #self._balanced_matrix = scipy.sparse.spdiags(np.exp(self._x.T), diags=0) @ self._original_matrix_csr @ scipy.sparse.spdiags(np.exp(-self._x.T), diags=0)
        self._balanced_matrix = scipy.sparse.spdiags(np.exp(self._x.T), diags=0, format="csr") @ self._original_matrix_csr @ scipy.sparse.spdiags(np.exp(-self._x.T), diags=0, format="csr")


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
        while criterion >= self._epsilon:
            self._update_x()
            self._update_balanced()
            criterion = self._balancing_criterion()
            it += 1
            #print(criterion)

        self._update_balanced()
        return self._balanced_matrix.toarray(), it




class MatrixBalancerDense:
    def __init__(self, matrix, epsilon=1e-6, device='cpu'):
        if not self._is_square(matrix):
            raise ValueError("Matrix must be square for matrix balancing.")

        # We only support square matrices
        self._n, _ = matrix.shape
        self._device = device
        self._epsilon = epsilon
        self._next_updates = []
        self._update_method = ...

        # TODO TEMP
        for i in range(matrix.shape[0]):
            matrix[i][i] = 0.0

        if self._device == 'gpu' and has_cuda:
            self._original_matrix = cp.asarray(matrix.copy())
            self._balanced_matrix = cp.asarray(matrix.copy())
            self._x = cp.zeros(self._n)
        else:
            self._original_matrix = matrix.copy()
            self._balanced_matrix = matrix.copy()
            self._x = np.zeros(self._n)

        self._rowsum = np.sum(self._original_matrix, axis=1)
        self._colsum = np.sum(self._original_matrix, axis=0)


    def _is_square(self, matrix):
        return matrix.shape[0] == matrix.shape[1]
    
    def _balancing_criterion(self, order=1):
        if self._device == 'gpu' and has_cuda:
            # TODO
            ...
            #criterion = cp.linalg.norm(row_sums - self._colsum, order) / cp.sum(row_sums)
        else:
            criterion = np.linalg.norm(self._rowsum - self._colsum, order) / np.sum(self._rowsum)

        return criterion

    def _cliped_exp(self, x):
        return np.exp(np.clip(x, -1.e1, 1.e1))

    def _update_x(self):
        k = self._update()
        if self._device == 'gpu' and has_cuda:
            #TODO
            ...
            # c_k = cp.sum(self._balanced_matrix[:, k])
            # r_k = cp.sum(self._balanced_matrix[k, :])

            # if c_k and r_k:
            #     self._x[k] += .5 * (cp.log(c_k) - cp.log(r_k))
            # elif c_k == 0:
            #     self._x[k] = -np.inf
            # else: # r_k == 0
            #     self._x[k] = np.inf
        else:
            self._rowsum -= self._cliped_exp(-self._x[k]) * np.multiply(self._original_matrix[: , k], self._cliped_exp(self._x))
            self._colsum -= self._cliped_exp(self._x[k]) * np.multiply(self._original_matrix[k], self._cliped_exp(-self._x))

            if self._colsum[k] and self._rowsum[k]:
                self._x[k] += .5 * (np.log(self._colsum[k]) - np.log(self._rowsum[k]))
            elif self._colsum[k] == 0:
                self._x[k] = -np.inf 
            else: # r_k == 0
                self._x[k] = np.inf

            self._rowsum += self._cliped_exp(-self._x[k]) * np.multiply(self._original_matrix[: , k], self._cliped_exp(self._x))
            self._colsum += self._cliped_exp(self._x[k]) * np.multiply(self._original_matrix[k], self._cliped_exp(-self._x))

            self._rowsum[k] = self._cliped_exp(self._x[k]) * np.sum(np.multiply(self._original_matrix[k], self._cliped_exp(-self._x)))
            self._colsum[k] = self._cliped_exp(-self._x[k]) * np.sum(np.multiply(self._original_matrix[:, k], self._cliped_exp(self._x)))

    
    def _update_balanced(self):
        if self._device == 'gpu' and has_cuda:
            self._balanced_matrix = cp.diag(cp.self._cliped_exp(self._x)) @ self._original_matrix @ cp.diag(cp.self._cliped_exp(-self._x))
        else:
            self._balanced_matrix = np.matmul(np.matmul(np.diag(self._cliped_exp(self._x)), self._original_matrix), np.diag(self._cliped_exp(-self._x)))

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
        while criterion >= self._epsilon:
            self._update_x()
            # self._update_balanced()
            criterion = self._balancing_criterion()
            #print("Criterion", criterion)
            it += 1

        self._update_balanced()
        return self._balanced_matrix, it

