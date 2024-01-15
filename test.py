from osborn import *
import utils
import numpy as np
import scipy

def validate_balance(original_matrix, balanced_matrix):
    # Check balancing
    row_sums_balanced = balanced_matrix.sum(axis=1)
    col_sums_balanced = balanced_matrix.sum(axis=0)

    diff = row_sums_balanced - col_sums_balanced
    diff_check = np.all(np.abs(diff) < 1e-6)



    # Compare eigenvalues before and after balancing
    eigenvalues_original, _ = np.linalg.eig(original_matrix)
    eigenvalues_balanced, _ = np.linalg.eig(balanced_matrix)
    eigenvalues_diff = np.abs(eigenvalues_original - eigenvalues_balanced)
    eigenvalues_check = np.all(eigenvalues_diff < 1e-6)

    print(f"Diff:\n{diff}")
    print(f"Eigenvalues difference:\n{eigenvalues_diff}")

    return diff_check and eigenvalues_check

# Example usage
original_matrix = scipy.sparse.coo_array(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64))
# original_matrix = np.zeros((3,3))
# original_matrix[0][1] = .5
# original_matrix[0][2] = 1.5
# original_matrix[1][2] = .25

balancer = MatrixBalancer(original_matrix, device='cpu', epsilon=1e-10)
balanced_matrix, _ = balancer.balance(method='cyclic')

is_valid = validate_balance(original_matrix.toarray(), balanced_matrix)

if is_valid:
    print("Balancing is valid.")
else:
    print("Balancing is not valid.")
