import numpy as np

A = np.array([[1, 3], [2, 4]])
B = np.array([[5, 7], [6, 8]])

print("Matrix A:")
print(A)
print("Matrix B:")
print(B)

print("A (x) B.T:")
C = np.kron(A, B.T)
print(C)

m, n = B.shape
dim = m * n
indices = np.arange(dim)
i_grid, j_grid = np.meshgrid(indices, indices, indexing='ij')

print("Row-major:")
print("vec(A) (x) vec(B).T:")
vec_A = A.reshape(-1, 1, order='C')
vec_B = B.reshape(-1, 1, order='C')
vec_C = vec_A @ vec_B.T
print(vec_C)
print("Authors:")
row_map = (i_grid % m) * n + (j_grid % n)
col_map = (i_grid // m) * m + (j_grid // m)
print(vec_C[row_map, col_map], np.allclose(C, vec_C[row_map, col_map]))
print("Users:")
row_map = (i_grid // n) * n + (j_grid // m)
col_map = (j_grid % m) * n + (i_grid % n)
print(vec_C[row_map, col_map], np.allclose(C, vec_C[row_map, col_map]))

print("Column-major:")
print("vec(A) (x) vec(B).T:")
vec_A = A.reshape(-1, 1, order='F')
vec_B = B.reshape(-1, 1, order='F')
vec_C = vec_A @ vec_B.T
print(vec_C)
print("Authors:")
row_map = (i_grid % m) * n + (j_grid % n)
col_map = (i_grid // m) * m + (j_grid // m)
print(vec_C[row_map, col_map], np.allclose(C, vec_C[row_map, col_map]))
print("Users:")
row_map = (j_grid // m) * m + (i_grid // n)
col_map = (i_grid % n) * m + (j_grid % m)
print(vec_C[row_map, col_map], np.allclose(C, vec_C[row_map, col_map]))



