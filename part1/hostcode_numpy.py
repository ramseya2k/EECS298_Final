import numpy as np
import time

# Define matrix sizes
M = 16
N = 16
K = 16

# Allocate memory for matrices A, B, and the result AB
matrix_A = np.random.randint(10, size=(M, K), dtype=np.int32)
matrix_B = np.random.randint(10, size=(K, N), dtype=np.int32)

# Perform matrix multiplication on the CPU using NumPy
start_time_cpu = time.time()
matrix_AB_cpu = np.matmul(matrix_A, matrix_B)
end_time_cpu = time.time()
execution_time_cpu = end_time_cpu - start_time_cpu

# Print results and execution time for CPU
print("Matrix A:")
print(matrix_A)
print("Matrix B:")
print(matrix_B)
print("Result AB (CPU):")
print(matrix_AB_cpu)
print(f"Execution Time (CPU): {execution_time_cpu:.6f} seconds")
