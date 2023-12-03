import numpy as np
import pynq
from pynq import Overlay

# Load the overlay
overlay = Overlay("/home/xilinx/jupyter_notebooks/test/pl_matmul_pynq.bit")

# Access the matrix multiplication IP instance in the overlay
matmul_inst = overlay.pl_matmul_a_0

# Define matrix sizes
M = 16
N = 16
K = 16

# Allocate memory for matrices A, B, and the result AB
matrix_A = pynq.allocate(shape=(M, K), dtype=np.int32)
matrix_B = pynq.allocate(shape=(K, N), dtype=np.int32)
matrix_AB = pynq.allocate(shape=(M, N), dtype=np.int32)

# Initialize matrices A and B with some values
for i in range(M):
    for j in range(K):
        matrix_A[i][j] = np.int32(i + j)

for i in range(K):
    for j in range(N):
        matrix_B[i][j] = np.int32(i - j)

# Initialize matrix AB with -1.0
for i in range(M):
    for j in range(N):
        matrix_AB[i][j] = np.int32(-1)

# Synchronize data to the FPGA
matrix_A.sync_to_device()
matrix_B.sync_to_device()
matrix_AB.sync_to_device()

# Set the physical addresses of matrices A, B, and AB
matmul_inst.mmio.write_reg(0x10, matrix_A.physical_address)
matmul_inst.mmio.write_reg(0x14, 0)
matmul_inst.mmio.write_reg(0x1C, matrix_B.physical_address)
matmul_inst.mmio.write_reg(0x20, 0)
matmul_inst.mmio.write_reg(0x28, matrix_AB.physical_address)
matmul_inst.mmio.write_reg(0x2C, 0)

# Trigger the matrix multiplication IP
matmul_inst.write(0x00, 1)

# Wait for the accelerator to finish
while matmul_inst.read(0x00) == 14:
    pass

# Synchronize data back from the FPGA
matrix_A.sync_from_device()
matrix_B.sync_from_device()
matrix_AB.sync_from_device()

# Print results
print("Matrix A:")
print(matrix_A)
print("Matrix B:")
print(matrix_B)
print("Result AB:")
print(matrix_AB)
