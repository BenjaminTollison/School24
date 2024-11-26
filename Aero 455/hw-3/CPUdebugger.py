import cupy as cp

# Create a simple CuPy array
gpu_array = cp.array([1, 2, 3, 4, 5])
gpu_array = cp.random.random((100, 100))
print("X shape:", gpu_array.shape)
print("X dtype:", gpu_array.dtype)
print("X size (bytes):", gpu_array.nbytes)


# Attempt to transfer it to NumPy
try:
    cpu_array = gpu_array.get()
    print("Successfully transferred to NumPy:", cpu_array)
except Exception as e:
    print("Error during array transfer:", e)
