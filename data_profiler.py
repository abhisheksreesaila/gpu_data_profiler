from pathlib import Path
import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Simple parameters for a 1D vector
vector_size = 10
dtype = DType.int32

# Use GPU if available, fallback to CPU
try:
    device = Accelerator() if accelerator_count() > 0 else CPU()
    print(f"Using device: {type(device).__name__}")
except:
    device = CPU()
    print("Using CPU fallback")

# Create the graph
graph = Graph(
    "max_vector_1d",
    forward=lambda x: ops.custom(
        # Use the new kernel name for 1D vectors
        name="max_1d_vector",
        device=DeviceRef.from_device(device),
        values=[x],
        # The output is now a single scalar value, represented as a tensor of shape [1]
        out_types=[
            TensorType(dtype=dtype, shape=[1], device=DeviceRef.from_device(device)),
        ],
        parameters={},
    ),
    # The input is now a 1D tensor
    input_types=[
        TensorType(dtype, shape=[vector_size], device=DeviceRef.from_device(device))
    ],
    custom_extensions=[Path(__file__).parent / "kernels"],
)

# Setup session and load model
session = InferenceSession(devices=[device])
model = session.load(graph)

# Create 1D test data
x_values = np.arange(vector_size, dtype=np.int32) # e.g., [0., 1., 2., ..., 1023.]
print("Input data shape:", x_values.shape)
print("Input data sample:", x_values[:10])


# Create tensor and move to device
x = Tensor.from_numpy(x_values).to(device)

try:
    # Execute the model
    results = model.execute(x)
    max_vals_tensor = results[0]
    
    # Move result back to CPU safely
    max_value = max_vals_tensor.to(CPU()).to_numpy()[0] # Extract the scalar from the array
    
    print("\nResult:")
    print("Max value:", max_value)
    
    # Verify with numpy
    expected_max = np.max(x_values)
    print("\nExpected:")
    print("Max value:", expected_max)

    assert np.isclose(max_value, expected_max), "Verification failed!"
    print("\nVerification successful!")
    
except Exception as e:
    print(f"Error during execution: {e}")
    print("Check that your Mojo kernel is properly compiled in the kernels/ directory")