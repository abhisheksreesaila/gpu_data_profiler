from pathlib import Path
import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Simple parameters for a 1D vector
vector_size = 100
# Let's use float32 for input to test the mean properly
input_dtype = DType.float32

# Use GPU if available, fallback to CPU
try:
    device = Accelerator() if accelerator_count() > 0 else CPU()
    print(f"Using device: {type(device).__name__}")
except:
    device = CPU()
    print("Using CPU fallback")

# --- Define Graph for MAX operation ---
max_graph = Graph(
    "max_vector_1d",
    forward=lambda x: ops.custom(
        name="max_1d_vector",
        device=DeviceRef.from_device(device),
        values=[x],
        out_types=[
            TensorType(dtype=input_dtype, shape=[1], device=DeviceRef.from_device(device)),
        ],
        parameters={},
    ),
    input_types=[
        TensorType(input_dtype, shape=[vector_size], device=DeviceRef.from_device(device))
    ],
    custom_extensions=[Path(__file__).parent / "kernels"],
)

# --- Define Graph for MIN operation ---
min_graph = Graph(
    "min_vector_1d",
    forward=lambda x: ops.custom(
        name="min_1d_vector",
        device=DeviceRef.from_device(device),
        values=[x],
        out_types=[
            TensorType(dtype=input_dtype, shape=[1], device=DeviceRef.from_device(device)),
        ],
        parameters={},
    ),
    input_types=[
        TensorType(input_dtype, shape=[vector_size], device=DeviceRef.from_device(device))
    ],
    custom_extensions=[Path(__file__).parent / "kernels"],
)

# --- Define Graph for MEAN operation ---
mean_graph = Graph(
    "mean_vector_1d",
    forward=lambda x: ops.custom(
        name="mean_1d_vector",
        device=DeviceRef.from_device(device),
        values=[x],
        # The output of mean should be float32 for precision
        out_types=[
            TensorType(dtype=DType.float32, shape=[1], device=DeviceRef.from_device(device)),
        ],
        parameters={},
    ),
    input_types=[
        TensorType(input_dtype, shape=[vector_size], device=DeviceRef.from_device(device))
    ],
    custom_extensions=[Path(__file__).parent / "kernels"],
)


# Setup session and load all three models
session = InferenceSession(devices=[device])
max_model = session.load(max_graph)
min_model = session.load(min_graph)
mean_model = session.load(mean_graph)


# Create 1D test data
x_values = np.arange(vector_size, dtype=np.float32)
np.random.shuffle(x_values)
print("Input data shape:", x_values.shape)
print("Input data sample:", x_values[:10])


# Create tensor and move to device
x = Tensor.from_numpy(x_values).to(device)

try:
    # --- Execute and Verify MAX ---
    print("\n--- Max Operation ---")
    max_results = max_model.execute(x)
    max_value = max_results[0].to(CPU()).to_numpy()[0]
    print("Result:", max_value)
    expected_max = np.max(x_values)
    print("Expected:", expected_max)
    assert np.isclose(max_value, expected_max), "Max verification failed!"
    print("Max verification successful!")

    # --- Execute and Verify MIN ---
    print("\n--- Min Operation ---")
    min_results = min_model.execute(x)
    min_value = min_results[0].to(CPU()).to_numpy()[0]
    print("Result:", min_value)
    expected_min = np.min(x_values)
    print("Expected:", expected_min)
    assert np.isclose(min_value, expected_min), "Min verification failed!"
    print("Min verification successful!")
    
    # --- Execute and Verify MEAN ---
    print("\n--- Mean Operation ---")
    mean_results = mean_model.execute(x)
    mean_value = mean_results[0].to(CPU()).to_numpy()[0]
    print("Result:", mean_value)
    expected_mean = np.mean(x_values)
    print("Expected:", expected_mean)
    assert np.isclose(mean_value, expected_mean), "Mean verification failed!"
    print("Mean verification successful!")

except Exception as e:
    print(f"Error during execution: {e}")
    print("Check that your Mojo kernel is properly compiled in the kernels/ directory")