from compiler import register
from gpu import barrier, block_dim, block_idx, thread_idx
from gpu.memory import AddressSpace, external_memory
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from sys import alignof, sizeof


# NOTE: We no longer need to import from utils.numerics for this kernel.

@register("min_1d_vector")
struct Min1DVector:
    """Finds the minimum value in a single 1D vector using a GPU."""

    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
    ](
        out_vals: OutputTensor[dtype=dtype, rank=1],
        in_vals: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[rank == 1, "rank must be 1"]()
        
        var dev_ctx = ctx.get_device_context()

        @parameter
        fn min_vector_gpu[block_size: Int, vector_size: Int](
            out_vals: __type_of(out_vals),
            in_vals: __type_of(in_vals),
        ):
            var tid = thread_idx.x
            
            var shared_mem = external_memory[
                Scalar[dtype],
                address_space=AddressSpace.SHARED,
                alignment=alignof[Scalar[dtype]](),
            ]()

            # --- Robust Initialization ---
            # Each thread initializes its local minimum with the first value it
            # is responsible for. This is safer than relying on a library
            # function for positive infinity.
            var thread_min_val: Scalar[dtype]
            if tid < vector_size:
                thread_min_val = in_vals[tid]
            else:
                # If a thread is outside the vector's bounds (e.g. vector_size=100,
                # block_size=256), initialize it with a true maximum value so it
                # doesn't interfere with the reduction. For int32, this is 2^31 - 1.
                thread_min_val = 2147483647

            # Start the loop at the *next* element for this thread.
            var i = tid + block_size
            while i < vector_size:
                var current_val = in_vals[i]
                if current_val < thread_min_val:
                    thread_min_val = current_val
                i += block_size

            shared_mem[tid] = thread_min_val
            barrier()

            # The reduction part remains the same
            var stride = block_size // 2
            while stride > 0:
                if tid < stride:
                    if shared_mem[tid + stride] < shared_mem[tid]:
                        shared_mem[tid] = shared_mem[tid + stride]
                barrier()
                stride //= 2

            if tid == 0:
                out_vals[0] = shared_mem[0]

        @parameter
        if target == "gpu":
            var shape = in_vals.shape()

            
            dev_ctx.enqueue_function[min_vector_gpu[256, 100]](
                out_vals,
                in_vals,
                grid_dim=1,
                block_dim=256,
                shared_mem_bytes=256 * sizeof[Scalar[dtype]](),
            )