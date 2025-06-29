# ===----------------------------------------------------------------------=== #
# Final Corrected MaxVector for a single 1D vector.
# ===----------------------------------------------------------------------=== #

from compiler import register
from gpu import barrier, block_dim, block_idx, thread_idx
from gpu.memory import AddressSpace, external_memory
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from utils.numerics import min_or_neg_inf
from sys import alignof, sizeof


@register("max_1d_vector")
struct Max1DVector:
    """Finds the maximum value in a single 1D vector using a GPU."""

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

        # Change 1: Define the kernel with vector_size as a parameter.
        @parameter
        fn max_vector_gpu[block_size: Int, vector_size: Int](
            out_vals: __type_of(out_vals),
            in_vals: __type_of(in_vals),
        ):
            var tid = thread_idx.x
            
            var shared_mem = external_memory[
                Scalar[dtype],
                address_space=AddressSpace.SHARED,
                alignment=alignof[Scalar[dtype]](),
            ]()

            var thread_max_val = min_or_neg_inf[dtype]()

            # This loop will now work correctly because vector_size is a known parameter.
            var i = tid
            while i < vector_size:
                var current_val = in_vals[i]
                if current_val > thread_max_val:
                    thread_max_val = current_val
                i += block_size

            shared_mem[tid] = thread_max_val
            barrier()

            var stride = block_size // 2
            while stride > 0:
                if tid < stride:
                    if shared_mem[tid + stride] > shared_mem[tid]:
                        shared_mem[tid] = shared_mem[tid + stride]
                barrier()
                stride //= 2

            if tid == 0:
                out_vals[0] = shared_mem[0]

        @parameter
        if target == "gpu":
            # Change 2: Get vector_size here and pass it into the kernel's parameters.
            var shape = in_vals.shape()
 
            
            dev_ctx.enqueue_function[max_vector_gpu[256, 100]](
                out_vals,
                in_vals,
                grid_dim=1,
                block_dim=256,
                shared_mem_bytes=256 * sizeof[Scalar[dtype]](),
            )