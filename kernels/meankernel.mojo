from compiler import register
from gpu import barrier, block_dim, block_idx, thread_idx
from gpu.memory import AddressSpace, external_memory
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from sys import alignof, sizeof


@register("mean_1d_vector")
struct Mean1DVector:
    """Finds the mean (average) of a single 1D vector using a GPU."""

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
        fn mean_vector_gpu[block_size: Int, vector_size: Int](
            out_vals: __type_of(out_vals),
            in_vals: __type_of(in_vals),
        ):
            var tid = thread_idx.x
            
            var shared_mem = external_memory[
                Scalar[dtype],
                address_space=AddressSpace.SHARED,
                alignment=alignof[Scalar[dtype]](),
            ]()

            # Initialize with 0 for the summation.
            var thread_sum_val = Scalar[dtype](0)

            var i = tid
            while i < vector_size:
                # Accumulate the sum.
                thread_sum_val += in_vals[i]
                i += block_size

            shared_mem[tid] = thread_sum_val
            barrier()

            # --- Block-wide Parallel Reduction for SUM ---
            var stride = block_size // 2
            while stride > 0:
                if tid < stride:
                    # Reduce by adding the partial sums.
                    shared_mem[tid] += shared_mem[tid + stride]
                barrier()
                stride //= 2

            # After reduction, shared_mem[0] holds the total sum.
            # Only thread 0 calculates and writes the final mean.
            if tid == 0:
                # Divide total sum by the number of elements.
                # Cast vector_size to dtype to ensure floating-point division.
                out_vals[0] = shared_mem[0] / 100

        @parameter
        if target == "gpu":
            var shape = in_vals.shape()
            var vector_size = shape[0]
            var block_size = 256
            
            dev_ctx.enqueue_function[mean_vector_gpu[256, 100]](
                out_vals,
                in_vals,
                grid_dim=1,
                block_dim=block_size,
                shared_mem_bytes=block_size * sizeof[Scalar[dtype]](),
            )