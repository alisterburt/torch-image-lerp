import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
cuda_source = '''
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuComplex.h>

// Complex atomic add helper
__device__ void atomicAdd(cuFloatComplex* address, cuFloatComplex val) {
    float* real_addr = (float*)address;
    float* imag_addr = real_addr + 1;
    atomicAdd(real_addr, cuCrealf(val));
    atomicAdd(imag_addr, cuCimagf(val));
}

// 2D kernel for float
__global__ void accumulate_2d_kernel_float(
    const int64_t* indices_0,
    const int64_t* indices_1, 
    const float* values,
    float* out,
    int64_t n_values,
    int64_t out_size_0,
    int64_t out_size_1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_values) {
        int64_t idx_0 = indices_0[idx];
        int64_t idx_1 = indices_1[idx];

        if (idx_0 >= 0 && idx_0 < out_size_0 && 
            idx_1 >= 0 && idx_1 < out_size_1) {
            int64_t out_idx = idx_0 * out_size_1 + idx_1;
            atomicAdd(&out[out_idx], values[idx]);
        }
    }
}

// 2D kernel for complex
__global__ void accumulate_2d_kernel_complex(
    const int64_t* indices_0,
    const int64_t* indices_1, 
    const cuFloatComplex* values,
    cuFloatComplex* out,
    int64_t n_values,
    int64_t out_size_0,
    int64_t out_size_1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_values) {
        int64_t idx_0 = indices_0[idx];
        int64_t idx_1 = indices_1[idx];

        if (idx_0 >= 0 && idx_0 < out_size_0 && 
            idx_1 >= 0 && idx_1 < out_size_1) {
            int64_t out_idx = idx_0 * out_size_1 + idx_1;
            atomicAdd(&out[out_idx], values[idx]);
        }
    }
}

// 3D kernel for float
__global__ void accumulate_3d_kernel_float(
    const int64_t* indices_0,
    const int64_t* indices_1,
    const int64_t* indices_2,
    const float* values,
    float* out,
    int64_t n_values,
    int64_t out_size_0,
    int64_t out_size_1,
    int64_t out_size_2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_values) {
        int64_t idx_0 = indices_0[idx];
        int64_t idx_1 = indices_1[idx];
        int64_t idx_2 = indices_2[idx];

        if (idx_0 >= 0 && idx_0 < out_size_0 && 
            idx_1 >= 0 && idx_1 < out_size_1 &&
            idx_2 >= 0 && idx_2 < out_size_2) {
            int64_t out_idx = idx_0 * out_size_1 * out_size_2 + 
                             idx_1 * out_size_2 + idx_2;
            atomicAdd(&out[out_idx], values[idx]);
        }
    }
}

// 3D kernel for complex
__global__ void accumulate_3d_kernel_complex(
    const int64_t* indices_0,
    const int64_t* indices_1,
    const int64_t* indices_2,
    const cuFloatComplex* values,
    cuFloatComplex* out,
    int64_t n_values,
    int64_t out_size_0,
    int64_t out_size_1,
    int64_t out_size_2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_values) {
        int64_t idx_0 = indices_0[idx];
        int64_t idx_1 = indices_1[idx];
        int64_t idx_2 = indices_2[idx];

        if (idx_0 >= 0 && idx_0 < out_size_0 && 
            idx_1 >= 0 && idx_1 < out_size_1 &&
            idx_2 >= 0 && idx_2 < out_size_2) {
            int64_t out_idx = idx_0 * out_size_1 * out_size_2 + 
                             idx_1 * out_size_2 + idx_2;
            atomicAdd(&out[out_idx], values[idx]);
        }
    }
}

// 4D kernel for float
__global__ void accumulate_4d_kernel_float(
    const int64_t* indices_0,
    const int64_t* indices_1,
    const int64_t* indices_2,
    const int64_t* indices_3,
    const float* values,
    float* out,
    int64_t n_values,
    int64_t out_size_0,
    int64_t out_size_1,
    int64_t out_size_2,
    int64_t out_size_3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_values) {
        int64_t idx_0 = indices_0[idx];
        int64_t idx_1 = indices_1[idx];
        int64_t idx_2 = indices_2[idx];
        int64_t idx_3 = indices_3[idx];

        if (idx_0 >= 0 && idx_0 < out_size_0 && 
            idx_1 >= 0 && idx_1 < out_size_1 &&
            idx_2 >= 0 && idx_2 < out_size_2 &&
            idx_3 >= 0 && idx_3 < out_size_3) {
            int64_t out_idx = idx_0 * out_size_1 * out_size_2 * out_size_3 + 
                             idx_1 * out_size_2 * out_size_3 + 
                             idx_2 * out_size_3 + idx_3;
            atomicAdd(&out[out_idx], values[idx]);
        }
    }
}

// 4D kernel for complex
__global__ void accumulate_4d_kernel_complex(
    const int64_t* indices_0,
    const int64_t* indices_1,
    const int64_t* indices_2,
    const int64_t* indices_3,
    const cuFloatComplex* values,
    cuFloatComplex* out,
    int64_t n_values,
    int64_t out_size_0,
    int64_t out_size_1,
    int64_t out_size_2,
    int64_t out_size_3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_values) {
        int64_t idx_0 = indices_0[idx];
        int64_t idx_1 = indices_1[idx];
        int64_t idx_2 = indices_2[idx];
        int64_t idx_3 = indices_3[idx];

        if (idx_0 >= 0 && idx_0 < out_size_0 && 
            idx_1 >= 0 && idx_1 < out_size_1 &&
            idx_2 >= 0 && idx_2 < out_size_2 &&
            idx_3 >= 0 && idx_3 < out_size_3) {
            int64_t out_idx = idx_0 * out_size_1 * out_size_2 * out_size_3 + 
                             idx_1 * out_size_2 * out_size_3 + 
                             idx_2 * out_size_3 + idx_3;
            atomicAdd(&out[out_idx], values[idx]);
        }
    }
}

// Host functions
void accumulate_2d_cuda(
    torch::Tensor indices_0,
    torch::Tensor indices_1,
    torch::Tensor values,
    torch::Tensor out
) {
    int64_t n_values = values.size(0);
    int64_t out_size_0 = out.size(0);
    int64_t out_size_1 = out.size(1);

    const int threads = 256;
    const int blocks = (n_values + threads - 1) / threads;

    if (values.dtype() == torch::kFloat32) {
        accumulate_2d_kernel_float<<<blocks, threads>>>(
            indices_0.data_ptr<int64_t>(),
            indices_1.data_ptr<int64_t>(),
            values.data_ptr<float>(),
            out.data_ptr<float>(),
            n_values,
            out_size_0,
            out_size_1
        );
    } else if (values.dtype() == torch::kComplexFloat) {
        accumulate_2d_kernel_complex<<<blocks, threads>>>(
            indices_0.data_ptr<int64_t>(),
            indices_1.data_ptr<int64_t>(),
            reinterpret_cast<const cuFloatComplex*>(values.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cuFloatComplex*>(out.data_ptr<c10::complex<float>>()),
            n_values,
            out_size_0,
            out_size_1
        );
    }
}

void accumulate_3d_cuda(
    torch::Tensor indices_0,
    torch::Tensor indices_1,
    torch::Tensor indices_2,
    torch::Tensor values,
    torch::Tensor out
) {
    int64_t n_values = values.size(0);
    int64_t out_size_0 = out.size(0);
    int64_t out_size_1 = out.size(1);
    int64_t out_size_2 = out.size(2);

    const int threads = 256;
    const int blocks = (n_values + threads - 1) / threads;

    if (values.dtype() == torch::kFloat32) {
        accumulate_3d_kernel_float<<<blocks, threads>>>(
            indices_0.data_ptr<int64_t>(),
            indices_1.data_ptr<int64_t>(),
            indices_2.data_ptr<int64_t>(),
            values.data_ptr<float>(),
            out.data_ptr<float>(),
            n_values,
            out_size_0,
            out_size_1,
            out_size_2
        );
    } else if (values.dtype() == torch::kComplexFloat) {
        accumulate_3d_kernel_complex<<<blocks, threads>>>(
            indices_0.data_ptr<int64_t>(),
            indices_1.data_ptr<int64_t>(),
            indices_2.data_ptr<int64_t>(),
            reinterpret_cast<const cuFloatComplex*>(values.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cuFloatComplex*>(out.data_ptr<c10::complex<float>>()),
            n_values,
            out_size_0,
            out_size_1,
            out_size_2
        );
    }
}

void accumulate_4d_cuda(
    torch::Tensor indices_0,
    torch::Tensor indices_1,
    torch::Tensor indices_2,
    torch::Tensor indices_3,
    torch::Tensor values,
    torch::Tensor out
) {
    int64_t n_values = values.size(0);
    int64_t out_size_0 = out.size(0);
    int64_t out_size_1 = out.size(1);
    int64_t out_size_2 = out.size(2);
    int64_t out_size_3 = out.size(3);

    const int threads = 256;
    const int blocks = (n_values + threads - 1) / threads;

    if (values.dtype() == torch::kFloat32) {
        accumulate_4d_kernel_float<<<blocks, threads>>>(
            indices_0.data_ptr<int64_t>(),
            indices_1.data_ptr<int64_t>(),
            indices_2.data_ptr<int64_t>(),
            indices_3.data_ptr<int64_t>(),
            values.data_ptr<float>(),
            out.data_ptr<float>(),
            n_values,
            out_size_0,
            out_size_1,
            out_size_2,
            out_size_3
        );
    } else if (values.dtype() == torch::kComplexFloat) {
        accumulate_4d_kernel_complex<<<blocks, threads>>>(
            indices_0.data_ptr<int64_t>(),
            indices_1.data_ptr<int64_t>(),
            indices_2.data_ptr<int64_t>(),
            indices_3.data_ptr<int64_t>(),
            reinterpret_cast<const cuFloatComplex*>(values.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cuFloatComplex*>(out.data_ptr<c10::complex<float>>()),
            n_values,
            out_size_0,
            out_size_1,
            out_size_2,
            out_size_3
        );
    }
}
'''

cpp_source = '''
#include <torch/extension.h>

void accumulate_2d_cuda(
    torch::Tensor indices_0,
    torch::Tensor indices_1,
    torch::Tensor values,
    torch::Tensor out
);

void accumulate_3d_cuda(
    torch::Tensor indices_0,
    torch::Tensor indices_1,
    torch::Tensor indices_2,
    torch::Tensor values,
    torch::Tensor out
);

void accumulate_4d_cuda(
    torch::Tensor indices_0,
    torch::Tensor indices_1,
    torch::Tensor indices_2,
    torch::Tensor indices_3,
    torch::Tensor values,
    torch::Tensor out
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("accumulate_2d", &accumulate_2d_cuda, "2D index put with atomic add");
    m.def("accumulate_3d", &accumulate_3d_cuda, "3D index put with atomic add");
    m.def("accumulate_4d", &accumulate_4d_cuda, "4D index put with atomic add");
}
'''

# Compile the extension
accumulate_cuda = None

def load_cuda_extension():
    """Load the CUDA extension for custom accumulate operations."""
    global accumulate_cuda
    if accumulate_cuda is None:
        accumulate_cuda = load_inline(
            name='accumulate_cuda',
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            verbose=True
        )
    return accumulate_cuda


def accumulate_cpu(indices, values, out):
    """CPU fallback using PyTorch's index_put_."""
    out.index_put_(indices, values, accumulate=True)


def accumulate(indices, values, out):
    """
    Custom accumulate operation with atomic addition supporting complex64.

    Parameters
    ----------
    indices : tuple of torch.Tensor
        Tuple of 1D tensors containing indices for each dimension.
        Each tensor should have dtype=torch.long and same length as values.
    values : torch.Tensor
        1D tensor of values to be added to output tensor.
        Should have dtype=torch.float32 or torch.complex64.
    out : torch.Tensor
        Output tensor where values will be atomically added.
        Number of dimensions should match len(indices).
        Should have same dtype as values.

    Returns
    -------
    torch.Tensor
        The modified output tensor (in-place operation).

    Examples
    --------
    >>> # 2D case with complex values
    >>> indices = (torch.tensor([0, 1, 0]), torch.tensor([1, 2, 1]))
    >>> values = torch.tensor([1.0+2j, 2.0-1j, 3.0+0j], dtype=torch.complex64)
    >>> out = torch.zeros(3, 4, dtype=torch.complex64)
    >>> accumulate(indices, values, out)
    """
    # Ensure everything is on the same device
    if len({indices[0].device, values.device, out.device}) != 1:
        raise ValueError("Tensors for accumulate must live on same device")

    # Validate inputs
    if not isinstance(indices, tuple):
        raise ValueError("indices must be a tuple of tensors")

    ndim = len(indices)
    if ndim < 2 or ndim > 4:
        raise ValueError("Only 2D, 3D, and 4D operations are supported")

    if out.dim() != ndim:
        raise ValueError(f"Output tensor must have {ndim} dimensions")

    # Check supported dtypes
    supported_dtypes = {torch.float32, torch.complex64}
    if values.dtype not in supported_dtypes:
        raise ValueError(f"Values tensor must have dtype float32 or complex64, "
                         f"got {values.dtype}")

    if out.dtype != values.dtype:
        raise ValueError("Output tensor must have same dtype as values tensor")

    # Validate shapes
    n_values = values.size(0)
    for i, idx in enumerate(indices):
        if idx.size(0) != n_values:
            raise ValueError(f"Index tensor {idx.shape} size mismatch with "
                             f"values")

    if 'cuda' in str(out.device):
        ext = load_cuda_extension()
        # Call appropriate kernel
        if ndim == 2:
            ext.accumulate_2d(indices[0], indices[1], values, out)
        elif ndim == 3:
            ext.accumulate_3d(indices[0], indices[1], indices[2], values, out)
        elif ndim == 4:
            ext.accumulate_4d(indices[0], indices[1], indices[2], indices[3], values, out)
    elif out.device == torch.device('cpu'):
        accumulate_cpu(indices, values, out)
    else:
        raise ValueError("Unsupported device")


# Example usage and testing
if __name__ == "__main__":
    # Test 2D case with complex values
    print("Testing 2D accumulate with complex64...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    indices = (
        torch.tensor([0, 1, 0, 2]).to(device),
        torch.tensor([1, 2, 1, 0]).to(device)
    )
    values = torch.tensor([1.0+2j, 2.0-1j, 3.0+0j, 4.0+1j], dtype=torch.complex64).to(device)
    out = torch.zeros(3, 4, device=device, dtype=torch.complex64)

    # Apply custom accumulate
    accumulate(indices, values, out)
    print("Result:", out)

    # Verify with PyTorch's index_put_
    expected = torch.zeros(3, 4, device=device, dtype=torch.complex64)
    expected.index_put_(indices, values, accumulate=True)
    print("Expected:", expected)
    print("Match:", torch.allclose(out, expected))

    # Test with float32 for backward compatibility
    print("\nTesting 2D accumulate with float32...")
    values_float = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).to(device)
    out_float = torch.zeros(3, 4, device=device, dtype=torch.float32)

    accumulate(indices, values_float, out_float)
    expected_float = torch.zeros(3, 4, device=device, dtype=torch.float32)
    expected_float.index_put_(indices, values_float, accumulate=True)
    print("Float32 match:", torch.allclose(out_float, expected_float))
