# Weight Precision in Lc0 Backends

This document explains the precision at which network weights (including those for Relative Positional Encoding - RPE) are stored, loaded, and used by various Lc0 backends.

## Weight Storage in `.pb` Files

Network weight files (`.pb` or `.pb.gz`) in Lc0 store weights in a **16-bit quantized format**, referred to as `LINEAR16` in the codebase. This means:

- Each weight value is stored as a 16-bit unsigned integer.
- Each layer's weights also include a `min` floating-point value and a `range` floating-point value.
- These `min` and `range` values are used to dequantize the 16-bit integers back into 32-bit floating-point numbers (`float`).

The dequantization formula used is approximately:
`float_value = min + (uint16_value * range) / 65535.0`

This process is handled by the `LayerAdapter` class when weights are initially loaded from the protobuf file. All internal representations of weights in structures like `MultiHeadWeights` or `LegacyWeights` (e.g., convolutional weights, fully connected layer weights, batch normalization parameters, and RPE weights) are therefore stored as 32-bit `float` values after this initial dequantization step.

## Backend-Specific Precision Handling

Once the weights are loaded and dequantized into 32-bit floats, each backend handles them as follows:

### 1. BLAS Backend

- The BLAS backend receives all weights (including RPE) as 32-bit floats.
- All computations within the BLAS backend are performed using **32-bit floating-point (fp32)** precision.

### 2. OpenCL Backend

- Similar to the BLAS backend, the OpenCL backend receives all weights (including RPE) as 32-bit floats.
- All computations within the OpenCL backend are performed using **32-bit floating-point (fp32)** precision.

### 3. CUDA Backend

The CUDA backend's behavior depends on its configuration:

- **fp32 Mode (e.g., `cudnn` backend option):**
    - It receives all weights (including RPE) as 32-bit floats.
    - Computations are performed using **32-bit floating-point (fp32)** precision.
- **fp16 Mode (e.g., `cudnn-fp16` backend option):**
    - It receives all weights (including RPE) as 32-bit floats from the initial load.
    - These 32-bit float weights are then **converted to 16-bit half-precision floating-point (`half`)** format.
    - All computations are subsequently performed using **16-bit half-precision (fp16)**.
    - The conversion path is: 16-bit quantized (file) -> 32-bit float (initial load) -> 16-bit half (for fp16 computation).

### 4. ONNX Backend

The ONNX backend involves an intermediate step of converting Lc0's native weight format to an ONNX model:

-   Weights (including RPE) are first loaded and dequantized to 32-bit floats as described in the "Weight Storage in `.pb` Files" section.
-   These 32-bit float weights are then used by the `ConvertWeightsToOnnx` utility to create an ONNX model. The ONNX model itself can be configured to store its weights in **fp32, fp16, or bf16 (BFloat16)** precision, depending on the options provided during conversion (e.g., the `datatype` backend option or older `--fp16` command-line flags).
-   The ONNX Runtime then executes this ONNX model. Generally, the precision of computation will match the precision of the weights within the generated ONNX model, especially when using GPU execution providers (like CUDA or DirectML) that have robust support for lower precision.

**Note on ONNX Runtime CPU Execution Provider:**

A specific consideration arises when using ONNX Runtime with its CPU Execution Provider (CPU EP). While the ONNX model file may have its weights stored at a lower precision like fp16:

-   The ONNX Runtime's CPU EP might, in some circumstances, internally execute certain operations or parts of the graph at a higher precision (typically fp32). This can occur for various reasons, including:
    -   Lack of a native, stable, or performant fp16/bf16 kernel for a specific operator or a particular configuration of an operator on the CPU.
    -   To ensure numerical stability or correctness for certain mathematical functions when implemented on general-purpose CPUs.
    -   Operations involving constants (e.g., some matrix multiplications where one input is a constant weight tensor) might be handled by CPU kernels that prefer fp32.
-   This behavior is internal to ONNX Runtime and depends on its version and the specifics of its CPU EP implementation. It means that even if your `.onnx` model's weights are, for example, fp16, some computations during CPU inference might still effectively happen in fp32.

Lc0's ONNX converter also has some internal logic for specific cases. For instance, when converting to BFloat16 with an ONNX opset older than 22, Lc0 might wrap certain operations (like `Conv`) with casts to fp32 and back to BFloat16. This is a workaround within Lc0's converter for limited BFloat16 operator support in those older opsets and is distinct from the general behavior of ONNX Runtime's CPU EP.

Therefore, the path for ONNX is: 16-bit quantized (file) -> 32-bit float (initial Lc0 load) -> fp32/fp16/bf16 (weights in the `.onnx` model file) -> computation by ONNX Runtime, where the actual operational precision for some parts on CPU might default to fp32 irrespective of the model's weight precision.

## Relative Positional Encoding (RPE) Precision

Relative Positional Encoding (RPE) weights are integral to attention mechanisms in some network architectures. Their precision handling follows the same general pipeline as other weights, with specific steps in each backend:

1.  **Storage and Initial Load:**
    *   RPE weights (e.g., `rpe_q`, `rpe_k`, `rpe_v` in the `MHA` struct) are stored as 16-bit quantized values in the `.pb` file, just like other weights.
    *   Upon loading, `LayerAdapter` dequantizes them into 32-bit `float` values. Thus, in CPU-side structures like `MultiHeadWeights`, RPE weights are represented as `std::vector<float>`.

2.  **Backend-Specific Handling of fp32 RPE Weights:**

    *   **BLAS Backend:** Receives the 32-bit `float` RPE weights and uses them directly in fp32 computations.
    *   **OpenCL Backend:** Receives the 32-bit `float` RPE weights and uses them directly in fp32 computations.
    *   **CUDA Backend (fp32 Mode):** Receives the 32-bit `float` RPE weights and uses them directly in fp32 computations.
    *   **CUDA Backend (fp16 Mode):**
        *   Receives the 32-bit `float` RPE weights from the initial load.
        *   These fp32 RPE weights are then explicitly **converted to 16-bit half-precision (`half`)** when being transferred to GPU memory for use by fp16-specialized layers (e.g., within the `EncoderBlock` class, which is templated on `DataType`).
        *   The CUDA kernels responsible for attention calculations involving RPE (e.g., `multiplyRPEAttentionLogits`) are also templated by `DataType`. When `DataType` is `half` (for fp16 mode), these kernels operate on the 16-bit half-precision RPE weights and perform computations in fp16.
    *   **ONNX Backend:**
        *   The 32-bit `float` RPE weights are included when converting the network to an ONNX model.
        *   The ONNX model's RPE weights will have the precision specified during the conversion (fp32, fp16, or bf16).
        *   ONNX Runtime then performs computations using RPE weights at that ONNX model precision.

In essence, RPE weights are not an exception to the general precision handling rules. They are initially brought to fp32 and then either used directly or converted by the specific backend (CUDA fp16, ONNX) to the target computational precision. There are no RPE-specific matmuls that unilaterally remain in fp32 if the rest of the CUDA fp16 backend is operating in half-precision.

### Code Context: RPE in CUDA fp16 Mode

To provide further clarity on RPE handling in the CUDA fp16 backend:

-   **Initial CPU-side Representation:** RPE weights (e.g., `rpe_q`, `rpe_k`, `rpe_v`) are stored within the `MultiHeadWeights::MHA` structure as `std::vector<float>`. This is after dequantization from the file by `LayerAdapter`.
-   **Transfer to GPU and Conversion:** When a CUDA backend network is initialized in fp16 mode, classes like `cudnn_backend::EncoderBlock` (which are templated, `EncoderBlock<DataType>`, where `DataType` becomes `half`) take these `float` vectors. During the `EncoderBlock`'s construction, it allocates GPU memory for RPE weights (e.g., `mha_rpe_q_`) as `DataType*` (i.e., `half*`). The `float` CPU weights are then copied and converted to `half` on the GPU using a kernel like `copyTypeConvertedKernel`.
    ```cpp
    // Simplified conceptual snippet from cudnn_backend::EncoderBlock constructor
    // (actual code involves more checks and specific weight names)
    // cpu_weights.mha.rpe_q is std::vector<float>
    // mha_rpe_q_ is half* (DataType* when DataType is half)
    // ReportCUDAErrors(cudaMalloc(&mha_rpe_q_, cpu_weights.mha.rpe_q.size() * sizeof(DataType)));
    // copyTypeConvertedKernel(mha_rpe_q_, cpu_weights.mha.rpe_q.data(), cpu_weights.mha.rpe_q.size(), stream);
    ```
-   **GPU Computation:** CUDA kernels that perform attention calculations involving RPE, such as `multiplyRPEAttentionLogits` or `multiplyRpeQKLogits` (found in `src/neural/cuda/kernels.h`), are also templated with `DataType`. When the network is in fp16 mode, these kernels are instantiated with `half` and thus operate entirely using 16-bit half-precision for both the RPE weights and the activations they interact with.

This explicit conversion and templated kernel design ensure that RPE-related computations align with the overall fp16 precision of the CUDA backend when so configured.
