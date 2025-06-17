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

- Weights (including RPE) are first loaded and dequantized to 32-bit floats as described above.
- These 32-bit float weights are then used by the `ConvertWeightsToOnnx` utility to create an ONNX model.
- The ONNX model itself can be configured to store its weights in **fp32, fp16, or bf16 (BFloat16)** precision, depending on the options provided during conversion (e.g., `--fp16` flag for the `lc0` command or backend options).
- The ONNX Runtime then executes this ONNX model. The precision of computation will match the precision of the weights within the generated ONNX model.
- The conversion path is: 16-bit quantized (file) -> 32-bit float (initial load) -> fp32/fp16/bf16 (weights in the ONNX model) -> computation by ONNX Runtime at that model precision.

## Relative Positional Encoding (RPE) Precision

As mentioned, RPE weights are treated the same way as all other network weights:

1.  Stored as 16-bit quantized values in the `.pb` file.
2.  Dequantized to 32-bit `float` upon loading (e.g., within the `MHA` struct in `MultiHeadWeights`).
3.  Handled by each backend according to the rules above:
    *   **BLAS/OpenCL/CUDA (fp32 mode):** Used as 32-bit floats.
    *   **CUDA (fp16 mode):** Converted from 32-bit float to 16-bit half for fp16 computation.
    *   **ONNX:** Converted from 32-bit float to the target ONNX model's weight precision (fp32, fp16, or bf16) and used accordingly by ONNX Runtime.

In summary, while the on-disk storage is 16-bit quantized, the primary in-memory representation after loading is 32-bit float. Backends then either use this fp32 precision directly or convert the weights further to fp16/bf16 for specific optimized computation paths.
