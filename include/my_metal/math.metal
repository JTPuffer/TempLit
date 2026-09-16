//
//  Metal_Math.metal
//  Metal_Math
//
//  Created by Matt Oliver on 13/08/2026.
//

#include <metal_stdlib>

using namespace metal;


kernel void matmul_contiguous(
    device const float* buffer_a [[buffer(0)]],
    device const float* buffer_b [[buffer(1)]],
    device float* result         [[buffer(2)]],

    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],

    constant uint* batch_stride_a [[buffer(12)]],
    constant uint* batch_stride_b [[buffer(13)]],
    constant uint* batch_stride_c [[buffer(14)]],
    constant uint* batch_decode_stride [[buffer(15)]],
    constant uint& batch_ndim [[buffer(16)]],

    uint3 tg [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    constexpr uint BM = 64;
    constexpr uint BK = 8;
    constexpr uint BN = 64;
    constexpr uint TM = 8;
    constexpr uint TN = 4;

    threadgroup float tileA[BK][BM];
    threadgroup float tileB[BK][BN];

    float thread_res[TM][TN] = {{0.0f}};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    const uint blockCol = tg.x * BN;
    const uint blockRow = tg.y * BM;

    uint remainder = tg.z;

    uint batch_offset_a = 0;
    uint batch_offset_b = 0;
    uint batch_offset_c = 0;

    for (uint d = 0; d < batch_ndim; ++d) {
        uint coord = remainder / batch_decode_stride[d];
        remainder %= batch_decode_stride[d];

        batch_offset_a += coord * batch_stride_a[d];
        batch_offset_b += coord * batch_stride_b[d];
        batch_offset_c += coord * batch_stride_c[d];
    }

    const uint innerColA4 = tid % (BK / 4);
    const uint innerRowA4 = tid / (BK / 4);

    const uint innerColB4 = tid % (BN / 4);
    const uint innerRowB4 = tid / (BN / 4);

    uint offsetA =
        batch_offset_a
        + blockRow * K
        + innerRowA4 * K
        + innerColA4 * 4;

    uint offsetB =
        batch_offset_b
        + blockCol
        + innerRowB4 * N
        + innerColB4 * 4;

    for (uint blockK = 0; blockK < K; blockK += BK) {

        float4 a4 =
            reinterpret_cast<device const float4*>(
                buffer_a + offsetA
            )[0];

        uint aK = innerColA4 * 4;

        tileA[aK + 0][innerRowA4] = a4.x;
        tileA[aK + 1][innerRowA4] = a4.y;
        tileA[aK + 2][innerRowA4] = a4.z;
        tileA[aK + 3][innerRowA4] = a4.w;

        reinterpret_cast<threadgroup float4*>(
            &tileB[innerRowB4][innerColB4 * 4]
        )[0] =
            reinterpret_cast<device const float4*>(
                buffer_b + offsetB
            )[0];

        offsetA += BK;
        offsetB += BK * N;

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (uint i = 0; i < BK; ++i) {

            for (uint j = 0; j < TM; ++j) {
                regM[j] =
                    tileA[i][local.y * TM + j];
            }

            for (uint j = 0; j < TN; ++j) {
                regN[j] =
                    tileB[i][local.x * TN + j];
            }

            for (uint m = 0; m < TM; ++m) {
                for (uint n = 0; n < TN; ++n) {
                    thread_res[m][n] +=
                        regM[m] * regN[n];
                }
            }
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    const uint baseRow =
        blockRow + local.y * TM;

    const uint baseCol =
        blockCol + local.x * TN;

    for (uint m = 0; m < TM; ++m) {
        for (uint n = 0; n < TN; ++n) {
            result[
                batch_offset_c
                + (baseRow + m) * N
                + baseCol + n
            ] = thread_res[m][n];
        }
    }
}

kernel void matmul_strided(
    device const float* buffer_a [[buffer(0)]],
    device const float* buffer_b [[buffer(1)]],
    device float* result         [[buffer(2)]],

    constant uint& M             [[buffer(3)]],
    constant uint& K             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],

    constant uint& stride_a0     [[buffer(6)]],
    constant uint& stride_a1     [[buffer(7)]],
    constant uint& stride_b0     [[buffer(8)]],
    constant uint& stride_b1     [[buffer(9)]],
    constant uint& stride_c0     [[buffer(10)]],
    constant uint& stride_c1     [[buffer(11)]],

    constant uint* batch_stride_a [[buffer(12)]],
    constant uint* batch_stride_b [[buffer(13)]],
    constant uint* batch_stride_c [[buffer(14)]],
    constant uint* batch_decode_stride [[buffer(15)]],
    constant uint& batch_ndim     [[buffer(16)]],

    uint3 tg    [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]],
    uint tid    [[thread_index_in_threadgroup]]
)
{
    constexpr uint BM = 64;
    constexpr uint BK = 8;
    constexpr uint BN = 64;

    constexpr uint TM = 8;
    constexpr uint TN = 4;

    threadgroup float tileA[BK][BM];
    threadgroup float tileB[BK][BN];

    float thread_res[TM][TN] = {{0.0f}};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    const uint block_col = tg.x * BN;
    const uint block_row = tg.y * BM;

    uint remainder = tg.z;

    uint batch_offset_a = 0;
    uint batch_offset_b = 0;
    uint batch_offset_c = 0;

    for (uint d = 0; d < batch_ndim; ++d) {
        uint coord = remainder / batch_decode_stride[d];
        remainder %= batch_decode_stride[d];

        batch_offset_a += coord * batch_stride_a[d];
        batch_offset_b += coord * batch_stride_b[d];
        batch_offset_c += coord * batch_stride_c[d];
    }

    /*
        128 threads for the default configuration:

            threads_x = BN / TN = 16
            threads_y = BM / TM = 8

            16 * 8 = 128
    */

    const uint innerColA4 = tid % (BK / 4);
    const uint innerRowA4 = tid / (BK / 4);

    const uint innerColB4 = tid % (BN / 4);
    const uint innerRowB4 = tid / (BN / 4);

    for (uint blockK = 0; blockK < K; blockK += BK) {


        const uint a_row =
            block_row + innerRowA4;

        const uint a_col =
            blockK + innerColA4 * 4;

        const uint aK =
            innerColA4 * 4;

        tileA[aK + 0][innerRowA4] =
            buffer_a[
                batch_offset_a
                + a_row * stride_a0
                + (a_col + 0) * stride_a1
            ];

        tileA[aK + 1][innerRowA4] =
            buffer_a[
                batch_offset_a
                + a_row * stride_a0
                + (a_col + 1) * stride_a1
            ];

        tileA[aK + 2][innerRowA4] =
            buffer_a[
                batch_offset_a
                + a_row * stride_a0
                + (a_col + 2) * stride_a1
            ];

        tileA[aK + 3][innerRowA4] =
            buffer_a[
                batch_offset_a
                + a_row * stride_a0
                + (a_col + 3) * stride_a1
            ];



        const uint b_row =
            blockK + innerRowB4;

        const uint b_col =
            block_col + innerColB4 * 4;

        tileB[innerRowB4][innerColB4 * 4 + 0] =
            buffer_b[
                batch_offset_b
                + b_row * stride_b0
                + (b_col + 0) * stride_b1
            ];

        tileB[innerRowB4][innerColB4 * 4 + 1] =
            buffer_b[
                batch_offset_b
                + b_row * stride_b0
                + (b_col + 1) * stride_b1
            ];

        tileB[innerRowB4][innerColB4 * 4 + 2] =
            buffer_b[
                batch_offset_b
                + b_row * stride_b0
                + (b_col + 2) * stride_b1
            ];

        tileB[innerRowB4][innerColB4 * 4 + 3] =
            buffer_b[
                batch_offset_b
                + b_row * stride_b0
                + (b_col + 3) * stride_b1
            ];

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

for (uint i = 0; i < BK; ++i) {

            for (uint j = 0; j < TM; ++j) {
                regM[j] =
                    tileA[i][
                        local.y * TM + j
                    ];
            }

            for (uint j = 0; j < TN; ++j) {
                regN[j] =
                    tileB[i][
                        local.x * TN + j
                    ];
            }

            for (uint m = 0; m < TM; ++m) {
                for (uint n = 0; n < TN; ++n) {
                    thread_res[m][n] +=
                        regM[m] * regN[n];
                }
            }
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }


    const uint base_row =
        block_row + local.y * TM;

    const uint base_col =
        block_col + local.x * TN;

    for (uint m = 0; m < TM; ++m) {
        for (uint n = 0; n < TN; ++n) {
            result[
                batch_offset_c
                + (base_row + m) * stride_c0
                + (base_col + n) * stride_c1
            ] = thread_res[m][n];
        }
    }
}

kernel void matmul_generic(
    device const float* buffer_a [[buffer(0)]],
    device const float* buffer_b [[buffer(1)]],
    device float* result         [[buffer(2)]],

    constant uint& M             [[buffer(3)]],
    constant uint& K             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],

    constant uint& stride_a0     [[buffer(6)]],
    constant uint& stride_a1     [[buffer(7)]],
    constant uint& stride_b0     [[buffer(8)]],
    constant uint& stride_b1     [[buffer(9)]],
    constant uint& stride_c0     [[buffer(10)]],
    constant uint& stride_c1     [[buffer(11)]],

    constant uint* batch_stride_a [[buffer(12)]],
    constant uint* batch_stride_b [[buffer(13)]],
    constant uint* batch_stride_c [[buffer(14)]],
    constant uint* batch_decode_stride [[buffer(15)]],
    constant uint& batch_ndim     [[buffer(16)]],

    uint3 index [[thread_position_in_grid]]
){
    uint col = index.x;
    uint row = index.y;
    uint remainder = index.z;

    uint batch_offset_a = 0;
    uint batch_offset_b = 0;
    uint batch_offset_c = 0;

    for (uint d = 0; d < batch_ndim; ++d) {
        uint coord = remainder / batch_decode_stride[d];
        remainder %= batch_decode_stride[d];
        batch_offset_a += coord * batch_stride_a[d];
        batch_offset_b += coord * batch_stride_b[d];
        batch_offset_c += coord * batch_stride_c[d];
    }

    float sum = 0.0f;
    for (uint j = 0; j < K; ++j) {
        sum += buffer_a[batch_offset_a + row * stride_a0 + j * stride_a1]
             * buffer_b[batch_offset_b + j * stride_b0 + col * stride_b1];
    }

    result[batch_offset_c + row * stride_c0 + col * stride_c1] = sum;
}
kernel void add(
    device const float* buffer_a [[buffer(0)]],
    device const float* buffer_b [[buffer(1)]],
    device float* result         [[buffer(2)]],

    constant uint* stride_a     [[buffer(3)]],
    constant uint* stride_b     [[buffer(4)]],
    constant uint* stride_c     [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    uint index [[thread_position_in_grid]]
){
    uint remainder = index;
    
    uint offsetA = 0;
    uint offsetB = 0;
    
    
    for (uint d = 0; d < ndim; ++d) {
        uint cord = remainder / stride_c[d];
        remainder = remainder % stride_c[d];
        offsetA+= cord * stride_a[d];
        offsetB+= cord * stride_b[d];
    }
    result[index] = buffer_a[offsetA] + buffer_b[offsetB];

}
kernel void add_scalar(
    device const float* buffer_a [[buffer(0)]],
    constant float& scalar       [[buffer(1)]],
    device float* result         [[buffer(2)]],
    uint index [[thread_position_in_grid]]
){
    result[index] = buffer_a[index] + scalar;
}
kernel void sub(
    device const float* buffer_a [[buffer(0)]],
    device const float* buffer_b [[buffer(1)]],
    device float* result         [[buffer(2)]],

    constant uint* stride_a     [[buffer(3)]],
    constant uint* stride_b     [[buffer(4)]],
    constant uint* stride_c     [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    uint index [[thread_position_in_grid]]
){
    uint remainder = index;
    
    uint offsetA = 0;
    uint offsetB = 0;
    
    
    for (uint d = 0; d < ndim; ++d) {
        uint cord = remainder / stride_c[d];
        remainder = remainder % stride_c[d];
        offsetA+= cord * stride_a[d];
        offsetB+= cord * stride_b[d];
    }
    result[index] = buffer_a[offsetA] - buffer_b[offsetB];

}
kernel void hadamard(
    device const float* buffer_a [[buffer(0)]],
    device const float* buffer_b [[buffer(1)]],
    device float* result         [[buffer(2)]],

    constant uint* stride_a     [[buffer(3)]],
    constant uint* stride_b     [[buffer(4)]],
    constant uint* stride_c     [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    uint index [[thread_position_in_grid]]
){
    uint remainder = index;
    
    uint offsetA = 0;
    uint offsetB = 0;
    
    
    for (uint d = 0; d < ndim; ++d) {
        uint cord = remainder / stride_c[d];
        remainder = remainder % stride_c[d];
        offsetA+= cord * stride_a[d];
        offsetB+= cord * stride_b[d];
    }
    result[index] = buffer_a[offsetA] * buffer_b[offsetB];
}
kernel void divide(
    device const float* buffer_a [[buffer(0)]],
    device const float* buffer_b [[buffer(1)]],
    device float* result         [[buffer(2)]],

    constant uint* stride_a     [[buffer(3)]],
    constant uint* stride_b     [[buffer(4)]],
    constant uint* stride_c     [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    uint index [[thread_position_in_grid]]
){
    uint remainder = index;
    
    uint offsetA = 0;
    uint offsetB = 0;
    
    
    for (uint d = 0; d < ndim; ++d) {
        uint cord = remainder / stride_c[d];
        remainder = remainder % stride_c[d];
        offsetA+= cord * stride_a[d];
        offsetB+= cord * stride_b[d];
    }
    result[index] = buffer_a[offsetA] / buffer_b[offsetB];
}
kernel void divide_scalar(
    device const float* buffer_a [[buffer(0)]],
    constant float& scalar       [[buffer(1)]],
    device float* result         [[buffer(2)]],
    uint index [[thread_position_in_grid]]
){
    result[index] = buffer_a[index] / scalar;
}
kernel void sqrt(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    uint index [[thread_position_in_grid]]

) {
    result[index] = sqrt(input[index]);
}
kernel void pow_scalar(
    device const float* input [[buffer(0)]],
    constant float& exponent  [[buffer(1)]],
    device float* result      [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = pow(input[index], exponent);
}
kernel void exp(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = exp(input[index]);
}
kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = 1.0f / (1.0f + exp(-input[index]));
}
kernel void sum(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index == 0) {
        float total = 0.0f;

        for (uint i = 0; i < count; ++i) {
            total += input[i];
        }

        result[0] = total;
    }
}

kernel void transpose(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    constant uint* stride_input [[buffer(2)]],
    constant uint* stride_output [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint remainder = index;
    uint input_offset = 0;

    for (uint d = 0; d < ndim; ++d) {
        uint coord = remainder / stride_output[d];
        remainder = remainder % stride_output[d];
        input_offset += coord * stride_input[d];
    }

    result[index] = input[input_offset];
}

kernel void sum_to_shape_1d(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    constant uint* stride_input     [[buffer(2)]],
    constant uint* stride_output     [[buffer(3)]],

    constant uint& ndim [[buffer(4)]],
    constant uint& shrink_dim [[buffer(5)]],
    constant uint& shrink_size [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    // garunteed input and output to have same dimensions
    uint remainder = index;

    uint offsetA = 0;
    //requires 1 dimension to bassically be 1
    // that is so shit and anoying
    for (uint d = 0; d < ndim; ++d) {
        // skip this dim because we wana be at the start
        uint cord = remainder / stride_output[d];
        remainder = remainder % stride_output[d];
        offsetA+= cord * stride_input[d];
    }

    float sum = 0.0f;

    for (uint i = 0; i < shrink_size; ++i) {
        sum += input[offsetA + i * stride_input[shrink_dim]];
    }
    result[index] = sum;
}

kernel void sum_to_shape(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],

    constant uint* stride_input [[buffer(2)]],
    constant uint* stride_output [[buffer(3)]],

    constant uint* reduce_dims [[buffer(4)]],
    constant uint* reduce_sizes [[buffer(5)]],

    constant uint& ndim [[buffer(6)]],
    constant uint& num_reduce_dims [[buffer(7)]],
    constant uint& reduce_count [[buffer(8)]],

    uint index [[thread_position_in_grid]]
) {
    // garunteed input and output to have same dimensions
    uint remainder = index;

    uint offsetA = 0;
    //requires 1 dimension to bassically be 1
    // that is so shit and anoying
    for (uint d = 0; d < ndim; ++d) {
        // skip this dim because we wana be at the start
        uint cord = remainder / stride_output[d];
        remainder = remainder % stride_output[d];
        offsetA+= cord * stride_input[d];
    }

    float sum = 0.0f;


    for (uint i = 0; i < reduce_count; ++i) {
        uint rem = i;
        uint reduce_offset = 0;
        // say we have
        // input  = (2, 3, 4)
        // output = (1, 3, 1)
        // so need to sum dim 0 and dim 4 together
        // need to get to
        // 0,0
        // 0,1
        //0,2
        //0,3
        //1,0
        // bassicaly a for loop for how many dimensions that you have
        //ok so reduced count represents what index you are on now you just need to decode it into arbiratery i j x y etc
        for (int j = int(num_reduce_dims) - 1; j >= 0; --j) {
            // for the 1st dimension where are we at then
            //var[i][j] == var[i * size + j]
            uint size = reduce_sizes[j];
            uint cord = rem % size;
            rem /=  size;
            //reverse iteration so it does the smallest number 1st whihc is hte one we wana incriment when accessing mem
            // so if we are at say index 7 we want 1,3 idealy
            reduce_offset += cord * stride_input[reduce_dims[j]];
        }
        sum += input[offsetA + reduce_offset];
    }

    result[index] = sum;
}

kernel void max_to_shape(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    constant uint* stride_input     [[buffer(2)]],
    constant uint* stride_output     [[buffer(3)]],

    constant uint& ndim [[buffer(4)]],
    constant uint& shrink_dim [[buffer(5)]],
    constant uint& shrink_size [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    // garunteed input and output to have same dimensions
    uint remainder = index;

    uint offsetA = 0;
    //requires 1 dimension to bassically be 1
    // that is so shit and anoying
    for (uint d = 0; d < ndim; ++d) {
        // skip this dim because we wana be at the start
        uint cord = remainder / stride_output[d];
        remainder = remainder % stride_output[d];
        offsetA+= cord * stride_input[d];
    }

    float best =  -INFINITY;

    for (uint i = 0; i < shrink_size; ++i) {
        float v = input[offsetA + i * stride_input[shrink_dim]];
        if(v> best){
            best = v;
        }
    }
    result[index] = best;
}
kernel void argmax(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    device uint* indices      [[buffer(2)]],
    constant uint& count      [[buffer(3)]],

    uint gid [[thread_position_in_grid]]
) {
    constexpr uint ITEMS = 16;

    float best = -INFINITY;
    uint index = 0;

    uint base = gid * ITEMS;

    for (uint j = 0; j < ITEMS; ++j) {
        uint idx = base + j;

        if (idx < count) {
            float v = input[idx];

            if (v > best) {
                best = v;
                index = idx;
            }
        }
    }

    result[gid] = best;
    indices[gid] = index;
}

kernel void argmax_reduce(
    device const float* input_values [[buffer(0)]],
    device const uint* input_indices [[buffer(1)]],
    device float* result_values      [[buffer(2)]],
    device uint* result_indices      [[buffer(3)]],
    constant uint& count             [[buffer(4)]],

    uint gid [[thread_position_in_grid]]
) {
    constexpr uint ITEMS = 16;

    float best = -INFINITY;
    uint best_index = 0;
    uint base = gid * ITEMS;

    for (uint j = 0; j < ITEMS; ++j) {
        uint idx = base + j;

        if (idx < count) {
            float value = input_values[idx];

            if (value > best) {
                best = value;
                best_index = input_indices[idx];
            }
        }
    }

    result_values[gid] = best;
    result_indices[gid] = best_index;
}
kernel void embedding_lookup(
    device const float* weights [[buffer(0)]],
    device const uint* ids      [[buffer(1)]],
    device float* output        [[buffer(2)]],

    constant uint& embedding_size [[buffer(3)]],

    uint index [[thread_position_in_grid]]
                             ) {

    uint id_index = index /embedding_size;

    uint offset = index % embedding_size;

    output[index] = weights[ids[id_index] * embedding_size + offset];
}
kernel void scatter_add_rows(
                        device const float* grad_output [[buffer(0)]],
                        device const uint* ids          [[buffer(1)]],
                        device atomic_float* grad_weights [[buffer(2)]],

    constant uint& embedding_size [[buffer(3)]],

    uint index [[thread_position_in_grid]]
                             ) {

    uint id_index = index /embedding_size;

    uint offset = index % embedding_size;
    uint token_id = ids[id_index];

    uint dst =
        token_id * embedding_size + offset;

    float grad =
        grad_output[id_index * embedding_size + offset];

    atomic_fetch_add_explicit(
        &grad_weights[dst],
        grad,
        memory_order_relaxed
    );
}
kernel void GELU(
    device const float* input  [[buffer(0)]],
    device float* result       [[buffer(1)]],
    uint index                 [[thread_position_in_grid]]
) {
    float x = input[index];

    result[index] =
        0.5f * x *
        (1.0f + tanh(
            sqrt(2.0f / 3.14159265f) *
            (x + 0.044715f * x * x * x)
        ));
}
kernel void GELU_derivative(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    uint index                [[thread_position_in_grid]]
) {
// i cant lie this is so long no way am i implimenting this in autograd
    float x = input[index];

    constexpr float a = 0.7978845608f; // sqrt(2 / pi)
    constexpr float b = 0.044715f;

    float x2 = x * x;
    float u = a * (x + b * x2 * x);

    float t = tanh(u);

    result[index] =
        0.5f * (1.0f + t)
        + 0.5f * x * (1.0f - t * t)
        * a * (1.0f + 3.0f * b * x2);
}

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* result      [[buffer(1)]],
    constant uint& last_dim [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint offset = index * last_dim;
    float best = -INFINITY;
// get max value
    for (uint i = 0; i < last_dim; ++i) {
        float v = input[offset + i];
        if (v > best) {
            best = v;
        }
    }
// calculate exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < last_dim; ++i) {
        result[offset + i] = exp(input[offset + i] - best);
        sum += result[offset + i];
    }
// normaisle
    for (uint i = 0; i < last_dim; ++i) {
        result[offset + i] /= sum;
    }
}

kernel void cross_entropy_rows(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* losses       [[buffer(2)]],
    constant uint& vocab_size  [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    uint offset = row * vocab_size;
    uint target = targets[row];
    if (target >= vocab_size) {
        losses[row] = INFINITY;
        return;
    }

    float maximum = -INFINITY;
    for (uint i = 0; i < vocab_size; ++i) {
        maximum = max(maximum, logits[offset + i]);
    }

    float sum_exp = 0.0f;
    for (uint i = 0; i < vocab_size; ++i) {
        sum_exp += exp(logits[offset + i] - maximum);
    }

    losses[row] =
        maximum + log(sum_exp) - logits[offset + target];
}

kernel void cross_entropy_rows_backward(
    device const float* logits      [[buffer(0)]],
    device const uint* targets      [[buffer(1)]],
    device float* grad_logits       [[buffer(2)]],
    constant uint& vocab_size       [[buffer(3)]],
    uint row                        [[thread_position_in_grid]]
) {
    uint offset = row * vocab_size;
    uint target = targets[row];
    if (target >= vocab_size) {
        for (uint i = 0; i < vocab_size; ++i) {
            grad_logits[offset + i] = INFINITY;
        }
        return;
    }

    // Find max
    float maximum = -INFINITY;

    for (uint i = 0; i < vocab_size; ++i) {
        maximum = max(maximum, logits[offset + i]);
    }

    // Sum exp
    float sum_exp = 0.0f;

    for (uint i = 0; i < vocab_size; ++i) {
        sum_exp += exp(logits[offset + i] - maximum);
    }


    // dL/dz = softmax(z) - y
    for (uint i = 0; i < vocab_size; ++i) {

        float p =
            exp(logits[offset + i] - maximum) / sum_exp;

        grad_logits[offset + i] = p;
    }
// this is actualy cool
    grad_logits[offset + target] -= 1.0f;
}

kernel void accumulate_gradient_squared_norm(
    device const float* gradient [[buffer(0)]],
    device atomic_float* squared_norm [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]]
) {
    constexpr uint threadgroup_size = 256;
    threadgroup float partial[threadgroup_size];

    float value = index < count ? gradient[index] : 0.0f;
    partial[local_index] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_size / 2; stride > 0; stride /= 2) {
        if (local_index < stride) {
            partial[local_index] += partial[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_index == 0) {
        atomic_fetch_add_explicit(
            squared_norm,
            partial[0],
            memory_order_relaxed
        );
    }
}

kernel void clip_gradient(
    device const float* gradient [[buffer(0)]],
    device const atomic_float* squared_norm [[buffer(1)]],
    constant float& max_norm [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    float norm_squared = atomic_load_explicit(
        squared_norm,
        memory_order_relaxed
    );
    float scale = min(
        1.0f,
        max_norm * rsqrt(max(norm_squared, 1e-12f))
    );
    destination[index] = gradient[index] * scale;
}

kernel void adamW(
    device const float* parameter [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    device float* destination [[buffer(2)]],

    device float* mom_1 [[buffer(3)]],
    device float* mom_2 [[buffer(4)]],
    device const float& beta1[[buffer(5)]],
    device const float& beta2[[buffer(6)]],
    device const float& decay[[buffer(7)]],
    device const float& learning_rate[[buffer(8)]],
    device const float& epsilon[[buffer(9)]],
    device const uint& step[[buffer(10)]],


    uint index [[thread_position_in_grid]]
                   ){

    // m = beta1 * m + (1 - beta1) * gradient
    mom_1[index] =
        beta1 * mom_1[index] +
        (1.0 - beta1) * gradient[index];

    mom_2[index] =
        beta2 * mom_2[index] +
        (1.0 - beta2) * gradient[index] * gradient[index];

    destination[index] =
        parameter[index] * decay -
        learning_rate *
            (mom_1[index] / (1.0 - pow(beta1, step))) /
            (sqrt(
                mom_2[index] / (1.0 - pow(beta2, step))
            ) + epsilon);

}

kernel void copy(
    device const float* buffer_a [[buffer(0)]],
    device float* result         [[buffer(1)]],
    uint index [[thread_position_in_grid]]
){
    result[index] = buffer_a[index];
}
