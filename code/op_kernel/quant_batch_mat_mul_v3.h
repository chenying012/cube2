/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 算子 kernel 类定义
 */

#ifndef QUANTBATCHMATMULV3_H
#define QUANTBATCHMATMULV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_batch_mat_mul_v3_tiling_data.h"
#include "quant_batch_mat_mul_v3_tiling_key.h"

namespace NsQuantBatchMatMulV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class QuantBatchMatMulV3 {
public:
    __aicore__ inline QuantBatchMatMulV3(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, const QuantBatchMatMulV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, const QuantBatchMatMulV3TilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Process()
{
    // TODO: 实现 Process 逻辑
}

} // namespace NsQuantBatchMatMulV3
#endif // QUANTBATCHMATMULV3_H
