/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scheduler_norm_mx_only_scaleb.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/scheduler/base/scheduler_norm_mx_only_scaleb.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_MX_ONLY_SCALEB_H
#endif
#ifndef IMPL_MATMUL_MODULES_SCHEDULER_SCHEDULER_NORM_MX_ONLY_SCALEB_H
#define IMPL_MATMUL_MODULES_SCHEDULER_SCHEDULER_NORM_MX_ONLY_SCALEB_H

#include "scheduler_intf.h"
#include "scheduler_norm_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <
    typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    PolicyType POLICY_TYPE>
class MatmulScheduler<
    IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE,
    enable_if_t<
        isNormEnableScheduler<A_TYPE, MM_CFG> && !MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB() &&
        isOnlyScaleBMxMatmul<A_TYPE, B_TYPE>>>
    : public MatmulNormSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE> {
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(CopyCubeInScaleB);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MmadCompute);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(MatmulUnitFlag);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using SrcScaleAT = uint8_t;
    using SrcScaleBT = typename B_TYPE::SCALE_T;
    using SrcA2T = typename GetL0DataType<typename A_TYPE::T, true>::Type;
    using SrcB2T = typename GetL0DataType<typename B_TYPE::T, true>::Type;
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulNormSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE>;

    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe)
    {
        BASE_MODULE::Init(cubeTiling, tpipe);
        MATMUL_MODULE(CopyCubeInScaleB)->Init();
        InitShareBufEnd(tpipe);
    }

    __aicore__ inline void Compute(bool partialSumEnabled = false)
    {
        if constexpr (!IsBasic(MM_CFG)) {
            // K outer loop only circulates multi-times
            ComputeMultiIter(partialSumEnabled);
        } else {
            // K outer loop only circulates once
            ComputeOneIter(partialSumEnabled);
        }
    }

    __aicore__ inline bool ScheduleOnce(bool partialSumEnabled)
    {
        MATMUL_MODULE(BiasScheduler)->SetBias(MATMUL_MODULE(BiasScheduler)->IsBias() && !partialSumEnabled);
        if (!MoveNext()) {
            return false;
        }
        if (!partialSumEnabled) {
            MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
        }
        Compute(partialSumEnabled);
        return true;
    }

protected:
    __aicore__ inline bool MoveOnFirstIterate()
    {
        // start MLoop and NLoop on the first iteration
        MATMUL_MODULE(MLoop)->OuterStart();
        MATMUL_MODULE(MLoop)->InnerStart();
        MATMUL_MODULE(NLoop)->OuterStart();
        MATMUL_MODULE(NLoop)->InnerStart();
        BASE_MODULE::isFirstIter_ = false;
        return true;
    }

    __aicore__ inline bool MoveNext()
    {
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreM != 0 && GetMIter(MM_CFG) == 1 && GetNIter(MM_CFG) == 1) {
            // only iterate once
            if (unlikely(BASE_MODULE::isFirstIter_)) {
                return MoveOnFirstIterate();
            } else {
                return false;
            }
        } else {
            return MoveNextMulti();
        }
    }

    __aicore__ inline bool MoveNextMulti()
    {
        if (unlikely(BASE_MODULE::isFirstIter_)) {
            return MoveOnFirstIterate();
        } else {
            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M) {
                return MoveOnIterateOrderM();
            } else if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                return MoveOnIterateOrderN();
            } else {
                if (likely(
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder() ==
                        static_cast<int>(IterateOrder::ORDER_N))) {
                    ASCENDC_ASSERT(
                        (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder() ==
                         static_cast<int>(IterateOrder::ORDER_N)),
                        {
                            KERNEL_LOG(
                                KERNEL_ERROR, "iterateOrder is %d , which should be ORDER_N",
                                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder());
                        });
                    return MoveOnIterateOrderN();
                } else {
                    return MoveOnIterateOrderM();
                }
            }
        }
        return true;
    }

    __aicore__ inline bool MoveOnIterateOrderM()
    {
        if constexpr (DoMatmulIBShareNorm(MM_CFG) && B_TYPE::ibShare) {
            ASCENDC_ASSERT(
                (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() >= MATMUL_MODULE(NLoop)->GetTotalIter()), {
                    KERNEL_LOG(
                        KERNEL_ERROR, "When iterateOrder is orderM and B is IBShare, stepN >= nIter is required");
                });
        }
        // when N inner loop is finished, clear left matrix's data in L1 buffer, and restart N inner loop
        if (!MATMUL_MODULE(NLoop)->InnerNext()) {
            MATMUL_MODULE(CopyCubeInA)->Reset();
            MATMUL_MODULE(NLoop)->InnerStart();
            // when M outer and inner loop both are finished, clear right matrix's data in L1 buffer,
            // and restart M outer and inner loop
            if (!MATMUL_MODULE(MLoop)->InnerNext()) {
                if (!MATMUL_MODULE(MLoop)->OuterNext()) {
                    MATMUL_MODULE(MLoop)->OuterStart();
                    MATMUL_MODULE(MLoop)->InnerStart();
                    MATMUL_MODULE(CopyCubeInB)->Reset();
                    MATMUL_MODULE(CopyCubeInScaleB)->Reset();
                    // when N outer loop is finished, all the iterations are finished, end process
                    if (!MATMUL_MODULE(NLoop)->OuterNext()) {
                        return false;
                    }
                    // M loop is restarted, N inner loop should be restarted
                    MATMUL_MODULE(NLoop)->InnerStart();
                }
            }
        }
        return true;
    }

    __aicore__ inline bool MoveOnIterateOrderN()
    {
        // when M inner loop is finished, clear right matrix's data in L1 buffer, and restart M inner loop
        if (!MATMUL_MODULE(MLoop)->InnerNext()) {
            MATMUL_MODULE(CopyCubeInB)->Reset();
            MATMUL_MODULE(CopyCubeInScaleB)->Reset();
            MATMUL_MODULE(MLoop)->InnerStart();
            if (!MoveOnIterateOrderNInner()) {
                return false;
            }
        }
        return true;
    }

private:
    __aicore__ inline bool MoveOnIterateOrderNInner()
    {
        // when N outer and inner loop both are finished, clear left matrix's data in L1 buffer,
        // and restart N outer and inner loop.
        if (!MATMUL_MODULE(NLoop)->InnerNext()) {
            if (!MATMUL_MODULE(NLoop)->OuterNext()) {
                MATMUL_MODULE(NLoop)->OuterStart();
                MATMUL_MODULE(NLoop)->InnerStart();
                MATMUL_MODULE(CopyCubeInA)->Reset();
                // when M outer loop is finished, all the iterations are finished, end process
                if (!MATMUL_MODULE(MLoop)->OuterNext()) {
                    return false;
                }
                // N loop is restarted, M inner loop should be restarted
                MATMUL_MODULE(MLoop)->InnerStart();
            }
        }
        return true;
    }

    __aicore__ inline void ComputeOneIter(bool partialSumEnabled)
    {
        // init split params for left and right matrix
        MxSplitParams aSplitParams = InitSplitAParams();
        MxSplitParams bSplitParams = InitSplitBParams();
        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        // CopyIn
        LocalTensor<SrcAT> aInput =
            MATMUL_MODULE(CopyCubeInA)
                ->LoadData(0, 0, MATMUL_MODULE(MLoop)->GetTileShape(), MATMUL_MODULE(KLoop)->GetTileShapeA());
        LocalTensor<SrcBT> bInput =
            MATMUL_MODULE(CopyCubeInB)
                ->LoadData(0, 0, MATMUL_MODULE(KLoop)->GetTileShapeB(), MATMUL_MODULE(NLoop)->GetTileShape());

        LocalTensor<SrcScaleAT> scaleAInput = {};
        LocalTensor<SrcScaleBT> scaleBInput =
            MATMUL_MODULE(CopyCubeInScaleB)
                ->LoadData(
                    0, 0, Ceil(MATMUL_MODULE(KLoop)->GetTileShapeB(), GetMxKfactor<SrcScaleBT>()),
                    MATMUL_MODULE(NLoop)->GetTileShape());

        BASE_MODULE::SplitBias(bSplitParams.axisL0Len);

        // prepare for Split
        bool bTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        bool aTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool scaleATransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleA();
        bool scaleBTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleB();
        // update some params in MxSplitParams which is related to k loop
        SplitPrepare(aTransposed, bTransposed, scaleATransposed, scaleBTransposed, aSplitParams, bSplitParams);
        MATMUL_MODULE(TBufPoolL0)->Allocate();
        // L1->L0
        LocalTensor<SrcA2T> aL0Input = SplitA(aInput, scaleAInput, aSplitParams, aTransposed, scaleATransposed);
        LocalTensor<SrcB2T> bL0Input = SplitB(bInput, scaleBInput, bSplitParams, bTransposed, scaleBTransposed);
        MATMUL_MODULE(TBufPoolL0)->EnQue();
        MATMUL_MODULE(TBufPoolL0)->DeQue();

        // prepare params and compute
        CubeCompute(
            MATMUL_MODULE(CubeOutBuffer)->GetTensor(), aL0Input, bL0Input, aSplitParams.axisL0Len,
            bSplitParams.axisL0Len, CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeA(), c0Size_), aTransposed, bTransposed,
            partialSumEnabled, !partialSumEnabled, true);

        MATMUL_MODULE(CopyCubeInA)->ClearLoadData(aInput);
        MATMUL_MODULE(CopyCubeInB)->ClearLoadData(bInput);
        MATMUL_MODULE(CopyCubeInScaleB)->ClearLoadData(scaleBInput);
    }

    __aicore__ inline void ComputeMultiIter(bool partialSumEnabled)
    {
        // init split params for left and right matrix
        MxSplitParams aSplitParams = InitSplitAParams();
        MxSplitParams bSplitParams = InitSplitBParams();

        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        do {
            int32_t kOuterLoopIdx = MATMUL_MODULE(KLoop)->GetOuterIdx();
            // CopyIn
            LocalTensor<SrcAT> aInput =
                MATMUL_MODULE(CopyCubeInA)
                    ->LoadData(
                        MATMUL_MODULE(MLoop)->GetInnerIdx(), kOuterLoopIdx, MATMUL_MODULE(MLoop)->GetTileShape(),
                        MATMUL_MODULE(KLoop)->GetTileShapeA());
            LocalTensor<SrcBT> bInput =
                MATMUL_MODULE(CopyCubeInB)
                    ->LoadData(
                        kOuterLoopIdx, MATMUL_MODULE(NLoop)->GetInnerIdx(), MATMUL_MODULE(KLoop)->GetTileShapeB(),
                        MATMUL_MODULE(NLoop)->GetTileShape());

            LocalTensor<SrcScaleAT> scaleAInput = {};
            LocalTensor<SrcScaleBT> scaleBInput =
                MATMUL_MODULE(CopyCubeInScaleB)
                    ->LoadData(
                        kOuterLoopIdx, MATMUL_MODULE(NLoop)->GetInnerIdx(),
                        Ceil(MATMUL_MODULE(KLoop)->GetTileShapeB(), GetMxKfactor<SrcScaleBT>()),
                        MATMUL_MODULE(NLoop)->GetTileShape());

            if (MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA()) {
                PadZerosOuterAxis(
                    aInput, MATMUL_MODULE(KLoop)->GetTileShapeA(), MATMUL_MODULE(MLoop)->GetTileShape(),
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK(),
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM());
            }

            if (!MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB()) {
                PadZerosOuterAxis(
                    bInput, MATMUL_MODULE(KLoop)->GetTileShapeB(), MATMUL_MODULE(NLoop)->GetTileShape(),
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK(),
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            }
            // update some params in MxSplitParams which is related to k loop
            bool cmatrixInit = false;
            bool cmatrixLast = false;
            BASE_MODULE::UpdateComputeParams(partialSumEnabled, cmatrixInit, cmatrixLast);
            BASE_MODULE::SplitBias(bSplitParams.axisL0Len);
            bool aTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
            bool bTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
            bool scaleATransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleA();
            bool scaleBTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleB();
            SplitPrepare(aTransposed, bTransposed, scaleATransposed, scaleBTransposed, aSplitParams, bSplitParams);
            // allocate L0 buffer
            MATMUL_MODULE(TBufPoolL0)->Allocate();
            LocalTensor<SrcA2T> aL0Input = SplitA(aInput, scaleAInput, aSplitParams, aTransposed, scaleATransposed);
            LocalTensor<SrcB2T> bL0Input = SplitB(bInput, scaleBInput, bSplitParams, bTransposed, scaleBTransposed);
            MATMUL_MODULE(TBufPoolL0)->EnQue();
            MATMUL_MODULE(TBufPoolL0)->DeQue();
            // prepare params and compute
            CubeCompute(
                MATMUL_MODULE(CubeOutBuffer)->GetTensor(), aL0Input, bL0Input, aSplitParams.axisL0Len,
                bSplitParams.axisL0Len, CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeA(), c0Size_), aTransposed,
                bTransposed, partialSumEnabled, cmatrixInit, cmatrixLast);

            MATMUL_MODULE(CopyCubeInA)->ClearLoadData(aInput, MATMUL_MODULE(MLoop)->GetInnerIdx(), kOuterLoopIdx);
            MATMUL_MODULE(CopyCubeInB)->ClearLoadData(bInput, kOuterLoopIdx, MATMUL_MODULE(NLoop)->GetInnerIdx());
            MATMUL_MODULE(CopyCubeInScaleB)
                ->ClearLoadData(scaleBInput, kOuterLoopIdx, MATMUL_MODULE(NLoop)->GetInnerIdx());
        } while (MATMUL_MODULE(KLoop)->OuterNext());
    }

private:
    template <typename T>
    __aicore__ inline void PadZerosOuterAxis(LocalTensor<T>& dst, int height, int width, int baseHeight, int baseWidth)
    {
        constexpr int32_t c0Size = AuxGetC0Size<T>();
        if (height < baseHeight) {
            const uint32_t alignedBaseHeight = Ceil(height, c0Size) * c0Size;
            auto zeroPadTensor = dst.template ReinterpretCast<uint16_t>();
            InitConstValueParams<uint16_t> zeroInitParams;
            zeroInitParams.repeatTimes = Ceil(width, c0Size);
            zeroInitParams.initValue = 0;
            zeroInitParams.dstGap = height;
            zeroInitParams.blockNum = alignedBaseHeight - height;
            uint32_t padStartOffset = height * BLOCK_CUBE;
            InitConstValue(zeroPadTensor[padStartOffset], zeroInitParams);
        }
    }

    __aicore__ inline void SplitPrepare(
        const bool aTransposed, const bool bTransposed, const bool scaleATransposed, const bool scaleBTransposed,
        MxSplitParams& aSplitParams, MxSplitParams& bSplitParams)
    {
        UpdateSplitParams(aSplitParams, bSplitParams);
        UpdateSplitMxParams(aSplitParams, bSplitParams, aTransposed, bTransposed);
        MATMUL_MODULE(LoadToA2)->Prepare(aTransposed, aSplitParams.kAxisL1Len, aSplitParams.axisL1Len);
        MATMUL_MODULE(LoadToB2)->Prepare(bTransposed, bSplitParams.kAxisL1Len);
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void UpdateSplitParams(MxSplitParams& aSplitParams, MxSplitParams& bSplitParams)
    {
        // update Split params related to K loop
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            aSplitParams.kAxisL1Offset =
                MATMUL_MODULE(KLoop)->GetOuterIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        } else if constexpr (!IsStaticPaddingEnable(MM_CFG)) {
            if constexpr (!IS_INTRA_BLOCK) {
                aSplitParams.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeA() * c0Size_;
            } else {
                aSplitParams.kAxisL1Len = MATMUL_MODULE(KLoop)->template GetTileBlockShapeA<true>() * c0Size_;
            }
        }
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
            bSplitParams.kAxisL1Offset =
                MATMUL_MODULE(KLoop)->GetOuterIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        } else if constexpr (!IsStaticPaddingEnable(MM_CFG)) {
            if constexpr (!IS_INTRA_BLOCK) {
                bSplitParams.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeB() * c0Size_;
            } else {
                bSplitParams.kAxisL1Len = MATMUL_MODULE(KLoop)->template GetTileBlockShapeB<true>() * c0Size_;
            }
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void UpdateSplitMxParams(
        MxSplitParams& aSplitParams, MxSplitParams& bSplitParams, const bool aTransposed, const bool bTransposed)
    {
        constexpr int32_t MX_EVEN_FACTOR = 2;
        if (aTransposed) {
            int32_t blockUseK = Ceil(MATMUL_MODULE(KLoop)->GetTileShapeA(), BLOCK_CUBE);
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
                blockUseK = Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), BLOCK_CUBE);
            }
            aSplitParams.axisL0Len = Ceil(MATMUL_MODULE(MLoop)->GetTileShape(), c0Size_) * c0Size_;
            aSplitParams.kAxisL1Len = Ceil(blockUseK, MX_EVEN_FACTOR) * MX_EVEN_FACTOR * BLOCK_CUBE;
        }
        if (!bTransposed) {
            int32_t blockUseK = Ceil(MATMUL_MODULE(KLoop)->GetTileShapeB(), BLOCK_CUBE);
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
                blockUseK = Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), BLOCK_CUBE);
            }
            bSplitParams.axisL0Len = Ceil(MATMUL_MODULE(NLoop)->GetTileShape(), c0Size_) * c0Size_;
            bSplitParams.kAxisL1Len = Ceil(blockUseK, MX_EVEN_FACTOR) * MX_EVEN_FACTOR * BLOCK_CUBE;
        }

        if constexpr (!IS_INTRA_BLOCK) {
            aSplitParams.kAuxMatrixL1Len =
                Ceil(MATMUL_MODULE(KLoop)->GetTileBlockShapeA() * c0Size_, GetMxKfactor<SrcScaleAT>());
        } else {
            aSplitParams.kAuxMatrixL1Len =
                Ceil(MATMUL_MODULE(KLoop)->template GetTileBlockShapeA<true>() * c0Size_, GetMxKfactor<SrcScaleAT>());
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            bSplitParams.kAuxMatrixL1Offset = Ceil(
                MATMUL_MODULE(KLoop)->GetOuterIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK(),
                GetMxKfactor<SrcScaleBT>());
        } else {
            if constexpr (!IS_INTRA_BLOCK) {
                bSplitParams.kAuxMatrixL1Len =
                    Ceil(MATMUL_MODULE(KLoop)->GetTileBlockShapeB() * c0Size_, GetMxKfactor<SrcScaleBT>());
            } else {
                bSplitParams.kAuxMatrixL1Len = Ceil(
                    MATMUL_MODULE(KLoop)->template GetTileBlockShapeB<true>() * c0Size_, GetMxKfactor<SrcScaleBT>());
            }
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline MxSplitParams InitSplitAParams()
    {
        MxSplitParams aSplitParams;
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            aSplitParams.axisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM(), BLOCK_CUBE);
            aSplitParams.axisL0Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            aSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK(), c0Size_);
        } else {
            if constexpr (IS_INTRA_BLOCK) {
                aSplitParams.axisL1Len = MATMUL_MODULE(MLoop)->template GetTileBlockShape<true>() * BLOCK_CUBE;
                aSplitParams.axisL0Len = MATMUL_MODULE(MLoop)->template GetBaseShape<true>();
            } else {
                aSplitParams.axisL1Len = MATMUL_MODULE(MLoop)->GetTileBlockShape() * BLOCK_CUBE;
                aSplitParams.axisL0Len = MATMUL_MODULE(MLoop)->GetBaseShape();
            }
        }

        aSplitParams.kAxisL1Offset = 0;
        aSplitParams.axisL0Len = BASE_MODULE::GetFixedMadM(aSplitParams.axisL0Len);
        // if input is from L1, update related params
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            if constexpr (IsBasic(MM_CFG)) {
                aSplitParams.axisL1Offset = 0;
            } else {
                aSplitParams.axisL1Offset =
                    MATMUL_MODULE(MLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            }
            aSplitParams.axisL1Len =
                CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreM<IS_INTRA_BLOCK>(), BLOCK_CUBE);
            if (MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeA<IS_INTRA_BLOCK>()) {
                aSplitParams.kAxisL1Len =
                    CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreK<IS_INTRA_BLOCK>(), BLOCK_CUBE);
            } else {
                aSplitParams.kAxisL1Len =
                    CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreK<IS_INTRA_BLOCK>(), c0Size_);
            }
        } else {
            aSplitParams.axisL1Offset = 0;
        }
        // for scaleA
        aSplitParams.auxMatrixL1Offset = 0;
        aSplitParams.kAuxMatrixL1Offset = 0;
        return aSplitParams;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline MxSplitParams InitSplitBParams()
    {
        MxSplitParams bSplitParams;
        // if it's constant tiling sence, get params from tiling, else get params from loop
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            bSplitParams.axisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN(), BLOCK_CUBE);
            bSplitParams.axisL0Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            bSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK(), c0Size_);
        } else {
            if constexpr (IS_INTRA_BLOCK) {
                bSplitParams.axisL1Len =
                    MATMUL_MODULE(NLoop)->template GetTileBlockShape<IS_INTRA_BLOCK>() * BLOCK_CUBE;
                bSplitParams.axisL0Len = MATMUL_MODULE(NLoop)->template GetBaseShape<IS_INTRA_BLOCK>();
            } else {
                bSplitParams.axisL1Len = MATMUL_MODULE(NLoop)->GetTileBlockShape() * BLOCK_CUBE;
                bSplitParams.axisL0Len = MATMUL_MODULE(NLoop)->GetBaseShape();
            }
        }
        bSplitParams.kAxisL1Offset = 0;
        // if input is from L1, update related params
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
            if constexpr (IsBasic(MM_CFG)) {
                bSplitParams.axisL1Offset = 0;
            } else {
                bSplitParams.axisL1Offset =
                    MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            }
            bSplitParams.axisL1Len =
                CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreN<IS_INTRA_BLOCK>(), BLOCK_CUBE);
            if (MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeB<IS_INTRA_BLOCK>()) {
                bSplitParams.kAxisL1Len =
                    CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreK<IS_INTRA_BLOCK>(), c0Size_);
            } else {
                bSplitParams.kAxisL1Len =
                    CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreK<IS_INTRA_BLOCK>(), BLOCK_CUBE);
            }
        } else {
            bSplitParams.axisL1Offset = 0;
        }
        // for scaleB
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            bSplitParams.kAuxMatrixL1Len = Ceil(
                CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->template GetSingleCoreK<IS_INTRA_BLOCK>(), c0Size_),
                GetMxKfactor<SrcScaleBT>());
            bSplitParams.auxMatrixL1Offset =
                MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        } else {
            bSplitParams.auxMatrixL1Offset = 0;
        }
        bSplitParams.kAuxMatrixL1Offset = 0;
        return bSplitParams;
    }

    __aicore__ inline LocalTensor<SrcA2T> SplitA(
        const LocalTensor<SrcAT>& aInput, const LocalTensor<SrcScaleAT>& scaleAInput, const MxSplitParams& aSplitParams,
        const bool aTransposed, const bool scaleATransposed)
    {
        auto aCachePos = MATMUL_MODULE(MLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() +
                         MATMUL_MODULE(KLoop)->GetInnerIdx();
        LocalTensor<SrcA2T> aL0Input;
        int32_t kL0Length = Ceil(MATMUL_MODULE(KLoop)->GetBaseShape(), c0Size_) * c0Size_;
        // Split
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::A2>(aCachePos))) {
            aL0Input = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, SrcA2T>();
            MATMUL_MODULE(LoadToA2)->Load(
                aL0Input, aInput, aSplitParams.axisL1Len, aSplitParams.kAxisL1Len, aSplitParams.axisL0Len, kL0Length,
                aSplitParams.axisL1Offset, aSplitParams.kAxisL1Offset, aTransposed, scaleAInput,
                aSplitParams.kAuxMatrixL1Len, aSplitParams.kAuxMatrixL1Offset, aSplitParams.auxMatrixL1Offset);
        } else {
            aL0Input = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, SrcA2T>();
        }
        return aL0Input;
    }

    __aicore__ inline LocalTensor<SrcB2T> SplitB(
        const LocalTensor<SrcBT>& bInput, const LocalTensor<SrcScaleBT>& scaleBInput, const MxSplitParams& bSplitParams,
        const bool bTransposed, const bool scaleATransposed)
    {
        auto bCachePos = MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() +
                         MATMUL_MODULE(KLoop)->GetInnerIdx();
        LocalTensor<SrcB2T> bL0Input;
        int32_t kL0Length = Ceil(MATMUL_MODULE(KLoop)->GetBaseShape(), c0Size_) * c0Size_;
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::B2>(bCachePos))) {
            bL0Input = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, SrcB2T>();
            MATMUL_MODULE(LoadToB2)->Load(
                bL0Input, bInput, bSplitParams.axisL1Len, bSplitParams.kAxisL1Len, bSplitParams.axisL0Len, kL0Length,
                bSplitParams.axisL1Offset, bSplitParams.kAxisL1Offset, bTransposed, scaleBInput,
                bSplitParams.kAuxMatrixL1Len, bSplitParams.kAuxMatrixL1Offset, bSplitParams.auxMatrixL1Offset);
        } else {
            bL0Input = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, SrcB2T>();
        }
        return bL0Input;
    }

    __aicore__ inline void CubeCompute(
        const LocalTensor<L0cT>& outputMatrix, const LocalTensor<SrcA2T>& aL0Input, const LocalTensor<SrcB2T>& bL0Input,
        const uint16_t madSizeM, const uint16_t madSizeN, const uint16_t madSizeK, const bool aTransposed,
        const bool bTransposed, const bool partialSumEnabled, const bool cmatrixInit, const bool cmatrixLast)
    {
        uint8_t mmadUnitFlag = MATMUL_MODULE(MatmulUnitFlag)->GetUnitFlag(cmatrixLast);
        bool isBias;
        bool mmadInitValue;
        bool mmadSource;
        BASE_MODULE::UpdateBiasParams(partialSumEnabled, cmatrixInit, mmadSource, mmadInitValue, isBias);
        MATMUL_MODULE(MmadCompute)
            ->Compute(
                outputMatrix, aL0Input, bL0Input, madSizeM, madSizeK, madSizeN, aTransposed, bTransposed, mmadUnitFlag,
                mmadSource, mmadInitValue, isBias);
        MATMUL_MODULE(TBufPoolL0)->Free();
        // clear data in related buffers
        MATMUL_MODULE(BiasScheduler)->Free();
    }

    constexpr static int32_t c0Size_ = AuxGetC0Size<typename A_TYPE::T>();
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC

#endif
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_MX_ONLY_SCALEB_H)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_NORM_MX_ONLY_SCALEB_H
#endif
