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
 * \file scheduler_mdl_mx_only_scaleb.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/scheduler/base/scheduler_mdl_mx_only_scaleb.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_MX_SCALEB_SCHEDULER_H__
#endif

#ifndef IMPL_MATMUL_MODULES_SCHEDULER_SCHEDULER_MDL_MX_ONLY_SCALEB_H
#define IMPL_MATMUL_MODULES_SCHEDULER_SCHEDULER_MDL_MX_ONLY_SCALEB_H

#include "scheduler_intf.h"
#include "scheduler_mdl_common.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <
    typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    PolicyType POLICY_TYPE>
class MatmulScheduler<
    IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE,
    enable_if_t<DoMatmulMDL(MM_CFG) && isOnlyScaleBMxMatmul<A_TYPE, B_TYPE>>>
    : public MatmulMDLSchedulerCommon<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE> {
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(CopyCubeInScaleB);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE(MatmulUnitFlag);
    MATMUL_USE_MODULE(MmadCompute);
    MATMUL_USE_MODULE(MatmulCrossCoreSync);

    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using SrcScaleAT = uint8_t;
    using SrcScaleBT = typename B_TYPE::SCALE_T;
    using BiasT = typename BIAS_TYPE::T;
    using SrcA2T = typename GetL0DataType<typename A_TYPE::T, true>::Type;
    using SrcB2T = typename GetL0DataType<typename B_TYPE::T, true>::Type;
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulMDLSchedulerCommon<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE>;
    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe)
    {
        BASE_MODULE::Init(cubeTiling, tpipe);
        MATMUL_MODULE(CopyCubeInScaleB)->Init();
        InitShareBufEnd(tpipe);
    }

    __aicore__ inline bool ScheduleOnce(bool partialSumEnabled)
    {
        MATMUL_MODULE(BiasScheduler)->SetBias(!partialSumEnabled && MATMUL_MODULE(BiasScheduler)->IsBias());
        if (!BASE_MODULE::MoveNext()) {
            return false;
        }
        if (!partialSumEnabled) {
            MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
        }
        ReduceK(partialSumEnabled);
        return true;
    }

private:
    __aicore__ inline MxSplitParams InitSplitAParams()
    {
        MxSplitParams aSplitParams;
        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        int32_t baseTileM = tiling.GetBaseM();
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            aSplitParams.axisL1Offset = MATMUL_MODULE(MLoop)->GetInnerIdx() * baseTileM;
            // ds && 82 mdl support multi singleshape in l1
            if constexpr (IsFullStaticTiling(MM_CFG) || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape()) {
                aSplitParams.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() != -1 ?
                                             MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() :
                                             MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM();
                aSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() != -1 ?
                                              MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() :
                                              MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
            } else {
                aSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
                aSplitParams.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM();
            }
        } else if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            aSplitParams.axisL1Len = tiling.GetStepM() * baseTileM;
            aSplitParams.kAxisL1Len = tiling.GetStepKa() * tiling.GetBaseK();
            aSplitParams.axisL1Offset =
                (MATMUL_MODULE(MLoop)->GetInnerIdx() - MATMUL_MODULE(MLoop)->GetOuterIdx() * tiling.GetStepM()) *
                baseTileM;
        } else {
            aSplitParams.axisL1Len = MATMUL_MODULE(MLoop)->GetTileBlockShape() * BLOCK_CUBE;
            aSplitParams.axisL1Offset =
                (MATMUL_MODULE(MLoop)->GetInnerIdx() - MATMUL_MODULE(MLoop)->GetOuterIdx() * tiling.GetStepM()) *
                baseTileM;
        }

        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            aSplitParams.axisL0Len = baseTileM;
        } else {
            aSplitParams.axisL0Len = MATMUL_MODULE(MLoop)->GetBaseShape();
        }
        if constexpr ((A_TYPE::format == CubeFormat::VECTOR) || (A_TYPE::format == CubeFormat::SCALAR)) {
            aSplitParams.axisL0Len = 1;
        } else {
            if (aSplitParams.axisL0Len == 1) {
                aSplitParams.axisL0Len = BLOCK_CUBE;
            }
        }
        return aSplitParams;
    }

    __aicore__ inline int32_t GetBL1OffsetFromGM()
    {
        int32_t baseTileN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        int32_t stepN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN();
        if (!MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB()) {
            if (baseTileN % BASE_MODULE::c0Size_ == 0 || stepN == 1) {
                return (MATMUL_MODULE(NLoop)->GetInnerIdx() - MATMUL_MODULE(NLoop)->GetOuterIdx() * stepN) * baseTileN;
            } else {
                return (MATMUL_MODULE(NLoop)->GetInnerIdx() - MATMUL_MODULE(NLoop)->GetOuterIdx() * stepN) *
                       CeilAlign(baseTileN, BASE_MODULE::c0Size_);
            }
        } else {
            return (MATMUL_MODULE(NLoop)->GetInnerIdx() - MATMUL_MODULE(NLoop)->GetOuterIdx() * stepN) * baseTileN;
        }
    }

    __aicore__ inline MxSplitParams InitSplitBParams()
    {
        MxSplitParams bSplitParams;
        int32_t baseTileN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        bSplitParams.axisL0Len = MATMUL_MODULE(NLoop)->GetBaseShape();
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            bSplitParams.axisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() * baseTileN;
            bSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb() *
                                      MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            bSplitParams.axisL0Len = baseTileN;
            bSplitParams.axisL1Offset = GetBL1OffsetFromGM();
        } else {
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
                bSplitParams.axisL1Offset = MATMUL_MODULE(NLoop)->GetInnerIdx() * baseTileN;
                // ds && 82 mdl support multi singleshape in l1
                if constexpr (IsFullStaticTiling(MM_CFG) || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape()) {
                    bSplitParams.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgN() != -1 ?
                                                 MATMUL_MODULE(MatmulShapeInfo)->GetOrgN() :
                                                 MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
                    bSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() != -1 ?
                                                  MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() :
                                                  MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
                } else {
                    bSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
                    bSplitParams.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
                }
            } else {
                bSplitParams.axisL1Len = MATMUL_MODULE(NLoop)->GetTileBlockShape() * BLOCK_CUBE;
                bSplitParams.axisL1Offset = GetBL1OffsetFromGM();
            }
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            bSplitParams.auxMatrixL1Offset = MATMUL_MODULE(NLoop)->GetInnerIdx() * baseTileN;
            bSplitParams.kAuxMatrixL1Len =
                Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), GetMxKfactor<SrcScaleBT>());
        } else {
            bSplitParams.auxMatrixL1Offset = GetBL1OffsetFromGM();
        }
        return bSplitParams;
    }

    __aicore__ inline void SplitPrepare(
        const bool partialSumEnabled, const bool aTransposed, const bool bTransposed, MxSplitParams& aSplitParams,
        MxSplitParams& bSplitParams, bool& cmatrixInit, bool& cmatrixLast)
    {
        UpdateSplitParams(aSplitParams, bSplitParams);
        BASE_MODULE::UpdateComputeParams(partialSumEnabled, cmatrixInit, cmatrixLast);
        if (aTransposed) {
            aSplitParams.axisL1Len = CeilAlign(MATMUL_MODULE(MLoop)->GetTileShape(), BASE_MODULE::c0Size_);
            aSplitParams.axisL0Len = CeilAlign(MATMUL_MODULE(MLoop)->GetBaseShape(), BASE_MODULE::c0Size_);
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
                aSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR);
            } else {
                aSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeA(), MX_BASEK_FACTOR);
            }
        }
        if (!bTransposed) {
            bSplitParams.axisL1Len = CeilAlign(MATMUL_MODULE(NLoop)->GetTileShape(), BASE_MODULE::c0Size_);
            bSplitParams.axisL0Len = CeilAlign(MATMUL_MODULE(NLoop)->GetBaseShape(), BASE_MODULE::c0Size_);
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
                bSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR);
            } else {
                bSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeB(), MX_BASEK_FACTOR);
            }
            if constexpr (IsScaleTransWithInlv<B_TYPE>) {
                if (IsSameTypeV<SrcScaleBT, AscendC::hif4_scale>) {
                    bSplitParams.kAuxMatrixL1Len =
                        Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR);
                } else {
                    bSplitParams.kAuxMatrixL1Len =
                        Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR) * MX_EVEN_FACTOR;
                }
            }
        }
        MATMUL_MODULE(LoadToA2)->Prepare(aTransposed, aSplitParams.kAxisL1Len, aSplitParams.axisL1Len);
        MATMUL_MODULE(LoadToB2)->Prepare(bTransposed, bSplitParams.kAxisL1Len);
    }

    __aicore__ inline void UpdateSplitParams(MxSplitParams& aSplitParams, MxSplitParams& bSplitParams)
    {
        int32_t kInnerLoopIdx = MATMUL_MODULE(KLoop)->GetInnerStartIdx();
        int32_t baseTileK = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            aSplitParams.kAxisL1Offset = kInnerLoopIdx * baseTileK;
        } else {
            // Mx scene K direction should be aligned to 64, baseK is already guaranteed to be aligned to 64
            aSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeA(), MX_BASEK_FACTOR);
            int32_t stepKa = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
            aSplitParams.kAxisL1Offset = (kInnerLoopIdx - kInnerLoopIdx / stepKa * stepKa) * baseTileK;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
            bSplitParams.kAxisL1Offset = kInnerLoopIdx * baseTileK;
        } else {
            // Mx scene K direction should be aligned to 64, baseK is already guaranteed to be aligned to 64
            bSplitParams.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeB(), MX_BASEK_FACTOR);
            int32_t stepKb = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
            bSplitParams.kAxisL1Offset = (kInnerLoopIdx - kInnerLoopIdx / stepKb * stepKb) * baseTileK;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            if constexpr (PhyPosIsL1(B_TYPE::scalePosition)) {
                bSplitParams.kAuxMatrixL1Len = Ceil(
                    MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK() * MATMUL_MODULE(KLoop)->GetScaleFactorKb(),
                    GetMxKfactor<SrcScaleBT>());
            }
            bSplitParams.kAuxMatrixL1Offset = Ceil(kInnerLoopIdx * baseTileK, GetMxKfactor<SrcScaleBT>());
        } else {
            int32_t stepKb = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
            bSplitParams.kAuxMatrixL1Len = MATMUL_MODULE(KLoop)->GetTileShapeScaleKb();
            bSplitParams.kAuxMatrixL1Offset =
                (kInnerLoopIdx - kInnerLoopIdx / (stepKb * MATMUL_MODULE(KLoop)->GetScaleFactorKb()) *
                                     (stepKb * MATMUL_MODULE(KLoop)->GetScaleFactorKb())) *
                Ceil(baseTileK, GetMxKfactor<SrcScaleBT>());
        }
    }

    __aicore__ inline void CopyIn(
        LocalTensor<SrcAT>& aInput, LocalTensor<SrcBT>& bInput, LocalTensor<SrcScaleAT>& scaleAInput,
        LocalTensor<SrcScaleBT>& scaleBInput)
    {
        aInput = MATMUL_MODULE(CopyCubeInA)
                     ->LoadData(
                         MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(KLoop)->GetInnerStartIdx(),
                         MATMUL_MODULE(MLoop)->GetTileShape(), MATMUL_MODULE(KLoop)->GetTileShapeA());
        bInput = MATMUL_MODULE(CopyCubeInB)
                     ->LoadData(
                         MATMUL_MODULE(KLoop)->GetInnerStartIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                         MATMUL_MODULE(KLoop)->GetTileShapeB(), MATMUL_MODULE(NLoop)->GetTileShape());
        scaleBInput = MATMUL_MODULE(CopyCubeInScaleB)
                          ->LoadData(
                              MATMUL_MODULE(KLoop)->GetInnerStartIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                              MATMUL_MODULE(KLoop)->GetTileShapeScaleKb(), MATMUL_MODULE(NLoop)->GetTileShape());
        if constexpr (MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape()) {
            MATMUL_MODULE(MatmulCrossCoreSync)->WaitL1Ready();
        }
    }

    __aicore__ inline void ReduceKOneIter(bool partialSumEnabled)
    {
        // init split params for left and right matrix
        MxSplitParams aSplitParams = InitSplitAParams();
        MxSplitParams bSplitParams = InitSplitBParams();
        MATMUL_MODULE(KLoop)->OuterStart();
        // CopyIn
        LocalTensor<SrcAT> aInput;
        LocalTensor<SrcBT> bInput;
        LocalTensor<SrcScaleAT> scaleAInput;
        LocalTensor<SrcScaleBT> scaleBInput;
        CopyIn(aInput, bInput, scaleAInput, scaleBInput);
        bool aTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool bTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        bool scaleATransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleA();
        bool scaleBTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleB();
        SplitPrepareOneIter(aTransposed, bTransposed, aSplitParams, bSplitParams);
        LocalTensor<BiasT> bias = BASE_MODULE::SplitBias(bSplitParams.axisL0Len);
        int32_t kStride;
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            kStride = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        } else {
            kStride = MATMUL_MODULE(KLoop)->GetBaseBlockShape() * BASE_MODULE::c0Size_;
        }
        int32_t scaleBStride = Ceil(kStride, GetMxKfactor<SrcScaleBT>());
        // start k inner loop
        MATMUL_MODULE(KLoop)->InnerStart();
        do {
            ComputeKDB(
                aInput, bInput, scaleAInput, scaleBInput, aSplitParams, bSplitParams, aTransposed, bTransposed,
                !partialSumEnabled, true);
            aSplitParams.kAxisL1Offset += kStride;
            bSplitParams.kAxisL1Offset += kStride;
            bSplitParams.kAuxMatrixL1Offset += scaleBStride;
        } while (MATMUL_MODULE(KLoop)->InnerNext());
        if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
            MATMUL_MODULE(BiasScheduler)->Free(bias);
        }
    }

    __aicore__ inline void ReduceK(bool partialSumEnabled)
    {
        if (BASE_MODULE::IsMDLKFullLoad()) {
            ReduceKOneIter(partialSumEnabled);
        } else {
            ReduceKMultiIter(partialSumEnabled);
        }
    }

    __aicore__ inline void ReduceKMultiIter(bool partialSumEnabled)
    {
        // init split params for left and right matrix
        MxSplitParams aSplitParams = InitSplitAParams();
        MxSplitParams bSplitParams = InitSplitBParams();
        bool aTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool bTransposed = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        // curKaOuterIdx and curKbOuterIdx are used to decide if left or right matrix need to clear its l1 buffer
        int32_t curKaOuterIdx = MATMUL_MODULE(KLoop)->GetOuterKaIdx();
        int32_t curKbOuterIdx = MATMUL_MODULE(KLoop)->GetOuterKbIdx();
        int32_t curScaleKaOuterIdx = MATMUL_MODULE(KLoop)->GetOuterScaleKaIdx();
        int32_t curScaleKbOuterIdx = MATMUL_MODULE(KLoop)->GetOuterScaleKbIdx();
        do {
            // CopyIn
            LocalTensor<SrcAT> aInput;
            LocalTensor<SrcBT> bInput;
            LocalTensor<SrcScaleAT> scaleAInput;
            LocalTensor<SrcScaleBT> scaleBInput;
            CopyIn(aInput, bInput, scaleAInput, scaleBInput);
            LocalTensor<BiasT> bias = BASE_MODULE::SplitBias(bSplitParams.axisL0Len);
            Compute(
                aInput, bInput, scaleAInput, scaleBInput, bias, partialSumEnabled, aTransposed, bTransposed,
                aSplitParams, bSplitParams);
            if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
                MATMUL_MODULE(BiasScheduler)->Free(bias);
            }
            ClearL1BufferCache(curKaOuterIdx, curKbOuterIdx, curScaleKaOuterIdx, curScaleKbOuterIdx);
        } while (MATMUL_MODULE(KLoop)->OuterNext());
        ResetCopyInBuffer();
    }

    __aicore__ inline void ClearL1BufferCache(
        int32_t& curKaOuterIdx, int32_t& curKbOuterIdx, int32_t& curScaleKaOuterIdx, int32_t& curScaleKbOuterIdx)
    {
        if constexpr (!PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            int32_t curKaIdx = MATMUL_MODULE(KLoop)->GetNextOuterKaIdx();
            // if next outerKaIdx is not equal to curKaOuterIdx, clear left matrix's data in L1 buffer
            if (curKaIdx != curKaOuterIdx && !MATMUL_MODULE(KLoop)->IsAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInA)->ClearLoadData();
                curKaOuterIdx = curKaIdx;
            }
        }

        if constexpr (!PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
            // if next outerKbIdx is not equal to curKbOuterIdx, clear right matrix's data in L1 buffer
            int32_t curKbIdx = MATMUL_MODULE(KLoop)->GetNextOuterKbIdx();
            if (curKbIdx != curKbOuterIdx && !MATMUL_MODULE(KLoop)->IsBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInB)->ClearLoadData();
                curKbOuterIdx = curKbIdx;
            }
        }

        if constexpr (!PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            int32_t curKbIdx = MATMUL_MODULE(KLoop)->GetNextOuterScaleKbIdx();
            // if next outerKaIdx is not equal to curScaleKbOuterIdx, clear leftScale matrix's data in L1 buffer
            if (curKbIdx != curScaleKbOuterIdx && !MATMUL_MODULE(KLoop)->IsScaleBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInScaleB)->ClearLoadData();
                curScaleKbOuterIdx = curKbIdx;
            }
        }
    }

    __aicore__ inline LocalTensor<SrcA2T> SplitA(
        const LocalTensor<SrcAT>& aInput, LocalTensor<SrcScaleAT>& scaleAInput, const MxSplitParams& aSplitParams,
        const bool aTransposed)
    {
        auto aCachePos = MATMUL_MODULE(MLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() +
                         MATMUL_MODULE(KLoop)->GetInnerIdx();
        int32_t kL0Length = CeilAlign(MATMUL_MODULE(KLoop)->GetBaseShape(), MX_BASEK_FACTOR);
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::A2>(aCachePos))) {
            LocalTensor<SrcA2T> aL0Input = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, SrcA2T>();
            MATMUL_MODULE(LoadToA2)->Load(
                aL0Input, aInput, aSplitParams.axisL1Len, aSplitParams.kAxisL1Len, aSplitParams.axisL0Len, kL0Length,
                aSplitParams.axisL1Offset, aSplitParams.kAxisL1Offset, aTransposed, scaleAInput,
                aSplitParams.kAuxMatrixL1Len, aSplitParams.kAuxMatrixL1Offset, aSplitParams.auxMatrixL1Offset);
            return aL0Input;
        } else {
            return MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, SrcA2T>();
        }
    }

    __aicore__ inline LocalTensor<SrcB2T> SplitB(
        const LocalTensor<SrcBT>& bInput, LocalTensor<SrcScaleBT>& scaleBInput, const MxSplitParams& bSplitParams,
        const bool bTransposed)
    {
        auto bCachePos = MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter() +
                         MATMUL_MODULE(KLoop)->GetInnerIdx();
        int32_t kL0Length = CeilAlign(MATMUL_MODULE(KLoop)->GetBaseShape(), MX_BASEK_FACTOR);
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::B2>(bCachePos))) {
            LocalTensor<SrcB2T> bL0Input = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, SrcB2T>();
            MATMUL_MODULE(LoadToB2)->Load(
                bL0Input, bInput, bSplitParams.axisL1Len, bSplitParams.kAxisL1Len, bSplitParams.axisL0Len, kL0Length,
                bSplitParams.axisL1Offset, bSplitParams.kAxisL1Offset, bTransposed, scaleBInput,
                bSplitParams.kAuxMatrixL1Len, bSplitParams.kAuxMatrixL1Offset, bSplitParams.auxMatrixL1Offset);
            return bL0Input;
        } else {
            return MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, SrcB2T>();
        }
    }

    __aicore__ inline void ResetCopyInBuffer()
    {
        // clear L1 buffers
        if constexpr (!PhyPosIsL1(A_TYPE::pos)) {
            if (!MATMUL_MODULE(KLoop)->IsAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInA)->Reset();
            }
        }

        if constexpr (!PhyPosIsL1(B_TYPE::pos)) {
            if (!MATMUL_MODULE(KLoop)->IsBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInB)->Reset();
            }
        }
        if constexpr (!PhyPosIsL1(B_TYPE::scalePosition)) {
            if (!MATMUL_MODULE(KLoop)->IsScaleBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInB)->Reset();
            }
        }
    }

    __aicore__ inline void CubeCompute(
        const LocalTensor<L0cT>& outputMatrix, const LocalTensor<SrcA2T>& aL0Input, const LocalTensor<SrcB2T>& bL0Input,
        const uint16_t madSizeM, const uint16_t madSizeN, const uint16_t madSizeK, const bool aTransposed,
        const bool bTransposed, const bool cmatrixInit, const bool cmatrixLast)
    {
        int32_t innerKStart;
        if (BASE_MODULE::IsMDLKFullLoad()) {
            innerKStart = 0;
        } else {
            innerKStart = MATMUL_MODULE(KLoop)->GetInnerStartIdx();
        }
        auto mmadUnitFlag = MATMUL_MODULE(MatmulUnitFlag)
                                ->GetUnitFlag(
                                    cmatrixLast && (MATMUL_MODULE(KLoop)->GetInnerIdx() ==
                                                    innerKStart + MATMUL_MODULE(KLoop)->GetInnerIter() - 1));
        if constexpr (IsMxDisableUnitFlag<A_TYPE, B_TYPE, MM_CFG>) {
            mmadUnitFlag = 0;
        }
        bool mmadSource;
        bool mmadInitValue;
        BASE_MODULE::UpdateMmadComputeParams(innerKStart, cmatrixInit, mmadSource, mmadInitValue);
        MATMUL_MODULE(MmadCompute)
            ->Compute(
                outputMatrix, aL0Input, bL0Input, madSizeM, madSizeK, madSizeN, aTransposed, bTransposed, mmadUnitFlag,
                mmadSource, mmadInitValue);
    }

    __aicore__ inline void Compute(
        const LocalTensor<SrcAT>& aInput, const LocalTensor<SrcBT>& bInput, LocalTensor<SrcScaleAT>& scaleAInput,
        LocalTensor<SrcScaleBT>& scaleBInput, LocalTensor<BiasT>& bias, const bool partialSumEnabled,
        const bool aTransposed, const bool bTransposed, MxSplitParams& aSplitParams, MxSplitParams& bSplitParams)
    {
        bool cmatrixInit = false;
        bool cmatrixLast = false;
        SplitPrepare(partialSumEnabled, aTransposed, bTransposed, aSplitParams, bSplitParams, cmatrixInit, cmatrixLast);
        int32_t kStride = MATMUL_MODULE(KLoop)->GetBaseBlockShape() * BASE_MODULE::c0Size_;
        int32_t scaleBStride = Ceil(kStride, GetMxKfactor<SrcScaleBT>());
        MATMUL_MODULE(KLoop)->InnerStart();
        do {
            ComputeKDB(
                aInput, bInput, scaleAInput, scaleBInput, aSplitParams, bSplitParams, aTransposed, bTransposed,
                cmatrixInit, cmatrixLast);
            aSplitParams.kAxisL1Offset += kStride;
            bSplitParams.kAxisL1Offset += kStride;
            bSplitParams.kAuxMatrixL1Offset += scaleBStride;
        } while (MATMUL_MODULE(KLoop)->InnerNext());
    }

    __aicore__ inline void ComputeKDB(
        const LocalTensor<SrcAT>& aInput, const LocalTensor<SrcBT>& bInput, LocalTensor<SrcScaleAT>& scaleAInput,
        LocalTensor<SrcScaleBT>& scaleBInput, const MxSplitParams& aSplitParams, const MxSplitParams& bSplitParams,
        const bool aTransposed, const bool bTransposed, const bool cmatrixInit, const bool cmatrixLast)
    {
        MATMUL_MODULE(TBufPoolL0)->Allocate();
        LocalTensor<SrcA2T> aL0Input = SplitA(aInput, scaleAInput, aSplitParams, aTransposed);
        LocalTensor<SrcB2T> bL0Input = SplitB(bInput, scaleBInput, bSplitParams, bTransposed);
        MATMUL_MODULE(TBufPoolL0)->EnQue();
        MATMUL_MODULE(TBufPoolL0)->DeQue();
        CubeCompute(
            MATMUL_MODULE(CubeOutBuffer)->GetTensor(), aL0Input, bL0Input, aSplitParams.axisL0Len,
            bSplitParams.axisL0Len, CeilAlign(MATMUL_MODULE(KLoop)->GetBaseShape(), MX_BASEK_FACTOR), aTransposed,
            bTransposed, cmatrixInit, cmatrixLast);
        MATMUL_MODULE(TBufPoolL0)->Free();
        MATMUL_MODULE(BiasScheduler)->Free();
    }

    __aicore__ inline void SplitPrepareOneIter(
        const bool aTransposed, const bool bTransposed, MxSplitParams& aSplitParams, MxSplitParams& bSplitParams)
    {
        aSplitParams.kAxisL1Offset = 0;
        bSplitParams.kAxisL1Offset = 0;
        aSplitParams.kAuxMatrixL1Offset = 0;
        bSplitParams.kAuxMatrixL1Offset = 0;

        if constexpr (PhyPosIsL1OrUB<MM_CFG>((A_TYPE::pos)) && IsStaticPaddingEnable(MM_CFG)) {
            aSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else if constexpr (
            PhyPosIsL1OrUB<MM_CFG>((A_TYPE::pos)) &&
            (IsFullStaticTiling(MM_CFG) || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape())) {
            aSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() != -1 ?
                                          MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() :
                                          MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else {
            aSplitParams.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeA() * BASE_MODULE::c0Size_;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>((B_TYPE::pos)) && IsStaticPaddingEnable(MM_CFG)) {
            bSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else if constexpr (
            PhyPosIsL1OrUB<MM_CFG>((B_TYPE::pos)) &&
            (IsFullStaticTiling(MM_CFG) || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape())) {
            bSplitParams.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() != -1 ?
                                          MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() :
                                          MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else {
            bSplitParams.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeB() * BASE_MODULE::c0Size_;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>((B_TYPE::scalePosition)) && IsStaticPaddingEnable(MM_CFG)) {
            bSplitParams.kAuxMatrixL1Len =
                Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK(), GetMxKfactor<SrcScaleBT>());
        } else if constexpr (
            PhyPosIsL1OrUB<MM_CFG>((B_TYPE::scalePosition)) &&
            (IsFullStaticTiling(MM_CFG) || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape())) {
            bSplitParams.kAuxMatrixL1Len = Ceil(
                MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() != -1 ?
                    MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() :
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK(),
                GetMxKfactor<SrcScaleBT>());
        } else {
            bSplitParams.kAuxMatrixL1Len = MATMUL_MODULE(KLoop)->GetTileShapeScaleKb();
        }
    }
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_MX_SCALEB_SCHEDULER_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_MX_SCALEB_SCHEDULER_H__
#endif
