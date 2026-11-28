/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_OPS_H
#endif

/**
 * \file vector_ops.h
 * \brief Aggregate header for Tensor API experimental vector interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_VECTOR_OPS_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_VECTOR_OPS_H

#include "tensor_api/tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/memory/register_load.h"
#include "tensor_api/experimental/arch/vector/memory/register_store.h"
#include "tensor_api/experimental/arch/vector/compute/register_copy.h"
#include "tensor_api/experimental/arch/vector/compute/register_arith.h"
#include "tensor_api/experimental/arch/vector/compute/register_logic.h"
#include "tensor_api/experimental/arch/vector/compute/register_compare.h"
#include "tensor_api/experimental/arch/vector/compute/register_fused.h"
#include "tensor_api/experimental/arch/vector/compute/register_convert.h"
#include "tensor_api/experimental/arch/vector/compute/register_broadcast.h"
#include "tensor_api/experimental/arch/vector/compute/register_permute_select.h"
#include "tensor_api/experimental/arch/vector/compute/register_gather.h"
#include "tensor_api/experimental/arch/vector/compute/register_mask.h"
#include "tensor_api/experimental/arch/vector/compute/register_reduce.h"
#include "tensor_api/experimental/arch/vector/compute/register_histogram.h"
#include "tensor_api/experimental/arch/vector/compute/register_index.h"
#include "tensor_api/experimental/arch/vector/control/register_sync.h"

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_VECTOR_OPS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_OPS_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_OPS_H
#endif
