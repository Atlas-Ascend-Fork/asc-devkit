# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Platform queries for static kernel specialization."""

from contextlib import contextmanager
from enum import Enum
from threading import RLock

from .ascendc_common_utility import CommonUtility
from .global_storage import global_var_storage


_SOC_LOCK = RLock()


class SuperKernelRestriction(Enum):
    """平台限制原因；OPC 与 JIT 按各自流程处理不支持的情况。"""

    OPERATOR = "operator is blocked from super kernel on this platform"
    LOCAL_MEMORY = "C310 super kernel does not support local memory"
    SIMT = "C310 super kernel does not support SIMT kernels"


def get_super_kernel_unsupported_reason(op_type, local_memory_size, simt_check):
    """复用当前编译平台的 SK 限制；仅在 C310 且无 local memory 时查询 SIMT。"""
    blocked_ops = {"MoeInitRoutingV2", "MoeInitRoutingV3", "MoeInitRoutingQuant"}
    if CommonUtility.is_v220() and op_type in blocked_ops:
        return SuperKernelRestriction.OPERATOR
    if CommonUtility.is_c310():
        if local_memory_size > 0:
            return SuperKernelRestriction.LOCAL_MEMORY
        if simt_check():
            return SuperKernelRestriction.SIMT
    return None


@contextmanager
def compile_soc(short_soc):
    """Bind the explicit SoC for legacy helpers and restore the compiler's previous value."""
    key = "ascendc_short_soc_version"
    with _SOC_LOCK:
        previous = global_var_storage.get_variable(key)
        try:
            global_var_storage.set_variable(key, short_soc)
            yield
        finally:
            global_var_storage.set_variable(key, previous)


def get_chip_version(short_soc):
    """Return the compiler architecture name for an explicit short SoC name."""
    if not isinstance(short_soc, str) or not short_soc:
        raise ValueError("MIX SK metadata requires an explicit short_soc")
    with compile_soc(short_soc):
        return CommonUtility.get_chip_version()


def validate_super_kernel_support(op_type, local_memory_size, object_path, load_symbols, short_soc):
    """Raise ValueError when the final static kernel violates the platform's SK restrictions."""
    from .ascendc_identify_meta_section_info import check_op_type_is_simt

    with compile_soc(short_soc):
        reason = get_super_kernel_unsupported_reason(
            op_type,
            local_memory_size,
            lambda: any(check_op_type_is_simt(object_path, symbol) for symbol in load_symbols),
        )
        if reason is not None:
            raise ValueError(reason.value + "; requires OPC")
