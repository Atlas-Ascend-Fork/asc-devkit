#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
opc api
"""

from asc_op_compile_base.common.utils import log as logger
from asc_opc_tool.constant import OpcOptions, OpcCompileMode
from asc_opc_tool.opc_common import get_int64_mode
from asc_opc_tool.single_op_post_compile import SingleOpPostCompile


__all__ = ["build_static_support_info", "compile_op", "get_int64_mode", "get_operator_metadata", "resolve_impl_mode"]


def build_static_support_info(shape_config, op_debug_config=""):
    """复用 OPC 规则生成静态 JSON 的 supportInfo，包括 staticKey 和输入输出描述处理"""
    return SingleOpPostCompile.build_static_support_info(shape_config, op_debug_config)


def get_operator_metadata(op_type, shape_config, soc_version):
    """查询所选 Python 实现的元数据，返回初始化映射、MC2 名称元组及源码绝对路径。"""
    if not isinstance(op_type, str) or not op_type:
        raise ValueError("op_type must be a non-empty string")
    from asc_opc_tool.op_compilation import OpCompilation
    from asc_opc_tool.op_info_store import load_op_info_store
    from asc_opc_tool.operator_metadata import read_operator_metadata

    try:
        load_op_info_store(soc_version)
        op_func = OpCompilation.resolve_operator_implementation(op_type, shape_config)
        return read_operator_metadata(op_func, op_type)
    except Exception as error:
        raise ValueError("cannot read Python operator metadata for {}: {}".format(op_type, error)) from error


def resolve_impl_mode(op_type, shape_config, soc_version):
    """查询所选 Python 实现的默认模式，处理 optional 模式及追加匹配 key 的规则"""
    mode = shape_config.get(OpcOptions.IMPL_MODE)
    if mode is None or mode == "":
        return mode, False

    from asc_opc_tool.op_compilation import OpCompilation
    from asc_opc_tool.op_info_store import load_op_info_store

    try:
        load_op_info_store(soc_version)
        return OpCompilation.resolve_impl_mode_for_shape(op_type, shape_config)
    except Exception as error:
        raise ValueError(f"cannot resolve implementation mode for {op_type}: {error}") from error


def compile_op(op_params, build_options):
    """
    Provides API compile operator for external callers.
    """
    from asc_opc_tool.opc import OpcOptionParser
    from asc_opc_tool.op_compilation import OpCompilation
    from asc_opc_tool.op_info_store import load_op_info_store

    opt_parser = OpcOptionParser()
    res = opt_parser.parse_build_options(build_options)
    if not res:
        logger.error("parse_build_options is failed.")
    opt_parser.set_option(OpcOptions.OP_COMPILE_MODE, OpcCompileMode.SINGLE_OP_DICT_MODE)

    if not opt_parser.check_input_params():
        logger.error("Opc tool compile failed.")
        return False

    load_op_info_store(opt_parser.get_option(OpcOptions.SOC_VERSION))
    opt_parser.set_option(OpcOptions.OP_PARAMS, op_params)
    op_compile = OpCompilation(opt_parser.get_all_options())
    if op_compile.op_compilation():
        logger.info("Opc tool compile success.")
        return True
    else:
        logger.error("Opc tool compile failed.")
        return False
