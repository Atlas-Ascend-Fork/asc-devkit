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
"""从所选 Python 算子实现读取明确的初始化与 MC2 声明，不执行算子函数。"""

import ast
import inspect
import os
import textwrap


def read_operator_metadata(op_func, op_type):
    """优先调用所选实现的查询接口，旧算子包兼容读取 OpInfo 字面量。"""
    source = "<unavailable source>"
    try:
        function = inspect.unwrap(op_func)
        filename = inspect.getsourcefile(function)
        if not filename:
            raise ValueError("Python source is unavailable")
        source = os.path.realpath(filename)
        # 跳转模块和装饰器可能改变模块名，使用真实函数的定义环境定位同一份查询。
        if "get_operator_metadata" in function.__globals__:
            query = function.__globals__["get_operator_metadata"]
            if not callable(query):
                raise ValueError("get_operator_metadata must be callable")
            values = query()
            if not isinstance(values, dict):
                raise ValueError("get_operator_metadata must return a metadata object")
        else:
            values = _read_legacy_metadata(function)
        initializers = values.get("init_value_list")
        contexts = values.get("mc2_ctx")
        if not isinstance(initializers, list) or not isinstance(contexts, list):
            raise ValueError("init_value_list and mc2_ctx must both be lists")
        output_initializers = {}
        for index, initializer in enumerate(initializers):
            if initializer is None:
                continue
            if not isinstance(initializer, str):
                raise ValueError("init_value_list entries must be strings or None")
            output_initializers[index] = initializer
        for context in contexts:
            if not isinstance(context, str) or not context.strip():
                raise ValueError("mc2_ctx entries must be non-empty strings")
        return output_initializers, tuple(contexts), source
    except Exception as error:
        raise ValueError("{} Python metadata at {}: {}".format(op_type, source, error)) from error


def _read_legacy_metadata(function):
    """旧包未生成查询接口时，只读取唯一 OpInfo 中明确的字面量声明。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
        if name == "OpInfo":
            calls.append(node)
    if len(calls) != 1:
        raise ValueError("expected one explicit OpInfo call, found {}".format(len(calls)))
    values = {}
    for keyword in calls[0].keywords:
        if keyword.arg in ("init_value_list", "mc2_ctx"):
            try:
                values[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError) as error:
                raise ValueError("{} must be a literal list".format(keyword.arg)) from error
    return values
