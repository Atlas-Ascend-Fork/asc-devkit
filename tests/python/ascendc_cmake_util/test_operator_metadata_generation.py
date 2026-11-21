#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import ast
from pathlib import Path
import sys

import pytest

from asc_op_compile_base.asc_op_compiler import OpInfo


REPO_ROOT = Path(__file__).resolve().parents[3]
UTIL_DIR = REPO_ROOT / "cmake/asc/fwk_modules/util"
sys.path.insert(0, str(UTIL_DIR))

import ascendc_impl_build
import opdesc_parser


def generate_module(tmp_path, monkeypatch, builtin, initializers, contexts, input_name="x"):
    monkeypatch.setenv("BUILD_BUILTIN_OPP", "1" if builtin else "0")
    lines = [
        "[MetadataExample]",
        "opInterface.value=metadata_example",
        "opFile.value=metadata_example",
        "input0.name=" + input_name,
        "input0.paramType=required",
        "input0.dtype=float16",
        "input0.format=ND",
    ]
    for index, value in enumerate(initializers):
        lines.extend(
            [
                "output{}.name=y{}".format(index, index),
                "output{}.paramType=required".format(index),
                "output{}.dtype=float16".format(index),
                "output{}.format=ND".format(index),
            ]
        )
        if value is not None:
            lines.append("output{}.initValue={}".format(index, value))
    if contexts:
        lines.append("mc2.ctx=" + ",".join(contexts))
    config = tmp_path / "ops.ini"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "metadata_example.cpp").write_text("", encoding="utf-8")
    descriptors = opdesc_parser.get_op_desc(str(config), [], [], ascendc_impl_build.AdpBuilder, None)
    builder = descriptors[0]
    builder.write_adapt(str(tmp_path), str(tmp_path / "generated"))
    path = tmp_path / "generated/metadata_example.py"
    return ast.parse(path.read_text(encoding="utf-8")), builder, str(path)


def isolated_generated_namespace(tree, builder, filename):
    # 保留真实生成的查询和 OpInfo 构造，隔离依赖平台、编译器的其余算子函数体。
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    query = next((node for node in functions if node.name == "get_operator_metadata"), None)
    assert query is not None, "generated adapter must expose get_operator_metadata"
    operator = next(node for node in functions if node.name == builder.op_intf)
    body = []
    for statement in operator.body:
        calls = [node for node in ast.walk(statement) if isinstance(node, ast.Call)]
        names = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
        if names.intersection({"get_operator_metadata", "OpInfo"}):
            body.append(statement)
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_operator_metadata"
        for statement in body
        for node in ast.walk(statement)
    ), "OPC adapter must consume the same query as JIT"
    module = ast.Module(body=[query] + body, type_ignores=[])
    namespace = {
        "OpInfo": OpInfo,
        "kernel_name": "metadata_example_test",
        "op_type": builder.op_type,
        "__inputs__": [{}],
        "__outputs__": [{} for _ in builder.output_name],
        "__attrs__": [],
    }
    default_arguments = operator.args.args[-len(operator.args.defaults) :]
    for argument, default in zip(default_arguments, operator.args.defaults):
        namespace[argument.arg] = ast.literal_eval(default)
    namespace.update({name: {"name": name} for name in builder.input_name + builder.output_name})
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("builtin", [False, True], ids=["custom", "builtin"])
@pytest.mark.parametrize(
    "initializers,contexts", [([], []), ([None, None], []), ([None, "0", "-1", "1.5"], ["ctx0", "ctx1"])]
)
def test_generated_query_preserves_original_opinfo_values(tmp_path, monkeypatch, builtin, initializers, contexts):
    tree, builder, filename = generate_module(tmp_path, monkeypatch, builtin, initializers, contexts)
    namespace = isolated_generated_namespace(tree, builder, filename)
    metadata = namespace["get_operator_metadata"]()
    actual = namespace["op_info"]
    assert isinstance(actual, OpInfo)
    assert actual.init_value_list == initializers
    assert actual.mc2_ctx == contexts
    assert metadata == {"init_value_list": initializers, "mc2_ctx": contexts}
    assert actual.init_value_list == builder.output_init_value
    assert actual.mc2_ctx == builder.mc2_ctx
    assert actual.kernel_name == namespace["kernel_name"]
    assert actual.op_type == builder.op_type
    assert actual.inputs is namespace["__inputs__"]
    assert actual.outputs is namespace["__outputs__"]
    assert actual.attrs is namespace["__attrs__"]
    assert actual.impl_mode == namespace["impl_mode"]
    assert actual.origin_inputs == [namespace[name] for name in builder.input_name]
    assert actual.origin_outputs == [namespace[name] for name in builder.output_name]
    assert actual.param_type_dynamic == builder.param_type_dynamic
    assert actual.param_type_list == builder.input_type + builder.output_type
    assert actual.output_shape_depend_on_compute == builder.output_shape_depend_on_compute


@pytest.mark.parametrize("builtin", [False, True], ids=["custom", "builtin"])
def test_generated_query_returns_independent_lists_without_compiling(tmp_path, monkeypatch, builtin):
    tree, builder, filename = generate_module(tmp_path, monkeypatch, builtin, [None, "0"], ["ctx"])
    namespace = isolated_generated_namespace(tree, builder, filename)

    def must_not_compile(*args, **kwargs):
        raise AssertionError("metadata queries must not invoke compilation")

    namespace["compile_op"] = must_not_compile
    namespace["get_code_channel"] = must_not_compile
    query = namespace["get_operator_metadata"]
    first = query()
    first["init_value_list"].append("-1")
    first["mc2_ctx"].clear()
    second = query()
    assert second == {"init_value_list": [None, "0"], "mc2_ctx": ["ctx"]}
    assert namespace["op_info"].init_value_list == [None, "0"]
    assert namespace["op_info"].mc2_ctx == ["ctx"]


@pytest.mark.parametrize("builtin", [False, True], ids=["custom", "builtin"])
def test_metadata_named_input_is_preserved(tmp_path, monkeypatch, builtin):
    tree, builder, filename = generate_module(tmp_path, monkeypatch, builtin, ["0"], ["ctx"], input_name="metadata")
    namespace = isolated_generated_namespace(tree, builder, filename)
    tensor = namespace[builder.input_name[0]]
    assert namespace["op_info"].origin_inputs[0] is tensor
    assert tensor == {"name": "metadata_in__"}
    assert namespace["op_info"].init_value_list == ["0"]
    assert namespace["op_info"].mc2_ctx == ["ctx"]
