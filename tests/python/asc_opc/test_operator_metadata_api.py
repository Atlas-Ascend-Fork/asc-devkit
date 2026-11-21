#! /usr/bin/env python3
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
import importlib.util
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import asc_opc_tool.opc_api as opc_api
import asc_opc_tool.op_info_store as op_info_store
from asc_opc_tool.op_info_store import OpPathParse, SubOpInfoStore


class TestOperatorMetadataApi(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="operator_metadata_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.shape = {"inputs": [{"shape": [8]}], "outputs": [{"shape": [8]}]}
        self.modules_before = set(sys.modules)
        self.path_before = sys.path[:]
        self.addCleanup(self.restore_imports)

    def restore_imports(self):
        sys.path[:] = self.path_before
        for name in set(sys.modules) - self.modules_before:
            module = sys.modules.get(name)
            filename = getattr(module, "__file__", "") or ""
            if filename.startswith(str(self.root)):
                del sys.modules[name]

    def load_function(self, content, name="example"):
        path = self.root / "implementation.py"
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        module_name = "metadata_implementation_" + self.root.name
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, name), str(path)

    def query(self, function):
        from asc_opc_tool.op_compilation import OpCompilation

        with mock.patch.object(op_info_store, "load_op_info_store") as load, mock.patch.object(
            OpCompilation, "resolve_operator_implementation", return_value=function
        ):
            result = opc_api.get_operator_metadata("Example", self.shape, "Ascend910B4")
        load.assert_called_once_with("Ascend910B4")
        return result

    def test_literal_values_are_read_without_executing_function(self):
        function, path = self.load_function("""
            def example():
                raise AssertionError("the metadata query must not execute the operator")
                return OpInfo(init_value_list=[None, "0", "-1"], mc2_ctx=["mc2_context_0"])
        """)
        self.assertEqual(self.query(function), ({1: "0", 2: "-1"}, ("mc2_context_0",), path))

    def test_query_interface_is_used_without_parsing_opinfo(self):
        function, path = self.load_function("""
            def get_operator_metadata():
                return {"init_value_list": [None, "0", "-1"], "mc2_ctx": ["ctx"]}
            def example():
                raise AssertionError("must not execute operator")
        """)
        self.assertEqual(self.query(function), ({1: "0", 2: "-1"}, ("ctx",), path))

    def test_query_interface_uses_actual_function_globals_after_unwrap(self):
        function, path = self.load_function("""
            from functools import wraps
            def get_operator_metadata():
                return {"init_value_list": ["7"], "mc2_ctx": []}
            def decorator(function):
                @wraps(function)
                def wrapper(*args, **kwargs):
                    raise AssertionError("must not execute decorator")
                return wrapper
            @decorator
            def original():
                raise AssertionError("must not execute operator")
            example = original
            example.__module__ = "unrelated_jump_module"
        """)
        self.assertEqual(self.query(function), ({0: "7"}, (), path))

    def test_query_interface_has_priority_over_legacy_opinfo_literals(self):
        function, path = self.load_function("""
            def get_operator_metadata():
                return {"init_value_list": ["3"], "mc2_ctx": ["actual_ctx"]}
            def example():
                return OpInfo(init_value_list=["legacy"], mc2_ctx=["legacy_ctx"])
        """)
        self.assertEqual(self.query(function), ({0: "3"}, ("actual_ctx",), path))

    def test_query_interface_errors_do_not_fall_back_to_legacy_literals(self):
        function, _ = self.load_function("""
            def get_operator_metadata():
                raise ValueError("metadata query failed")
            def example():
                return OpInfo(init_value_list=["legacy"], mc2_ctx=[])
        """)
        with self.assertRaisesRegex(ValueError, "metadata query failed"):
            self.query(function)

    def test_non_callable_query_interface_is_not_treated_as_legacy(self):
        function, _ = self.load_function("""
            get_operator_metadata = []
            def example():
                return OpInfo(init_value_list=["legacy"], mc2_ctx=[])
        """)
        with self.assertRaises(ValueError):
            self.query(function)

    def test_invalid_query_results_do_not_fall_back_to_legacy_literals(self):
        results = (
            "None",
            "[]",
            '{"init_value_list": [None]}',
            '{"init_value_list": [0], "mc2_ctx": []}',
            '{"init_value_list": [None], "mc2_ctx": [""]}',
            '{"init_value_list": [None], "mc2_ctx": ()}',
        )
        for result in results:
            with self.subTest(result=result):
                function, _ = self.load_function(
                    "def get_operator_metadata():\n    return " + result + "\n"
                    "def example():\n"
                    '    return OpInfo(init_value_list=["legacy"], mc2_ctx=[])\n'
                )
                with self.assertRaises(ValueError):
                    self.query(function)

    def test_alias_and_decorated_implementation_are_unwrapped(self):
        function, path = self.load_function("""
            from functools import wraps
            def decorator(function):
                @wraps(function)
                def wrapper(*args, **kwargs):
                    raise AssertionError("must not execute decorator")
                return wrapper
            @decorator
            def original():
                return OpInfo(init_value_list=["0"], mc2_ctx=[])
            example = original
        """)
        self.assertEqual(self.query(function), ({0: "0"}, (), path))

    def test_invalid_or_ambiguous_metadata_is_rejected(self):
        bodies = (
            'values = ["0"]; return OpInfo(init_value_list=values, mc2_ctx=[])',
            "return OpInfo(init_value_list=[0], mc2_ctx=[])",
            'return OpInfo(init_value_list=[None], mc2_ctx=[""])',
            "return OpInfo(init_value_list=[None], mc2_ctx=[None])",
            "return OpInfo(init_value_list=[None], mc2_ctx=())",
            "return OpInfo(init_value_list=[None])",
            "return None",
            "OpInfo(init_value_list=[None], mc2_ctx=[]); return OpInfo(init_value_list=[None], mc2_ctx=[])",
        )
        for body in bodies:
            with self.subTest(body=body):
                function, path = self.load_function("def example():\n    " + body + "\n")
                with self.assertRaises(ValueError) as caught:
                    self.query(function)
                self.assertIn("Example", str(caught.exception))
                self.assertIn(path, str(caught.exception))

    def test_missing_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Example.*source"):
            self.query(len)

    def test_vendor_selection_uses_real_module_priority(self):
        from asc_opc_tool import op_manager

        vendors = []
        for name, value in (("metadata_first", "1"), ("metadata_second", "2")):
            vendor = self.root / "vendors" / name
            package = vendor / "op_impl/ai_core/tbe" / (name + "_impl")
            dynamic = package / "dynamic"
            dynamic.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (dynamic / "__init__.py").write_text("", encoding="utf-8")
            (dynamic / "example.py").write_text(
                "def example():\n"
                '    raise AssertionError("must not execute operator")\n'
                '    return OpInfo(init_value_list=["' + value + '"], mc2_ctx=[])\n',
                encoding="utf-8",
            )
            vendors.append(str(vendor))
        with mock.patch.object(op_info_store, "load_op_info_store"), mock.patch.object(
            op_manager, "get_dynamic_compile_static_from_opstore", return_value="true"
        ), mock.patch.object(op_manager, "get_op_impl_switch_from_opstore", return_value=None), mock.patch.multiple(
            OpPathParse(), custom_opp_path_list=[], vendors_opp_path_list=vendors
        ):
            first = opc_api.get_operator_metadata("Example", self.shape, "Ascend910B4")
            OpPathParse().vendors_opp_path_list = list(reversed(vendors))
            second = opc_api.get_operator_metadata("Example", self.shape, "Ascend910B4")
        self.assertEqual(first[0], {0: "1"})
        self.assertEqual(second[0], {0: "2"})
        self.assertTrue(first[2].startswith(vendors[0]))
        self.assertTrue(second[2].startswith(vendors[1]))

    def test_static_inputs_preserve_dynamic_compile_static_resolution(self):
        from asc_opc_tool import op_compilation
        from asc_opc_tool import op_manager
        from asc_opc_tool.op_compilation import OpCompilation

        function, _ = self.load_function("""
            def example():
                return OpInfo(init_value_list=[None], mc2_ctx=[])
        """)
        with mock.patch.object(
            op_manager, "get_dynamic_compile_static_from_opstore", return_value="true"
        ), mock.patch.object(op_manager, "get_op_impl_switch_from_opstore", return_value=None), mock.patch.object(
            op_compilation, "get_single_op_operator", return_value=function
        ) as resolve:
            self.assertIs(OpCompilation.resolve_operator_implementation("Example", self.shape), function)
        resolve.assert_called_once_with("Example", "true", False)
        self.assertNotIn("is_dynamic", self.shape)

    def test_compile_definition_priority_does_not_require_platform_query(self):
        store = SubOpInfoStore()
        custom = {"Example": {"opFile": {"value": "custom_impl"}}}
        vendor = {"Example": {"opFile": {"value": "vendor_impl"}}}
        builtin = {"Example": {"opFile": {"value": "builtin_impl"}}}
        with mock.patch.multiple(
            store,
            op_kernel_info_dict={},
            op_builtin_info_dict=builtin,
            op_custom_info_list=[custom],
            op_vendor_info_list=[vendor],
        ), mock.patch.object(op_info_store.platform_info, "get_soc_spec", return_value="ascend910b") as query:
            for expected in ("custom_impl", "vendor_impl", "builtin_impl"):
                self.assertTrue(store.construct_op_kernel_info("Example"))
                self.assertEqual(store.op_kernel_info_dict["Example"].op_info["opFileName"], expected)
                if expected == "custom_impl":
                    store.op_custom_info_list = []
                elif expected == "vendor_impl":
                    store.op_vendor_info_list = []
            self.assertFalse(store.construct_op_kernel_info("Missing"))
        query.assert_not_called()

    def test_invalid_custom_definition_does_not_hide_valid_vendor(self):
        store = SubOpInfoStore()
        invalid = {"Example": {"attr": {"list": "missing_attr"}}}
        vendor = {"Example": {"opFile": {"value": "vendor_impl"}}}
        with mock.patch.multiple(
            store,
            op_kernel_info_dict={},
            op_builtin_info_dict={},
            op_custom_info_list=[invalid],
            op_vendor_info_list=[vendor],
        ):
            self.assertTrue(store.construct_op_kernel_info("Example"))
            self.assertEqual(store.op_kernel_info_dict["Example"].op_info["opFileName"], "vendor_impl")


if __name__ == "__main__":
    unittest.main()
