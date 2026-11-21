#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import copy
import unittest

from asc_opc_tool.opc_api import build_static_support_info


class TestStaticSupportInfoApi(unittest.TestCase):
    @staticmethod
    def tensor(shape, dtype="float32"):
        return {"shape": shape, "dtype": dtype, "format": "ND"}

    def assert_int64_mode(self, inputs, outputs, expected):
        shape = {"inputs": inputs, "outputs": outputs, "attrs": [], "impl_mode": ""}
        original = copy.deepcopy(shape)
        result = build_static_support_info(shape)
        self.assertIs(result["int64Mode"], expected)
        self.assertEqual(shape, original)

    def test_small_tensor_and_int64_dtype_keep_32_bit_mode(self):
        self.assert_int64_mode([self.tensor([16], "int64")], [self.tensor([16])], False)

    def test_input_element_count_over_int32_limit(self):
        self.assert_int64_mode([self.tensor([2147483648], "uint8")], [], True)

    def test_input_byte_count_boundary(self):
        for elements, expected in ((536870911, False), (536870912, True)):
            with self.subTest(elements=elements):
                self.assert_int64_mode([self.tensor([elements])], [], expected)

    def test_output_byte_count_over_int32_limit(self):
        self.assert_int64_mode([self.tensor([1])], [self.tensor([536870912])], True)

    def test_dynamic_tensor_group_uses_existing_first_descriptor_rule(self):
        small = self.tensor([8])
        large = self.tensor([536870912])
        for group, expected in (([large, small], True), ([small, large], False), ([], False)):
            with self.subTest(group=group):
                self.assert_int64_mode([None, group], [], expected)

    def test_default_debug_config_is_omitted(self):
        support = build_static_support_info({"inputs": [], "outputs": [], "attrs": []})
        self.assertNotIn("op_debug_config", support)

    def test_explicit_empty_debug_config_is_omitted(self):
        support = build_static_support_info({"inputs": [], "outputs": [], "attrs": []}, op_debug_config="")
        self.assertNotIn("op_debug_config", support)

    def test_multiple_debug_options_are_preserved_verbatim(self):
        debug_config = "dump_cce,check_overflow,DumpCce,custom_option=1"
        support = build_static_support_info({"inputs": [], "outputs": [], "attrs": []}, op_debug_config=debug_config)
        self.assertEqual(support["op_debug_config"], debug_config)

    def test_debug_text_is_not_interpreted_as_log_level(self):
        support = build_static_support_info({"inputs": [], "outputs": [], "attrs": []}, "debug")
        self.assertEqual(support["op_debug_config"], "debug")


if __name__ == "__main__":
    unittest.main()
