#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Contract tests for the static kernel platform helpers."""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from asc_op_compile_base import asc_op_compiler
from asc_op_compile_base.common.platform import set_current_compile_soc_info


class StaticKernelPlatformTests(unittest.TestCase):
    def setUp(self):
        set_current_compile_soc_info("Ascend910B4")

    def test_architecture_query_and_failed_sk_check_restore_soc(self):
        storage = asc_op_compiler.global_var_storage
        key = "ascendc_short_soc_version"
        previous = storage.get_variable(key)
        storage.set_variable(key, "Ascend310P")
        try:
            self.assertEqual(asc_op_compiler.get_chip_version("Ascend950"), "c310")
            self.assertEqual(asc_op_compiler.get_chip_version("Ascend910B"), "c220")
            self.assertEqual(storage.get_variable(key), "Ascend310P")
            with self.assertRaisesRegex(ValueError, "local memory"):
                asc_op_compiler.validate_super_kernel_support("SwiGlu", 1024, "", (), "Ascend950")
            self.assertEqual(storage.get_variable(key), "Ascend310P")
        finally:
            storage.set_variable(key, previous)

    def test_v220_sk_blacklist(self):
        with self.assertRaisesRegex(ValueError, "blocked"):
            asc_op_compiler.validate_super_kernel_support("MoeInitRoutingV3", 0, "", (), "Ascend910B")
        asc_op_compiler.validate_super_kernel_support("SwiGlu", 0, "", (), "Ascend910B")

    def test_architecture_failure_restores_soc(self):
        storage = asc_op_compiler.global_var_storage
        key = "ascendc_short_soc_version"
        previous = storage.get_variable(key)
        with patch.object(asc_op_compiler.CommonUtility, "get_chip_version", side_effect=RuntimeError("query failed")):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                asc_op_compiler.get_chip_version("Ascend950")
        self.assertEqual(storage.get_variable(key), previous)

    def test_architecture_requires_explicit_short_soc(self):
        for value in (None, "", False, 123):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "short_soc"):
                asc_op_compiler.get_chip_version(value)

    def test_c310_simt_checks_final_object_and_load_symbol(self):
        from asc_op_compile_base.asc_op_compiler import ascendc_identify_meta_section_info as meta

        storage = asc_op_compiler.global_var_storage
        previous = storage.get_variable("ascendc_short_soc_version")
        with patch.object(meta, "check_op_type_is_simt", return_value=True) as check:
            with self.assertRaisesRegex(ValueError, "SIMT"):
                asc_op_compiler.validate_super_kernel_support("SwiGlu", 0, "/data/hxj/final.o", ("entry",), "Ascend950")
        check.assert_called_once_with("/data/hxj/final.o", "entry")
        self.assertEqual(storage.get_variable("ascendc_short_soc_version"), previous)

    def test_shared_sk_restrictions_match_platform_rules(self):
        from asc_op_compile_base.asc_op_compiler.static_kernel_platform import (
            SuperKernelRestriction,
            compile_soc,
            get_super_kernel_unsupported_reason,
        )

        cases = (
            ("Ascend910B", "MoeInitRoutingV2", 0, False, SuperKernelRestriction.OPERATOR, 0),
            ("Ascend910_93", "MoeInitRoutingV3", 0, False, SuperKernelRestriction.OPERATOR, 0),
            ("Ascend910B", "MoeInitRoutingQuant", 0, False, SuperKernelRestriction.OPERATOR, 0),
            ("Ascend910B", "SwiGlu", 1024, True, None, 0),
            ("Ascend950", "MoeInitRoutingV3", 0, False, None, 1),
            ("Ascend950", "SwiGlu", 1024, True, SuperKernelRestriction.LOCAL_MEMORY, 0),
            ("Ascend350", "SwiGlu", 0, True, SuperKernelRestriction.SIMT, 1),
            ("Ascend950", "SwiGlu", -1, False, None, 1),
            ("Ascend310P", "MoeInitRoutingV3", 1024, True, None, 0),
        )
        for soc, op_type, local_memory, simt, expected, calls in cases:
            with self.subTest(soc=soc, op_type=op_type, local_memory=local_memory, simt=simt):
                observed = []

                def check_simt():
                    observed.append(True)
                    return simt

                with compile_soc(soc):
                    actual = get_super_kernel_unsupported_reason(op_type, local_memory, check_simt)
                self.assertIs(actual, expected)
                self.assertEqual(len(observed), calls)

    def test_c310_simt_callback_failure_restores_soc(self):
        from asc_op_compile_base.asc_op_compiler.static_kernel_platform import (
            compile_soc,
            get_super_kernel_unsupported_reason,
        )

        storage = asc_op_compiler.global_var_storage
        previous = storage.get_variable("ascendc_short_soc_version")

        def failed_simt_query():
            raise RuntimeError("SIMT query failed")

        with self.assertRaisesRegex(RuntimeError, "SIMT query failed"):
            with compile_soc("Ascend950"):
                get_super_kernel_unsupported_reason("SwiGlu", 0, failed_simt_query)
        self.assertEqual(storage.get_variable("ascendc_short_soc_version"), previous)

    def test_opc_postprocess_preserves_support_field_semantics(self):
        from asc_op_compile_base.asc_op_compiler import ascendc_compile_gen_json as gen_json
        from asc_op_compile_base.asc_op_compiler.ascendc_common_utility import CompileInfo
        from asc_op_compile_base.asc_op_compiler.get_op_tiling import OpInfo, TilingInfo
        from asc_op_compile_base.asc_op_compiler.static_kernel_platform import compile_soc
        from asc_op_compile_base.common.buildcfg import build_config
        from asc_op_compile_base.common.context.op_context import OpContext

        cases = (
            ("Ascend910B", "MoeInitRoutingV2", 0, False, None, None, 0),
            ("Ascend910B", "MoeInitRoutingV3", 0, False, 0, 0, 0),
            ("Ascend910_93", "MoeInitRoutingQuant", 0, False, 1, 1, 0),
            ("Ascend910B", "SwiGlu", 1024, True, 0, 1, 0),
            ("Ascend950", "MoeInitRoutingV3", 0, False, 0, 1, 1),
            ("Ascend950", "SwiGlu", 1024, True, 1, 0, 0),
            ("Ascend350", "SwiGlu", 0, True, None, 0, 1),
            ("Ascend950", "SwiGlu", -1, False, None, 1, 1),
        )
        storage = asc_op_compiler.global_var_storage
        for soc, op_type, local_memory, simt, initial, expected, simt_calls in cases:
            with self.subTest(soc=soc, op_type=op_type, local_memory=local_memory, initial=initial):
                with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as folder:
                    path = Path(folder)
                    kernel_name = "static_test_kernel"
                    metadata_path = path / (kernel_name + ".json")
                    metadata = {} if initial is None else {"supportSuperKernel": initial}
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    (path / (kernel_name + ".o")).write_bytes(b"object contents used for SHA256")
                    compile_info = CompileInfo()
                    compile_info.kernel_name = kernel_name
                    tiling_info = TilingInfo()
                    tiling_info.local_memory_size = local_memory
                    op_info = OpInfo(op_type=op_type, inputs=[], outputs=[], mc2_ctx=[])
                    with ExitStack() as stack:
                        stack.enter_context(compile_soc(soc))
                        stack.enter_context(OpContext())
                        stack.enter_context(build_config(op_debug_config=[]))
                        stack.enter_context(patch.object(storage, "ascendc_enable_super_kernel", False))
                        stack.enter_context(patch.object(storage, "ascendc_tiling_no_register", False))
                        stack.enter_context(
                            patch.object(asc_op_compiler.CommonUtility, "get_kernel_meta_dir", return_value=folder)
                        )
                        stack.enter_context(patch.object(asc_op_compiler.CommonUtility, "print_compile_log"))
                        check_simt = stack.enter_context(
                            patch.object(gen_json, "_get_simt_type_in_staic", return_value=simt)
                        )
                        gen_json._json_post_process(compile_info, op_info, tiling_info, False, False, None)
                    result = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if expected is None:
                        self.assertNotIn("supportSuperKernel", result)
                    else:
                        self.assertEqual(result["supportSuperKernel"], expected)
                    self.assertEqual(check_simt.call_count, simt_calls)

    def test_pure_kernel_uses_compiler_loading_fields(self):
        for core_type, physical_core, magic in (
            ("VectorCore", "AIV", "RT_DEV_BINARY_MAGIC_ELF_AIVEC"),
            ("AiCore", "AIC", "RT_DEV_BINARY_MAGIC_ELF_AICUBE"),
        ):
            with self.subTest(core_type=core_type):
                metadata = asc_op_compiler.build_elf_platform_metadata(core_type, None, False)
                self.assertEqual(metadata["coreType"], core_type)
                self.assertEqual(metadata["core_type"], physical_core)
                self.assertEqual(metadata["magic"], magic)
                self.assertEqual(metadata["intercoreSync"], 0)

    def test_mix_sync_field_does_not_require_ffts(self):
        for short_soc in ("Ascend910B", "Ascend950", "Ascend350", "Ascend960DT", "Ascend960PR", "MC62"):
            with self.subTest(short_soc=short_soc):
                metadata = asc_op_compiler.build_elf_platform_metadata("MIX", "1:2", False, short_soc)
                self.assertEqual(
                    metadata,
                    {"coreType": "MIX", "taskRation": "1:2", "magic": "RT_DEV_BINARY_MAGIC_ELF", "intercoreSync": 1},
                )

    def test_mix_without_dfx_uses_platform_ffts_rule_and_restores_soc(self):
        storage = asc_op_compiler.global_var_storage
        key = "ascendc_short_soc_version"
        previous = storage.get_variable(key)
        storage.set_variable(key, "Ascend310P")
        try:
            for short_soc in ("Ascend910B", "Ascend950", "Ascend960DT", "Ascend960PR", "MC62", "Ascend350"):
                with self.subTest(short_soc=short_soc):
                    metadata = asc_op_compiler.build_elf_platform_metadata("MIX", "0:1", None, short_soc)
                    self.assertEqual(metadata["magic"], "RT_DEV_BINARY_MAGIC_ELF")
                    self.assertEqual(metadata["intercoreSync"], 1)
                    self.assertEqual(storage.get_variable(key), "Ascend310P")
        finally:
            storage.set_variable(key, previous)
