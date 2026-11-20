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
import argparse
import json
import os
import platform
import sys
import tempfile
import unittest
from unittest import mock

THIS_FILE_NAME = __file__
FILE_PATH = os.path.dirname(os.path.realpath(THIS_FILE_NAME))
TOP_PATH = os.path.join(FILE_PATH, "../../../")
FRAMEWORK_PATH = os.path.join(TOP_PATH, "tools/build/")
sys.path.insert(0, FRAMEWORK_PATH)

import asc_op_compile_base.common.ops_deliverables_conf as ops_deliverables_conf
from asc_op_compile_base.asc_op_compiler import op_tiling
from asc_op_compile_base.asc_op_compiler import get_op_tiling

sys.path.insert(0, os.path.join(TOP_PATH, "cmake/asc/fwk_modules/util/"))
import ascendc_check_optype_duplicate


VALID_CONF = {
    "version": "1.0",
    "placeholders": {"socversion": "short soc", "os": "os name", "arch": "arch name"},
    "deliverables": [
        {
            "name": "ophost",
            "type": "shared_library",
            "paths": [
                "{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/{os}/{arch}/libophost_cv.so",
                "{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/{os}/{arch}/libophost_nn.so",
                "{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/{os}/{arch}/libophost_legacy.so",
            ],
        },
        {
            "name": "ops-info",
            "type": "json",
            "paths": [
                "{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/config/{socversion}/aic-{socversion}-ops-info-nn.json"
            ],
        },
    ],
}


def _conf_to_str(conf=VALID_CONF):
    return json.dumps(conf)


class TestOpsDeliverablesConf(unittest.TestCase):
    def setUp(self):
        ops_deliverables_conf.clear_conf_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.opp_path = self._tmp.name

    def _write_conf(self, content=_conf_to_str(), file_name="ops_deliverables.json"):
        conf_dir = os.path.join(self.opp_path, "conf")
        os.makedirs(conf_dir, exist_ok=True)
        conf_file = os.path.join(conf_dir, file_name)
        with open(conf_file, "w", encoding="utf-8") as f:
            f.write(content)
        return conf_file

    def _get_paths(self, **kwargs):
        args = {"opp_path": self.opp_path, "os_name": "linux", "arch": "aarch64"}
        args.update(kwargs)
        return ops_deliverables_conf.get_deliverable_paths("ophost", **args)

    def test_no_conf_returns_none(self):
        self.assertIsNone(self._get_paths())

    def test_conf_paths_resolved_in_order(self):
        self._write_conf()
        paths = self._get_paths()
        expect_prefix = os.path.join(
            self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "op_host", "lib", "linux", "aarch64"
        )
        self.assertEqual(
            paths,
            [
                os.path.join(expect_prefix, "libophost_cv.so"),
                os.path.join(expect_prefix, "libophost_nn.so"),
                os.path.join(expect_prefix, "libophost_legacy.so"),
            ],
        )

    def test_entry_not_declared_returns_none(self):
        self._write_conf()
        self.assertIsNone(ops_deliverables_conf.get_deliverable_paths("opgraph", opp_path=self.opp_path))

    def test_socversion_path_skipped_without_soc_version(self):
        self._write_conf()
        paths = ops_deliverables_conf.get_deliverable_paths(
            "ops-info", opp_path=self.opp_path, os_name="linux", arch="aarch64"
        )
        self.assertEqual(paths, [])
        paths = ops_deliverables_conf.get_deliverable_paths(
            "ops-info", opp_path=self.opp_path, soc_version="910B", os_name="linux", arch="aarch64"
        )
        self.assertEqual(len(paths), 1)
        self.assertIn("aic-910b-ops-info-nn.json", paths[0])

    def test_invalid_json_treated_as_no_conf(self):
        self._write_conf(content="{invalid json")
        self.assertIsNone(self._get_paths())

    def test_invalid_structure_treated_as_no_conf(self):
        self._write_conf(content=json.dumps({"version": "1.0", "deliverables": "not-a-list"}))
        self.assertIsNone(self._get_paths())

    def test_oversize_conf_treated_as_no_conf(self):
        self._write_conf()
        with mock.patch.object(ops_deliverables_conf, "MAX_CONF_FILE_SIZE", 10):
            self.assertIsNone(self._get_paths())

    def test_default_conf_file_under_opp_path(self):
        # conf_file_path not given: read {opp_path}/conf/ops_deliverables.json by default
        self._write_conf()
        paths = self._get_paths()
        self.assertEqual(len(paths), 3)
        self.assertEqual(
            paths[0],
            os.path.join(
                self.opp_path,
                "built-in",
                "op_impl",
                "ai_core",
                "tbe",
                "op_host",
                "lib",
                "linux",
                "aarch64",
                "libophost_cv.so",
            ),
        )

    def test_get_existing_paths_filtered_by_disk(self):
        lib_dir = os.path.join(
            self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "op_host", "lib", "linux", "aarch64"
        )
        os.makedirs(lib_dir, exist_ok=True)
        cv_lib = os.path.join(lib_dir, "libophost_cv.so")
        with open(cv_lib, "w", encoding="utf-8") as f:
            f.write("")
        # VALID_CONF declares cv/nn/legacy, only cv exists on disk
        self._write_conf()
        paths = ops_deliverables_conf.get_existing_deliverable_paths(
            "ophost", opp_path=self.opp_path, os_name="linux", arch="aarch64"
        )
        self.assertEqual(paths, [cv_lib])

    def test_get_existing_paths_no_conf_returns_none(self):
        logs = []
        ret = ops_deliverables_conf.get_existing_deliverable_paths(
            "ophost", opp_path=self.opp_path, os_name="linux", arch="aarch64", log_func=logs.append
        )
        self.assertIsNone(ret)
        # fallback cases log, so the compat branch is observable
        self.assertTrue(any("fall back" in msg for msg in logs))

    def test_get_existing_paths_all_missing_returns_none(self):
        # compatibility first: entry declared but no declared path exists on disk -> fall back
        self._write_conf()
        logs = []
        ret = ops_deliverables_conf.get_existing_deliverable_paths(
            "ophost", opp_path=self.opp_path, os_name="linux", arch="aarch64", log_func=logs.append
        )
        self.assertIsNone(ret)
        self.assertTrue(any("fall back" in msg for msg in logs))

    def test_get_existing_paths_success_is_silent(self):
        lib_dir = os.path.join(
            self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "op_host", "lib", "linux", "aarch64"
        )
        os.makedirs(lib_dir, exist_ok=True)
        cv_lib = os.path.join(lib_dir, "libophost_cv.so")
        with open(cv_lib, "w", encoding="utf-8") as f:
            f.write("")
        self._write_conf()
        logs = []
        paths = ops_deliverables_conf.get_existing_deliverable_paths(
            "ophost", opp_path=self.opp_path, os_name="linux", arch="aarch64", log_func=logs.append
        )
        self.assertEqual(paths, [cv_lib])
        # a successful conf hit is silent
        self.assertEqual(logs, [])

    def test_wildcard_expanded_sorted(self):
        lib_dir = os.path.join(self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "op_host", "lib")
        os.makedirs(lib_dir, exist_ok=True)
        for so_name in ("libophost_b.so", "libophost_a.so", "libophost_legacy.so"):
            with open(os.path.join(lib_dir, so_name), "w", encoding="utf-8") as f:
                f.write("")
        wildcard_conf = {
            "version": "1.0",
            "deliverables": [
                {
                    "name": "ophost",
                    "type": "shared_library",
                    "paths": ["{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/*"],
                }
            ],
        }
        self._write_conf(content=_conf_to_str(wildcard_conf))
        paths = self._get_paths()
        self.assertEqual(
            [os.path.basename(p) for p in paths], ["libophost_a.so", "libophost_b.so", "libophost_legacy.so"]
        )

    def test_parser_copies_byte_identical(self):
        common_copy = os.path.join(TOP_PATH, "tools/build/asc_op_compile_base/common/ops_deliverables_conf.py")
        adapter_copy = os.path.join(TOP_PATH, "tools/build/asc_op_compile_base/adapter/ops_deliverables_conf.py")
        with open(common_copy, "rb") as f:
            common_bytes = f.read()
        with open(adapter_copy, "rb") as f:
            adapter_bytes = f.read()
        self.assertEqual(common_bytes, adapter_bytes)


class TestLoadOpHostTilingLibConfSwitch(unittest.TestCase):
    def setUp(self):
        ops_deliverables_conf.clear_conf_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.opp_path = self._tmp.name

    def _fake_libs(self):
        # the fallback branch filters libs of the incompatible arch, so the fake lib dir
        # must follow the current machine arch to survive the filter on any runner
        lib_dir = os.path.join(
            self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "op_host", "lib", "linux", platform.machine()
        )
        os.makedirs(lib_dir, exist_ok=True)
        cv_lib = os.path.join(lib_dir, "libophost_cv.so")
        legacy_lib = os.path.join(lib_dir, "libophost_legacy.so")
        for lib in (cv_lib, legacy_lib):
            with open(lib, "w", encoding="utf-8") as f:
                f.write("")
        return cv_lib, legacy_lib

    @mock.patch("asc_op_compile_base.asc_op_compiler.get_op_tiling.ctypes.CDLL")
    def test_conf_branch_loads_declared_paths(self, mock_cdll):
        cv_lib, legacy_lib = self._fake_libs()
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            with mock.patch.object(get_op_tiling, "get_existing_deliverable_paths", return_value=[cv_lib, legacy_lib]):
                get_op_tiling.load_op_host_tiling_lib()
        loaded = [call.args[0] for call in mock_cdll.call_args_list]
        self.assertEqual(loaded, [cv_lib, legacy_lib])

    @mock.patch("asc_op_compile_base.asc_op_compiler.get_op_tiling.ctypes.CDLL")
    def test_conf_branch_keeps_legacy_loaded_last(self, mock_cdll):
        cv_lib, legacy_lib = self._fake_libs()
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            with mock.patch.object(get_op_tiling, "get_existing_deliverable_paths", return_value=[legacy_lib, cv_lib]):
                get_op_tiling.load_op_host_tiling_lib()
        loaded = [call.args[0] for call in mock_cdll.call_args_list]
        self.assertEqual(loaded, [cv_lib, legacy_lib])

    @mock.patch("asc_op_compile_base.asc_op_compiler.get_op_tiling.glob.glob")
    @mock.patch("asc_op_compile_base.asc_op_compiler.get_op_tiling.ctypes.CDLL")
    def test_fallback_branch_when_no_conf(self, mock_cdll, mock_glob):
        cv_lib, legacy_lib = self._fake_libs()
        mock_glob.return_value = [legacy_lib, cv_lib]
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            with mock.patch.object(get_op_tiling, "get_existing_deliverable_paths", return_value=None):
                get_op_tiling.load_op_host_tiling_lib()
        loaded = [call.args[0] for call in mock_cdll.call_args_list]
        # legacy lib is sorted last in the fallback branch
        self.assertEqual(loaded, [cv_lib, legacy_lib])

    @mock.patch("asc_op_compile_base.asc_op_compiler.get_op_tiling.glob.glob")
    @mock.patch("asc_op_compile_base.asc_op_compiler.get_op_tiling.ctypes.CDLL")
    def test_fallback_branch_when_conf_result_empty(self, mock_cdll, mock_glob):
        # compatibility first: entry declared but nothing exists (empty result) also falls back
        cv_lib, legacy_lib = self._fake_libs()
        mock_glob.return_value = [legacy_lib, cv_lib]
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            with mock.patch.object(get_op_tiling, "get_existing_deliverable_paths", return_value=[]):
                get_op_tiling.load_op_host_tiling_lib()
        loaded = [call.args[0] for call in mock_cdll.call_args_list]
        self.assertEqual(loaded, [cv_lib, legacy_lib])


class TestGetBuiltinImplDirsConfSwitch(unittest.TestCase):
    def setUp(self):
        ops_deliverables_conf.clear_conf_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.opp_path = self._tmp.name

    def test_conf_impl_dirs_existing_only(self):
        impl_dir = os.path.join(self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "impl", "ops-nn", "dynamic")
        os.makedirs(impl_dir, exist_ok=True)
        drift_dir = os.path.join(self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "impl", "ops-cv", "dynamic")
        conf = {
            "version": "1.0",
            "deliverables": [{"name": "opsrc", "type": "src_file", "paths": [impl_dir, drift_dir]}],
        }
        conf_file = os.path.join(self.opp_path, "conf", "ops_deliverables.json")
        os.makedirs(os.path.dirname(conf_file), exist_ok=True)
        with open(conf_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(conf))
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            dirs = get_op_tiling._get_builtin_impl_dirs_from_conf()
        self.assertEqual(dirs, {os.path.normpath(os.path.abspath(impl_dir))})

    def test_no_conf_returns_none(self):
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            self.assertIsNone(get_op_tiling._get_builtin_impl_dirs_from_conf())


class TestOpTilingConfSwitch(unittest.TestCase):
    def setUp(self):
        ops_deliverables_conf.clear_conf_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.opp_path = self._tmp.name

    @mock.patch("asc_op_compile_base.asc_op_compiler.op_tiling.get_existing_deliverable_paths")
    def test_ophost_paths_from_conf(self, mock_get):
        lib_dir = os.path.join(
            self.opp_path, "built-in", "op_impl", "ai_core", "tbe", "op_host", "lib", "linux", "aarch64"
        )
        os.makedirs(lib_dir, exist_ok=True)
        cv_lib = os.path.join(lib_dir, "libophost_cv.so")
        with open(cv_lib, "w", encoding="utf-8") as f:
            f.write("")
        mock_get.return_value = [cv_lib]
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            paths = op_tiling._get_ophost_tiling_paths_from_conf()
        self.assertEqual(paths, [cv_lib])

    @mock.patch("asc_op_compile_base.asc_op_compiler.op_tiling.get_existing_deliverable_paths")
    def test_ophost_paths_no_conf_returns_none(self, mock_get):
        mock_get.return_value = None
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            self.assertIsNone(op_tiling._get_ophost_tiling_paths_from_conf())


class TestCheckOptypeDuplicateConfSwitch(unittest.TestCase):
    def setUp(self):
        ops_deliverables_conf.clear_conf_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.opp_path = self._tmp.name

    def _write_ops_info(self, optypes, domain="nn"):
        ops_info = {op_type: {} for op_type in optypes}
        json_file = os.path.join(self.opp_path, "aic-910b-ops-info-{}.json".format(domain))
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ops_info, f)
        return json_file

    def _check(self, ini_optypes):
        args = argparse.Namespace(soc_version="910b")
        with mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": self.opp_path}):
            return ascendc_check_optype_duplicate.check_optype_duplicate(args, ini_optypes)

    def test_conf_branch_reads_declared_json(self):
        json_file = self._write_ops_info(["Add"])
        with mock.patch.object(
            ascendc_check_optype_duplicate, "get_existing_deliverable_paths", return_value=[json_file]
        ):
            with mock.patch.object(ascendc_check_optype_duplicate.glob, "glob") as mock_glob:
                ret = self._check(["Add"])
        self.assertEqual(ret, 2)
        mock_glob.assert_not_called()

    @mock.patch.object(ascendc_check_optype_duplicate, "get_existing_deliverable_paths", return_value=None)
    def test_fallback_when_no_conf(self, mock_get):
        json_file = self._write_ops_info(["Add"])
        with mock.patch.object(ascendc_check_optype_duplicate.glob, "glob", return_value=[json_file]) as mock_glob:
            ret = self._check(["Add"])
        self.assertEqual(ret, 2)
        legacy_pattern = os.path.normpath(mock_glob.call_args[0][0])
        self.assertIn(os.path.join("ai_core", "tbe", "config"), legacy_pattern)

    @mock.patch.object(ascendc_check_optype_duplicate, "get_existing_deliverable_paths", return_value=[])
    def test_fallback_when_conf_result_empty(self, mock_get):
        with mock.patch.object(ascendc_check_optype_duplicate.glob, "glob", return_value=[]) as mock_glob:
            ret = self._check(["Add"])
        self.assertEqual(ret, 0)
        mock_glob.assert_called_once()

    def test_conf_branch_catches_ras_op_when_declared(self):
        # P2 regression: the ops-info declaration must cover the RAS (ra5) domain file; once the
        # opbase list is synced, the conf branch must catch duplicate RAS built-in ops
        nn_json = self._write_ops_info(["Add"])
        ra5_json = self._write_ops_info(["MatmulAbftVerify"], domain="ra5")
        with mock.patch.object(
            ascendc_check_optype_duplicate, "get_existing_deliverable_paths", return_value=[nn_json, ra5_json]
        ):
            with mock.patch.object(ascendc_check_optype_duplicate.glob, "glob") as mock_glob:
                ret = self._check(["MatmulAbftVerify"])
        self.assertEqual(ret, 2)
        mock_glob.assert_not_called()

    @mock.patch.object(ascendc_check_optype_duplicate, "get_existing_deliverable_paths", return_value=None)
    def test_fallback_mixed_install_catches_ras_op(self, mock_get):
        # P2 regression: mixed install without conf must still catch duplicate RAS built-in
        # ops via the legacy config dir scan
        nn_json = self._write_ops_info(["Add"])
        ra5_json = self._write_ops_info(["MatmulAbftVerify"], domain="ra5")
        with mock.patch.object(
            ascendc_check_optype_duplicate.glob, "glob", return_value=[nn_json, ra5_json]
        ) as mock_glob:
            ret = self._check(["MatmulAbftVerify"])
        self.assertEqual(ret, 2)
        mock_glob.assert_called_once()


if __name__ == "__main__":
    unittest.main()
