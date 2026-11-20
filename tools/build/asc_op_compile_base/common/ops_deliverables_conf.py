#!/usr/bin/python
# -*- coding: utf-8 -*-
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
ops deliverables conf parser

Parse `{ASCEND_OPP_PATH}/conf/ops_deliverables.json`, which is published by the opbase package with CANN
(available since CANN 9.2.0). All consumers of the built-in operator package deliverables should resolve
paths through this parser instead of hard-coding paths/names, with the fallback to legacy hard-coded logic
when the conf file is unavailable or the required entry is not declared.

Contract:
1. Query an entry by its `name`, and use the declared paths in declaration order.
2. Placeholders in paths are replaced by the current environment:
   `{ASCEND_OPP_PATH}` -> opp root directory, `{socversion}` -> short soc version (lower-case),
   `{os}` -> os name (e.g. linux), `{arch}` -> hardware architecture (e.g. aarch64).
3. `*` in a path is a wildcard and is expanded with `glob` in sorted order.
4. Paths that do not exist on disk are NOT filtered here; the caller must check existence before use
   ("declared means effective only if it exists") to be robust against drift between conf and reality.

Return value convention of `get_deliverable_paths`:
- `None`  : conf file missing/invalid, or the entry is not declared. The caller should fall back to
            the legacy hard-coded logic.
- `list`  : declared paths after placeholder replacement (may be empty, meaning declared but not
            applicable in the current environment). The caller should filter by existence.

`get_existing_deliverable_paths` is the compatibility-first wrapper of `get_deliverable_paths`
and the recommended entry for consumers: it additionally filters the resolved paths by disk
existence, and returns None whenever the conf is unavailable, the entry is not declared, or none
of the declared paths exists on disk, so that the caller falls back to the legacy hard-coded
logic in all these cases.

Logging convention: a successful conf hit is silent; diagnostic logs are only emitted on the
abnormal (fallback) cases.

This module is maintained as byte-identical copies for different deployment forms (wheel installed
package and tbe deployed package). Keep all copies in sync on any change, and only use python standard
library here.
"""

import glob
import json
import os
import platform
import threading

# conf file relative path under the opp root directory
CONF_RELATIVE_PATH = os.path.join("conf", "ops_deliverables.json")
ENV_OPP_PATH = "ASCEND_OPP_PATH"
OPP_PATH_DEFAULT = "/usr/local/Ascend/cann/opp"
# size limit of the conf file, oversize conf is treated as no conf for safety
MAX_CONF_FILE_SIZE = 1024 * 1024

PLACEHOLDER_OPP_PATH = "{ASCEND_OPP_PATH}"
PLACEHOLDER_SOC_VERSION = "{socversion}"
PLACEHOLDER_OS = "{os}"
PLACEHOLDER_ARCH = "{arch}"

_LOG_PREFIX = "[ops deliverables conf] "

# cache: {abs conf path: ((mtime, size), deliverables_or_None)}
_CONF_CACHE = {}
_CONF_CACHE_LOCK = threading.Lock()


def _default_log(msg):
    print(_LOG_PREFIX + msg, flush=True)


def get_opp_path(opp_path=None):
    """
    Resolve the opp root directory from the given value, env var or the default install path.
    """
    if opp_path:
        return opp_path
    return os.environ.get(ENV_OPP_PATH) or OPP_PATH_DEFAULT


def get_os_name():
    """
    Os name used in conf paths, e.g. "linux".
    """
    return platform.system().lower()


def get_arch():
    """
    Hardware architecture used in conf paths, e.g. "aarch64", "x86_64".
    """
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def _load_conf(conf_file_path, log):
    """
    Load and cache the conf file. Return the `deliverables` list, or None if the conf is
    missing, oversize or invalid (bad json / bad structure), in which case callers should
    treat the conf as not available.
    """
    cache_key = os.path.abspath(conf_file_path)
    try:
        stat_res = os.stat(conf_file_path)
    except OSError:
        # conf file not exists is a common case (old CANN version), not a warning
        return None
    if not os.path.isfile(conf_file_path):
        return None
    if stat_res.st_size > MAX_CONF_FILE_SIZE:
        log(
            "conf file {} size {} exceeds limit {}, treat as no conf".format(
                conf_file_path, stat_res.st_size, MAX_CONF_FILE_SIZE
            )
        )
        return None

    file_stat = (stat_res.st_mtime, stat_res.st_size)
    with _CONF_CACHE_LOCK:
        cached = _CONF_CACHE.get(cache_key)
        if cached is not None and cached[0] == file_stat:
            return cached[1]
        try:
            with open(conf_file_path, "r", encoding="utf-8") as f:
                conf = json.load(f)
        except (OSError, ValueError) as e:
            log("failed to load conf file {}, treat as no conf, reason: {}".format(conf_file_path, str(e)))
            _CONF_CACHE[cache_key] = (file_stat, None)
            return None
        if not isinstance(conf, dict):
            log("invalid conf file {}: top level is not a dict, treat as no conf".format(conf_file_path))
            _CONF_CACHE[cache_key] = (file_stat, None)
            return None
        deliverables = conf.get("deliverables")
        if not isinstance(deliverables, list):
            log("invalid conf file {}: 'deliverables' is not a list, treat as no conf".format(conf_file_path))
            deliverables = None
        _CONF_CACHE[cache_key] = (file_stat, deliverables)
        return deliverables


def _find_entry_paths(deliverables, name):
    """
    Find paths of the entry with the given name. Return None if the entry is not declared.
    """
    for entry in deliverables:
        if isinstance(entry, dict) and entry.get("name") == name:
            paths = entry.get("paths")
            if paths is not None and not isinstance(paths, list):
                return None
            return paths
    return None


def _resolve_path(raw_path, opp_root, placeholder_values, log):
    """
    Replace placeholders in one raw path. Return the resolved path, or None to skip this
    path (placeholder not resolvable in the current environment).

    :param placeholder_values: (placeholder, value) pairs to substitute; a path containing
                               a placeholder whose value is not available is skipped.
    """
    resolved = raw_path.replace(PLACEHOLDER_OPP_PATH, opp_root)
    for placeholder, value in placeholder_values:
        if placeholder not in resolved:
            continue
        if not value:
            return None
        resolved = resolved.replace(placeholder, str(value))
    if "{" in resolved and "}" in resolved:
        log("path '{}' contains unresolvable placeholder, ignore it".format(raw_path))
        return None
    return resolved


def get_deliverable_paths(
    name, opp_path=None, soc_version=None, os_name=None, arch=None, conf_file_path=None, log_func=None
):
    """
    Query the ordered paths of a deliverable entry by name.

    :param name: entry name in the conf file, e.g. "ophost", "opsrc", "ops-info".
    :param opp_path: opp root directory, default to ASCEND_OPP_PATH env or the default install path.
    :param soc_version: short soc version (lower-case), required by paths containing `{socversion}`;
                        such paths are skipped when it is not provided.
    :param os_name: os name, default to the current platform.
    :param arch: hardware architecture, default to the current platform.
    :param conf_file_path: conf file path, default to `{opp_path}/conf/ops_deliverables.json`.
    :param log_func: callable to receive diagnostic messages, default to print.
    :return: None if the conf is unavailable or the entry is not declared (caller falls back);
             otherwise a list of resolved paths in declaration order (empty list means declared
             but not applicable in the current environment). Callers must filter by existence.
    """
    log = log_func if callable(log_func) else _default_log
    opp_root = get_opp_path(opp_path)
    conf_file = conf_file_path or os.path.join(opp_root, CONF_RELATIVE_PATH)
    deliverables = _load_conf(conf_file, log)
    if deliverables is None:
        return None
    entry_paths = _find_entry_paths(deliverables, name)
    if entry_paths is None:
        log("entry '{}' is not declared in {}, caller should fall back to legacy logic".format(name, conf_file))
        return None

    placeholder_values = (
        (PLACEHOLDER_SOC_VERSION, str(soc_version).lower() if soc_version else None),
        (PLACEHOLDER_OS, os_name or get_os_name()),
        (PLACEHOLDER_ARCH, arch or get_arch()),
    )
    paths = []
    for raw_path in entry_paths:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        resolved = _resolve_path(raw_path, opp_root, placeholder_values, log)
        if resolved is None:
            continue
        if "*" in resolved:
            paths.extend(sorted(glob.glob(resolved)))
        else:
            paths.append(resolved)
    return paths


def get_existing_deliverable_paths(
    name, opp_path=None, soc_version=None, os_name=None, arch=None, conf_file_path=None, log_func=None
):
    """
    Compatibility-first query of the ordered paths of a deliverable entry, filtered by
    disk existence, which is the recommended entry for consumers.

    Return None if the conf is unavailable, the entry is not declared, or none of the
    declared paths exists on disk; the caller should fall back to the legacy hard-coded
    logic in all these cases. Otherwise return the existing paths in declaration order
    (non-empty). A successful conf hit is silent; logs are only emitted on the fallback
    cases.
    """
    log = log_func if callable(log_func) else _default_log
    paths = get_deliverable_paths(
        name,
        opp_path=opp_path,
        soc_version=soc_version,
        os_name=os_name,
        arch=arch,
        conf_file_path=conf_file_path,
        log_func=log_func,
    )
    if paths is None:
        log("deliverables conf unavailable for entry '{}', fall back to legacy logic".format(name))
        return None
    valid_paths = [path for path in paths if os.path.exists(path)]
    if not valid_paths:
        log("no declared path of entry '{}' exists on disk, fall back to legacy logic".format(name))
        return None
    return valid_paths


def clear_conf_cache():
    """
    Clear the conf cache, for test or when the conf file is known to be refreshed.
    """
    with _CONF_CACHE_LOCK:
        _CONF_CACHE.clear()
