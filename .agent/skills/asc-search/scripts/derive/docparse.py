# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""文档正文解析：标题、导航链接、npu 标记与中文产品支持列表。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .gate import load_product_map

NPU_MARKER_RE = re.compile(r"<!--\s*npu=\"(?P<marker>[^\"]+)\"\s+(?P<sid>id\d+)\s*-->")
NPU_END_RE = re.compile(r"<!--\s*end\s+(?P<sid>id\d+)\s*-->")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)#]+\.md)(?:#[^)]*)?\)")
H1_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
PRODUCT_ROW_RE = re.compile(r"^ {0,3}(?:[-*+]|\d+[.)])\s+(?P<name>.+?)[：:]\s*(?P<support>不支持|支持)\s*[。.]?\s*$")
PRODUCT_MAP_PATH = Path(__file__).resolve().parents[2] / "config/product-map.yaml"


def _normalize_product_name(name: str) -> str:
    return re.sub(r"\s+", "", name).casefold()


def doc_title(body: str, fallback: str) -> str:
    match = H1_RE.search(body)
    return match.group(1).strip() if match else fallback


def doc_navigation(body: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in MARKDOWN_LINK_RE.finditer(body):
        target = match.group("target").strip()
        if target.endswith(".md") and not target.startswith(("http", "/")):
            seen.setdefault(target, None)
    return list(seen)


def doc_product_support(body: str, product_map: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """从 npu 标记或产品支持章节的已知中文名称推导产品矩阵。

    有标记时沿用标记，但已知名称与标记不符则报错。无标记只读支持章节的列表，
    不扫描其他章节或代码块；同一产品的支持结论不一致时保留 mixed。
    """
    if product_map is None:
        product_map = load_product_map(PRODUCT_MAP_PATH)
    product_names = {
        _normalize_product_name(name): marker
        for marker, names in product_map.get("product_names", {}).items()
        for name in names
    }
    support: dict[str, dict[str, Any]] = {}
    open_markers: dict[str, str] = {}
    support_level: int | None = None
    fence: str | None = None
    for line_number, line in enumerate(body.splitlines(), 1):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            token, suffix = fence_match.groups()
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip():
                fence = None
        plain_text = fence is None and fence_match is None
        heading = HEADING_RE.match(line) if plain_text else None
        if heading:
            level = len(heading[1])
            if support_level is not None and level <= support_level:
                support_level = None
            if heading[2] == "产品支持情况":
                support_level = level

        for match in NPU_MARKER_RE.finditer(line):
            open_markers[match.group("sid")] = match.group("marker")
        line_markers = set(open_markers.values())
        for match in NPU_END_RE.finditer(line):
            open_markers.pop(match.group("sid"), None)

        visible_line = NPU_END_RE.sub("", NPU_MARKER_RE.sub("", line))
        row = PRODUCT_ROW_RE.match(visible_line) if support_level is not None and plain_text else None
        named_marker = product_names.get(_normalize_product_name(row["name"])) if row else None
        if named_marker and line_markers and line_markers != {named_marker}:
            raise ValueError(
                f"product support conflict at line {line_number}: npu={sorted(line_markers)} "
                f"but {row['name']!r} maps to {named_marker}"
            )
        if open_markers:
            if "不支持" in line:
                verdict = "不支持"
            elif "支持" in line:
                verdict = "支持"
            else:
                continue
            markers = list(dict.fromkeys(open_markers.values()))
        elif named_marker:
            verdict = row["support"]
            markers = [named_marker]
        else:
            continue
        for marker in markers:
            support.setdefault(marker, {"support": verdict, "lines": 0})
            if support[marker]["support"] != verdict:
                support[marker]["support"] = "mixed"
            support[marker]["lines"] += 1
    return support
