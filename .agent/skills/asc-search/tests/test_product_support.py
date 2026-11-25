# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Product support from explicit markers and section-scoped Chinese names."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from derive.docparse import doc_product_support  # noqa: E402


class ProductSupportTests(unittest.TestCase):
    def test_markerless_known_products_have_distinct_ids(self):
        result = doc_product_support("""## 产品支持情况
- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：支持
- Atlas 200I/500 A2推理产品：不支持
- Atlas推理系列产品AI Core：支持
- Atlas推理系列产品Vector Core：不支持
- Atlas训练系列产品：不支持
""")
        self.assertEqual(
            result,
            {
                "950": {"support": "支持", "lines": 1},
                "A3": {"support": "不支持", "lines": 1},
                "910b": {"support": "支持", "lines": 1},
                "310b": {"support": "不支持", "lines": 1},
                "310p": {"support": "mixed", "lines": 2},
                "910": {"support": "不支持", "lines": 1},
            },
        )

    def test_removing_start_marker_keeps_support(self):
        result = doc_product_support("""## 产品支持情况
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
""")
        self.assertEqual(result, {"950": {"support": "支持", "lines": 1}})

    def test_marked_and_unmarked_rows_can_coexist_without_double_counting(self):
        result = doc_product_support("""## 产品支持情况
<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
- Atlas A3系列产品：不支持
""")
        self.assertEqual(result, {"950": {"support": "支持", "lines": 1}, "A3": {"support": "不支持", "lines": 1}})

    def test_explicit_markers_keep_legacy_behavior(self):
        self.assertEqual(
            doc_product_support('<!-- npu="custom" id1 -->\n某产品：不支持\n'),
            {"custom": {"support": "不支持", "lines": 1}},
        )

    def test_name_and_marker_conflict_is_reported(self):
        with self.assertRaisesRegex(ValueError, r"product support conflict.*950.*A3"):
            doc_product_support('## 产品支持情况\n<!-- npu="950" id1 -->\n- Atlas A3系列产品：支持\n')

    def test_names_match_whole_labels_with_whitespace_normalization(self):
        result = doc_product_support("""## 产品支持情况
* Ascend 950PR & 950DT 系列产品: 支持
- 新型Ascend 950PR&950DT系列产品：支持
- Atlas A2未知系列产品：支持
- 未知950产品：支持
""")
        self.assertEqual(result, {"950": {"support": "支持", "lines": 1}})

    def test_inline_marker_cannot_hide_a_conflicting_product_name(self):
        for row in (
            '- <!-- npu="950" id1 -->Atlas A3系列产品：支持\n<!-- end id1 -->',
            '<!-- npu="950" id1 -->- Atlas A3系列产品：支持<!-- end id1 -->',
        ):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, r"product support conflict.*950.*A3"):
                    doc_product_support("## 产品支持情况\n" + row)

    def test_fallback_only_reads_support_section_and_not_code_examples(self):
        result = doc_product_support("""# asc_demo
- Ascend 950PR&950DT系列产品：支持
## 产品支持情况
```markdown
## 产品支持情况
- Atlas A3系列产品：支持
```
- Atlas A2系列产品：不支持
### 补充说明
- Atlas训练系列产品：不支持
## 约束说明
- Ascend 950PR&950DT系列产品：支持
## 样例
```markdown
## 产品支持情况
- Ascend 950PR&950DT系列产品：支持
```
""")
        self.assertEqual(result, {"910b": {"support": "不支持", "lines": 1}, "910": {"support": "不支持", "lines": 1}})

    def test_unknown_or_non_verdict_text_does_not_imply_support(self):
        self.assertEqual(
            doc_product_support("""## 产品支持情况
- Atlas A3系列产品：暂不支持
- Ascend 950PR&950DT系列产品：支持情况待确认
- 新产品：支持
"""),
            {},
        )

    def test_indented_markdown_code_is_not_product_support(self):
        self.assertEqual(
            doc_product_support("""## 产品支持情况

    - Atlas A3系列产品：支持

\t- Atlas A2系列产品：支持

    ```markdown
    - Ascend 950PR&950DT系列产品：支持
    ```
"""),
            {},
        )


if __name__ == "__main__":
    unittest.main()
