#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version_check.py —— 版本号三处一致门禁

规矩：**tag 说的版本 = VERSION 文件 = CHANGELOG 最新条目**。

为什么值得单独一道门禁
----------------------
三个仓都踩过同一类问题：改了代码没改版本号，或者改了版本号没写 CHANGELOG。
于是「v0.5.0 的发布页，内容其实是 0.4.0」—— 而 Release 是**不可撤销的公开记录**，
说错了没法改口，只能再发一个补丁版，反而更乱。

用法::

    python tools/version_check.py                    # 校验 VERSION 与 CHANGELOG 一致
    python tools/version_check.py --expect 0.1.0     # 再校验它等于某个值（CI tag 校验用）
    python tools/version_check.py --selftest

退出码：0 = 一致；1 = 不一致。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

RE_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
RE_HEAD = re.compile(r"^##\s*\[(\d+\.\d+\.\d+)\]", re.M)


def here() -> Path:
    return Path(__file__).resolve().parent.parent


def read_version(root: Path) -> str | None:
    p = root / "VERSION"
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8").strip()


def latest_changelog(root: Path) -> str | None:
    p = root / "CHANGELOG.md"
    if not p.is_file():
        return None
    m = RE_HEAD.search(p.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else None


def check(root: Path, expect: str | None = None, quiet: bool = False) -> list[str]:
    problems: list[str] = []
    v = read_version(root)
    cl = latest_changelog(root)

    if v is None:
        problems.append("缺少 VERSION 文件")
    elif not RE_VERSION.match(v):
        problems.append(f"VERSION 格式不是 x.y.z：{v!r}")

    if cl is None:
        problems.append("CHANGELOG.md 里没有形如 `## [x.y.z]` 的条目")

    if v and cl and v != cl:
        problems.append(f"VERSION({v}) != CHANGELOG 最新条目({cl})")

    if expect is not None and v != expect:
        problems.append(f"VERSION({v}) != 期望值({expect})")

    if not quiet:
        print(f"VERSION = {v}   CHANGELOG 最新 = {cl}"
              + (f"   期望 = {expect}" if expect else ""))
        if problems:
            print(f"  ❌ {len(problems)} 项不一致")
            for p in problems:
                print(f"    · {p}")
        else:
            print("  ✅ 版本号三处一致")
    return problems


def _selftest() -> int:
    """先证明它抓得到错，再证明它不误报。"""
    fails: list[str] = []

    def case(name, setup, want_hit, expect=None):
        d = Path(tempfile.mkdtemp(prefix="vc_"))
        try:
            setup(d)
            got = check(d, quiet=True, expect=expect)
            ok = bool(got) == want_hit
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name} -> {type(e).__name__}: {e}")
            fails.append(name)
            return
        finally:
            shutil.rmtree(d, ignore_errors=True)
        print(("  [PASS] " if ok else "  [FAIL] ") + name)
        if not ok:
            fails.append(name)

    def w(d, rel, text):
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    print("version_check 自测：")
    case("三处一致 → 放行",
         lambda d: (w(d, "VERSION", "0.1.0"),
                w(d, "CHANGELOG.md", "# 更新日志\n\n## [0.1.0] - 2026-10-19\n")),
         False)
    case("VERSION 缺 → 抓到",
         lambda d: w(d, "CHANGELOG.md", "## [0.1.0]\n"), True)
    case("CHANGELOG 落后一版 → 抓到",
         lambda d: (w(d, "VERSION", "0.2.0"),
                w(d, "CHANGELOG.md", "## [0.1.0]\n")), True)
    case("tag 与 VERSION 不符 → 抓到",
         lambda d: (w(d, "VERSION", "0.1.0"),
                w(d, "CHANGELOG.md", "## [0.1.0]\n")),
         True, expect="0.2.0")
    case("tag 与 VERSION 一致 → 放行",
         lambda d: (w(d, "VERSION", "0.1.0"),
                w(d, "CHANGELOG.md", "## [0.1.0]\n")),
         False, expect="0.1.0")
    if fails:
        print(f"\n自测失败 {len(fails)} 项")
        return 1
    print("\n自测全部通过（5 项）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="版本号三处一致门禁")
    ap.add_argument("--root", default=str(here()))
    ap.add_argument("--expect", default=None, help="期望的版本号（CI 用 tag 传入）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    return 1 if check(Path(a.root), a.expect, a.quiet) else 0


if __name__ == "__main__":
    sys.exit(main())
