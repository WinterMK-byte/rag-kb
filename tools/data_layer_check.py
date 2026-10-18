#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_layer_check.py —— 「数据层没有被误提交」门禁

为什么要有这个工具
------------------
这个仓库是从一个 4.9 GB 的真实知识库里**抽出方法层**发布的。
抽的动作本身是一次性的，**会不会漏才是长期的**——
哪天有人在仓库根目录跑一次 build 脚本，`vector_store/` 就长出来了，
下一次 `git add .` 的时候，4.9 GB 资料和一堆有版权的原文就进了公开仓库。

.gitignore 挡得住 `git status`，挡不住别人 `git add -f`，
也挡不住 CI 里某个脚本先把索引生成到仓库里再打包。
所以这里用的是**事后的机械检查**，而不是指望提交前记得看一眼。

检查五件事
----------
① 禁入目录   —— models/、temp/、vector_store*/、text_cache*/ 一个都不许有
② 禁入文件   —— 题库正文、评测集备份
③ 模型权重   —— *.safetensors / *.bin / *.gguf / *.onnx / *.pt …
④ 单文件体积  —— 超过 50 MB（GitHub 警告）、超过 100 MB（GitHub 直接拒收）
⑤ 仓库总大小  —— 超过 100 MB 就是告警（本仓库应当在 1 MB 量级）

用法::

    python tools/data_layer_check.py                 # 扫本仓库
    python tools/data_layer_check.py --root <目录>
    python tools/data_layer_check.py --selftest      # 自测判据（5 个用例）

退出码
------
    0 = 干净
    1 = 发现违规
"""

from __future__ import annotations

import argparse
import os
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

WARN_MB = 50.0       # GitHub 开始警告
HARD_MB = 100.0      # GitHub 直接拒收
REPO_MAX_MB = 100.0  # 整个仓库的合理上限（本仓库实测 0.76 MB）

# ① 禁入目录 —— 这些是「本机接线」，体积占数据层的绝大部分
FORBIDDEN_DIRS = ("models", "temp", "_scratch")
FORBIDDEN_DIR_PREFIX = ("vector_store", "text_cache")

# ② 禁入文件
FORBIDDEN_FILES = {
    "questions_math.jsonl",           # 真题 OCR 全文
}

# ③ 模型权重 —— 一律不许以文件的形式进仓库，README 里写 pull 命令就够了
FORBIDDEN_EXT = (".safetensors", ".bin", ".gguf", ".onnx", ".pt", ".pth", ".ckpt", ".npy")

SKIP_DIRS = {".git", "__pycache__", ".github", ".venv", "venv", "node_modules"}


def check(root: Path, quiet: bool = False) -> list[str]:
    """返回违规清单；空表 = 干净"""
    problems: list[str] = []
    total = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = Path(dirpath).relative_to(root)

        # ① 禁入目录
        for d in list(dirnames):
            show = str(rel_dir / d) if str(rel_dir) != "." else d
            if d in FORBIDDEN_DIRS:
                problems.append(f"禁入目录存在：{show}/")
            elif d.startswith(FORBIDDEN_DIR_PREFIX):
                sz = sum(f.stat().st_size for f in (Path(dirpath) / d).rglob("*") if f.is_file())
                problems.append(
                    f"禁入目录存在：{show}/（{sz / 1048576:.1f} MB）"
                    f" ← 索引/缓存是原文资料的派生品，版权同源"
                )

        for fn in filenames:
            p = Path(dirpath) / fn
            rel = p.relative_to(root)
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            total += sz
            mb = sz / 1048576

            # ② 禁入文件
            if fn in FORBIDDEN_FILES:
                problems.append(f"禁入文件存在：{rel}（{mb:.2f} MB）← 原文，版权")
                continue
            if fn.endswith(".bak") or ".bak_" in fn:
                problems.append(f"备份文件不该入库：{rel}")

            # ③ 模型权重
            if fn.lower().endswith(FORBIDDEN_EXT):
                problems.append(
                    f"模型权重不许入库：{rel}（{mb:.1f} MB）← 写 pull 命令，不要传文件"
                )

            # ④ 单文件体积
            if mb > HARD_MB:
                problems.append(f"单文件 {mb:.1f} MB 超过 {HARD_MB} MB → GitHub 会**直接拒收**：{rel}")
            elif mb > WARN_MB:
                problems.append(
                    f"单文件 {mb:.1f} MB 超过 {WARN_MB} MB → GitHub 会警告（走 LFS 或外链）：{rel}"
                )

    # ⑤ 仓库总大小
    tm = total / 1048576
    if tm > REPO_MAX_MB:
        problems.append(
            f"仓库总文件 {tm:.1f} MB 超过上限 {REPO_MAX_MB} MB"
            f" —— 本仓库是「方法层」，应当在 1 MB 量级。多半是数据层被误加进来了。"
        )

    if not quiet:
        print(f"扫描 {root}")
        print(f"  总大小 {tm:.2f} MB")
        if problems:
            print(f"  ❌ 发现 {len(problems)} 项违规")
            for p in problems:
                print(f"    · {p}")
        else:
            print("  ✅ 数据层干净：只有方法层，没有数据层")
    return problems


def _selftest() -> int:
    """★ 门禁工具必须先证明自己**抓得到错**，否则等于没有门禁。"""
    failures: list[str] = []

    def case(name: str, setup, want_hit: bool):
        d = Path(tempfile.mkdtemp(prefix="dlc_"))
        got: list[str] = []
        try:
            setup(d)
            got = check(d, quiet=True)
            ok = bool(got) == want_hit
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}  ->  {type(e).__name__}: {e}")
            failures.append(f"{name}: 抛异常 {e}")
            return
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if ok:
            print(f"  [PASS] {name}")
        else:
            exp, act = ("抓到", "放行") if want_hit else ("放行", "抓到")
            print(f"  [FAIL] {name}  ->  期望{exp}，实际{act}  {got[:2]}")
            failures.append(f"{name}: 期望{exp}实际{act}")

    def mkdir_(d, rel):
        (d / rel).mkdir(parents=True, exist_ok=True)

    print("data_layer_check 自测：")
    case("干净仓库（只有 py 和 md）",
         lambda d: ((d / "kb_ask.py").write_text("# ok\n", encoding="utf-8"),
                    (d / "README.md").write_text("# ok\n", encoding="utf-8")),
         False)
    case("vector_store/ 被误加进来",
         lambda d: (mkdir_(d, "vector_store_v11"),
                    (d / "vector_store_v11" / "index.bin").write_bytes(b"x" * 2048)),
         True)
    case("models/ 被误加进来",
         lambda d: (mkdir_(d, "models/bge"),
                    (d / "models" / "bge" / "model.safetensors").write_bytes(b"x" * 2048)),
         True)
    case("题库正文被传上来",
         lambda d: (d / "questions_math.jsonl").write_text('{"k":"v"}\n', encoding="utf-8"),
         True)
    case("评测集 .bak 备份被传上来",
         lambda d: (d / "eval_set.jsonl.bak_20261006").write_text("{}\n", encoding="utf-8"),
         True)

    if failures:
        print(f"\n自测失败 {len(failures)} 项")
        return 1
    print("\n自测全部通过（5 项）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检查数据层没有被误提交")
    ap.add_argument("--root", default=None, help="要扫的根目录（默认本仓库）")
    ap.add_argument("--selftest", action="store_true", help="跑自测")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    return 1 if check(root) else 0


if __name__ == "__main__":
    sys.exit(main())
