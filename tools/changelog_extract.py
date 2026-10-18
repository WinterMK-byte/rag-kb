# -*- coding: utf-8 -*-
"""从 CHANGELOG 抽出某一个版本的段落 —— 给发版流程当 Release notes。

为什么不让发版时手写 notes：
  手写的 notes 与 `CHANGELOG.md` 必然**两份**，而两份就会不同步。
  发版是唯一一个「用户第一眼看到的就是这段文字」的地方，正是最不能漂的地方。
  ⇒ 让 Release notes **派生**自 CHANGELOG，单一真源。

用法：
  python tools/changelog_extract.py 0.5.0            # 打到 stdout
  python tools/changelog_extract.py 0.5.0 --out notes.md
  python tools/changelog_extract.py --list           # 列出所有可抽的版本
  python tools/changelog_extract.py --selftest
退出码：0 = 抽到了；1 = 没这个版本（★ 发版流程里必须失败，否则会发一份空 notes）
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_HEAD = re.compile(r'(?m)^##\s*\[([^\]]+)\]')


def versions_in(text):
    return [m.group(1).strip() for m in RE_HEAD.finditer(text)]


def extract(text, version):
    """-> 该版本段落（含标题行），或 None。"""
    matches = list(RE_HEAD.finditer(text))
    for i, m in enumerate(matches):
        if m.group(1).strip() != version:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].rstrip()
        # 段落末尾常有一条分隔线 —— 带进 Release notes 里是噪音
        body = re.sub(r'\n-{3,}\s*$', '', body)
        return body
    return None


def _opt(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def _positional(argv):
    """第一个不是选项、也不是选项值的参数（--root X / --out X 的值要跳过）。"""
    values = set()
    for name in ("--root", "--out"):
        if name in argv:
            values.add(argv[argv.index(name) + 1])
    for a in argv:
        if not a.startswith("-") and a not in values:
            return a
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return _selftest()

    root = _opt(argv, "--root", ROOT)
    path = os.path.join(root, "CHANGELOG.md")
    if not os.path.isfile(path):
        print("找不到 CHANGELOG.md：%s" % path)
        return 1
    text = open(path, encoding="utf-8", errors="replace").read()

    if "--list" in argv:
        for v in versions_in(text):
            print(v)
        return 0

    # ★ 必须 strip：Windows 的 `--list` 会输出 CRLF，shell 里 `$(...)` 取到的是 "0.5.0\r"，
    #   于是「明明在 CHANGELOG 里却报没有」。CI 跑在 ubuntu（LF）所以一直没暴露。
    version = (_positional(argv) or "").strip()
    if not version:
        print("用法：python tools/changelog_extract.py <版本> [--out 文件]")
        print("可用版本：%s" % ", ".join(versions_in(text)))
        return 1

    body = extract(text, version)
    if body is None:
        print("❌ CHANGELOG.md 里没有 [%s]" % version, file=sys.stderr)
        print("   可用版本：%s" % ", ".join(versions_in(text)), file=sys.stderr)
        return 1

    out = _opt(argv, "--out")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(body + "\n")
        print("已写出 %s（%d 字）" % (out, len(body)))
    else:
        print(body)
    return 0


def _selftest():
    tmp = os.path.join(tempfile.gettempdir(), "wbb-changelog-extract")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    fails = []

    sample = (
        "# 更新日志\n\n"
        "## [0.5.0] - 2026-10-16\n\n### 新增\n\n- 甲\n\n---\n\n"
        "## [0.4.0] - 2026-10-16\n\n### 新增\n\n- 乙\n\n---\n\n"
        "## [0.1.0] - 2026-10-12\n\n- 丙\n"
    )
    open(os.path.join(tmp, "CHANGELOG.md"), "w", encoding="utf-8").write(sample)

    def run(args):
        return subprocess.run([sys.executable, "-X", "utf8", os.path.abspath(__file__)]
                              + args + ["--root", tmp],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    r = run(["0.5.0"])
    body = r.stdout
    ok = (r.returncode == 0 and "甲" in body and "乙" not in body
          and body.startswith("## [0.5.0]") and not body.rstrip().endswith("---"))
    print("  %s %-26s %s" % ("✅" if ok else "❌", "抽到本版本且不越界", "不应含下一版本内容"))
    if not ok:
        fails.append("抽段")

    r = run(["0.4.0"])
    ok = r.returncode == 0 and "乙" in r.stdout and "丙" not in r.stdout
    print("  %s %-26s %s" % ("✅" if ok else "❌", "中间版本", "非末段也正确截断"))
    if not ok:
        fails.append("中间段")

    r = run(["9.9.9"])
    ok = r.returncode == 1
    print("  %s %-26s %s" % ("✅" if ok else "❌", "不存在的版本 -> 退出码 1", "否则会发一份空 notes"))
    if not ok:
        fails.append("缺版本")

    r = run(["--list"])
    ok = r.returncode == 0 and r.stdout.split() == ["0.5.0", "0.4.0", "0.1.0"]
    print("  %s %-26s %s" % ("✅" if ok else "❌", "--list", "能列出全部版本"))
    if not ok:
        fails.append("list")

    # ★ 回归：带 \r / 空格 的版本参数必须仍然取得到。
    #   现场是 Windows 的 `--list` 输出 CRLF，shell 里 `$(...)` 拿到 "0.5.0\r"
    #   → 报「明明在 CHANGELOG 里却没有」。CI 在 ubuntu（LF）永远发现不了。
    r = run(["0.5.0\r"])
    ok = r.returncode == 0 and "甲" in r.stdout
    print("  %s %-26s %s" % ("✅" if ok else "❌", "参数带 \\r 仍可抽", "Windows CRLF 场景"))
    if not ok:
        fails.append("crlf")

    shutil.rmtree(tmp, ignore_errors=True)
    print("结果：%s" % ("✅ 全过" if not fails else "❌ 失败 %s" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
