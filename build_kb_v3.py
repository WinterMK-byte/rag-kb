# -*- coding: utf-8 -*-
"""结构化分块建库 v3（复用 build_kb_v2 的提取/嵌入/落盘，仅替换分块策略）

v2 → v3 分块改动（2026-09-17，build_kb_v2.py 保持不动可回退）：
  1) 标题/结构感知分块：先按 Markdown 标题（#）或编号标题（"第X章" / "1.2.3 xxx" / "Chapter N"）切小节，
     再把相邻小节合并到接近 CHUNK_SIZE；每个片段前缀注入标题路径（如 "generators.md / Star Graph"），
     便于召回命中定位与答案溯源；
  2) 固定长度分块保留为兜底：识别不出结构（标题少于 2 个）或单个小节超长时，
     仍走 build_kb_v2.chunk_text 的定长 + 重叠切分，保证不漏内容；
  3) 输出到新目录（建议 vector_store_v4），旧库 vector_store 原样保留，随时可切回。

用法（与 build_kb_v2 一致）:
    py -3.12 -X utf8 build_kb_v3.py --root "C:\\Users\\<you>\\Desktop\\Resource" ^
        --root "C:\\Users\\<you>\\Desktop\\Marvis产出\\RAG技术知识库" ^
        --out vector_store_v4 --text-cache "<text_cache 目录>" --max-chunks 60 --save-every 20 ^
        --report <报告.json>
"""
import os
import re
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import build_kb_v2 as b2                                          # noqa: E402

CHUNK_SIZE = b2.CHUNK_SIZE          # 2000
CHUNK_OVERLAP = b2.CHUNK_OVERLAP    # 200

MD_HEAD = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
NUM_HEAD = re.compile(r'^\s*(第[一二三四五六七八九十百零〇0-9]{1,3}[章节讲部分篇]|Chapter\s+\d+|\d{1,2}(?:\.\d{1,2}){0,3}[\.、]?\s+\S)')


def _is_heading(line):
    """返回 (level, title) 或 None。"""
    s = line.rstrip()
    if not s.strip():
        return None
    m = MD_HEAD.match(s)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m2 = NUM_HEAD.match(s)
    if m2 and len(s.strip()) <= 80:
        t = s.strip()
        num = re.match(r'^\s*(\d+(?:\.\d+)*)', t)
        lvl = (num.group(1).count('.') + 1) if num else 1
        return lvl, t
    return None


def split_sections(text):
    """把文本切成 [(标题路径, 小节正文)]；无标题时返回 [('', 全文)]。"""
    stack = []
    secs = []
    cur = []
    for line in text.splitlines():
        h = _is_heading(line)
        if h:
            if cur:
                secs.append((' / '.join(t for _l, t in stack), '\n'.join(cur)))
                cur = []
            lvl, title = h
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title))
        else:
            cur.append(line)
    if cur:
        secs.append((' / '.join(t for _l, t in stack), '\n'.join(cur)))
    return secs


_ORIG_CHUNK = b2.chunk_text           # 原始定长分块，必须在替换前保存（否则 struct_chunks 内部会递归调到自己）


def _flat(s):
    return re.sub(r'\s+', ' ', s).strip()


def struct_chunks(text, max_chunks, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """结构感知分块（标题路径前缀 + 定长兜底）。

    关键点：以「目标块长」驱动合并，而不是每个标题各成一块 ——
    否则技巧类文档（大量 ### 小标题）会被切成几十个几十字的碎块。
    """
    if not text or not text.strip():
        return []
    secs = split_sections(text)
    titles = [t for t, _b in secs if t]
    if len(titles) < 2:                                  # 结构不足 → 定长兜底
        return [_flat(c) for c in _ORIG_CHUNK(text, max_chunks) if _flat(c)]

    chunks, buf, buf_len, buf_head = [], [], 0, None

    def flush():
        nonlocal buf, buf_len, buf_head
        if not buf or len(chunks) >= max_chunks:
            buf, buf_len, buf_head = [], 0, None
            return
        body = ' '.join(buf)                             # 各小节已自带标题路径前缀
        if len(body) <= int(size * 1.5):
            chunks.append(_flat(body))
        else:                                            # 超长 → 定长兜底
            room = max(1, max_chunks - len(chunks))
            for c in _ORIG_CHUNK(body, room):
                c = _flat(c)
                if not c:
                    continue
                chunks.append(c)
                if len(chunks) >= max_chunks:
                    break
        buf, buf_len, buf_head = [], 0, None

    for head, body in secs:
        if len(chunks) >= max_chunks:
            break
        b = _flat(body)
        if not b:
            continue
        piece = (head + '：' + b) if head else b
        if len(piece) > int(size * 1.4):                 # 单小节超长 → 独立定长切
            flush()
            room = max(1, max_chunks - len(chunks))
            for c in _ORIG_CHUNK(piece, room):
                c = _flat(c)
                if c:
                    chunks.append(c)
                if len(chunks) >= max_chunks:
                    break
            continue
        if buf and buf_len + len(piece) + 1 > int(size * 1.2):
            flush()
        if buf_head is None:
            buf_head = head
        buf.append(piece)
        buf_len += len(piece) + 1
        if buf_len >= size:
            flush()
    flush()
    return chunks[:max_chunks]


def main():
    print('=== build_kb_v3：结构化分块（标题路径前缀 + 定长兜底）===')
    b2.chunk_text = struct_chunks          # 替换分块实现，其余流程完全复用 v2
    b2.main()


if __name__ == '__main__':
    main()
