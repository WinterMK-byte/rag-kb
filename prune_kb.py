# -*- coding: utf-8 -*-
"""从既有向量库派生一个「剔除指定目录」的新向量库（原库完全不动）

为什么用派生而不是重建：
  - 重建要重新跑 8283 个片段的嵌入（需 ollama，慢且吃内存）；
  - 派生只做行裁剪，秒级完成，且原库随时可回退。

用法:
    py -3.12 -X utf8 prune_kb.py --src vector_store_v4 --out vector_store_v5 ^
        --exclude robomaster --exclude skills

说明:
  - --exclude 按「路径中的目录/文件名片段」匹配（不区分大小写），可重复传；
  - metadata.json 与 vectors.npy 行序严格对齐，裁剪时同步过滤；
  - bm25_cache_*.pkl 的缓存 key 含 metadata 指纹，新库不会复用旧缓存，无需迁移；
  - latin_mask.npy 会按新 meta 重新生成。
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = os.path.dirname(os.path.abspath(__file__))


def _latin_ratio(s):
    import re
    s = re.sub(r'\s+', '', s or '')
    if not s:
        return 0.0
    n = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    return n / float(len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='源向量库目录（相对本文件或绝对路径）')
    ap.add_argument('--out', required=True, help='输出向量库目录')
    ap.add_argument('--exclude', action='append', default=[],
                    help='要剔除的路径片段，可重复传')
    ap.add_argument('--dry-run', action='store_true', help='只统计，不写文件')
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(BASE, args.src)
    out = args.out if os.path.isabs(args.out) else os.path.join(BASE, args.out)

    meta_path = os.path.join(src, 'metadata.json')
    vec_path = os.path.join(src, 'vectors.npy')
    print('源库: %s' % src)
    meta = json.load(open(meta_path, encoding='utf-8'))
    vecs = np.load(vec_path).astype(np.float32)
    print('  片段 %d ｜ 向量 %s ｜ 维度 %d' % (len(meta), vecs.shape, vecs.shape[1]))
    if len(meta) != vecs.shape[0]:
        print('!! metadata 与 vectors 行数不一致，已中止（避免错位）')
        return 1

    pats = [p.strip().lower() for p in args.exclude if p.strip()]
    if not pats:
        print('!! 未指定 --exclude，已中止')
        return 1
    print('剔除规则: %s' % pats)

    keep = np.ones(len(meta), dtype=bool)
    dropped = {}
    for i, m in enumerate(meta):
        fp = (m.get('file_path') or '').lower()
        for p in pats:
            if p in fp:
                keep[i] = False
                dropped[p] = dropped.get(p, 0) + 1
                break

    n_keep = int(keep.sum())
    n_drop = len(meta) - n_keep
    print('\n保留 %d 片段 / 剔除 %d 片段' % (n_keep, n_drop))
    for p, c in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print('   -%s : %d 片段' % (p, c))
    if args.dry_run:
        print('\n[dry-run] 未写文件')
        return 0

    new_meta = [m for i, m in enumerate(meta) if keep[i]]
    new_vecs = vecs[keep]
    os.makedirs(out, exist_ok=True)
    json.dump(new_meta, open(os.path.join(out, 'metadata.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    np.save(os.path.join(out, 'vectors.npy'), new_vecs)
    np.save(os.path.join(out, 'latin_mask.npy'),
            np.array([_latin_ratio(m.get('text') or '') >= 0.6 for m in new_meta], dtype=bool))

    cfg = {}
    cpath = os.path.join(src, 'config.json')
    if os.path.exists(cpath):
        cfg = json.load(open(cpath, encoding='utf-8'))
    cfg['count'] = len(new_meta)
    cfg['pruned_from'] = os.path.basename(src)
    cfg['pruned_exclude'] = pats
    cfg['pruned_at'] = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    print('\n已写出: %s' % out)
    for n in sorted(os.listdir(out)):
        print('   %-24s %10d B' % (n, os.path.getsize(os.path.join(out, n))))
    print('\n源库未改动: %s' % src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
