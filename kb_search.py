# -*- coding: utf-8 -*-
"""本地知识库向量检索脚本
用法:
    python kb_search.py "问题文本" [--top K]
依赖: numpy, requests, Ollama qwen3-embedding:0.6b 模型已加载

⚠️ 本脚本是**早期单库版**，指向 `vector_store`（旧无版本号目录，已归档）。
   现役检索请用 `kb_search_v4.py`（多路召回+重排）或直接跑 `kb_ask.py` / `查资料.bat`。
"""
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
VSTORE = os.path.join(BASE, 'vector_store')
# ⚠️ 2026-09-27 由 nomic-embed-text（768 维）改为现役 qwen3-embedding:0.6b（1024 维）。
MODEL = 'qwen3-embedding:0.6b'


def load_store():
    meta = json.load(open(os.path.join(VSTORE, 'metadata.json'), 'r', encoding='utf-8'))
    vecs = np.load(os.path.join(VSTORE, 'vectors.npy'))
    return meta, vecs


def embed_query(q):
    r = requests.post('http://localhost:11434/api/embed',
                      json={'model': MODEL, 'input': [q]}, timeout=120)
    r.raise_for_status()
    return np.array(r.json()['embeddings'][0], dtype=np.float32)


def cosine_sim(mat, vec):
    # mat: [N, dim], vec: [dim] → 返回 [N]
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec) + 1e-9
    return np.dot(mat, vec) / norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', help='检索问题')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--max-chars', type=int, default=300, help='片段展示字数')
    args = parser.parse_args()

    meta, vecs = load_store()
    print(f'向量库已加载: {len(meta)} 条, 维度 {vecs.shape[1]}\n')
    t0 = time.time()
    qv = embed_query(args.query)
    print(f'问题向量化耗时 {time.time()-t0:.1f}s\n')

    scores = cosine_sim(vecs, qv)
    idx = np.argsort(-scores)[:args.top]

    print(f'===== Top {args.top} 检索结果 =====\n')
    for rank, i in enumerate(idx, 1):
        m = meta[i]
        rel = m['rel_path']
        text_len = m['text_len']
        print(f'[{rank}] 相似度 {scores[i]:.4f} | 文件: {rel}')
        print(f'    路径: {m["file_path"]} (片段 {m["chunk_index"]})')
        if m['has_text']:
            t = m.get('text', '')
            print(f'    ─ 片段预览 ─')
            print('    ' + t.replace('\n', '\n    ')[:args.max_chars])
        else:
            print(f'    → 该文件仅元数据索引（无提取文本），可打开文件查看')
        print()


if __name__ == '__main__':
    main()
