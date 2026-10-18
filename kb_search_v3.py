# -*- coding: utf-8 -*-
"""本地知识库检索脚本 v3（性能优化版：并行 + 候选下调 + 结果缓存）

设计原则：检索思路与 v2 完全一致（查询改写 + 混合召回 + LLM 重排），只压耗时，不改算法。

v2 → v3 的性能改造（实测 2026-09-15）：
  0) 后端本地化（关键）：商汤免费池限速严重（并发/连续调用立即 429，重试退避 8s+，
     实测单题抖动到 25-40s），改用本地 Ollama qwen2.5:7b（GPU）承担改写与重排，
     零费用、无网络限速；--backend cloud 可切回商汤免费池（易 429，仅备选）
  1) 改写与首轮召回并行：启用改写时，原句 dense + BM25 召回不等改写结果，改写 LLM 在后台同时跑
     （改写默认只在 deep 档 / 显式 --rewrite 时开启：本地小模型改写经 50 题实测为负收益）
  2) 重排分块并行：30 条候选切 5 组并行打分（GPU）；片段预览 220 → 120 字
  3) 两阶段重排：全量并行粗排 + RRF 融合（w1=0.4）→ 仅对前 5 条单块精排再融合（w2=0.8），
     只微调头部顺序、尾部保持 RRF 序，避免精排抖动拉低 Hit@5
  4) 结果缓存：查询改写（磁盘持久化 cache/rewrite_cache.json）、embedding（进程内）二次命中秒回
  5) 失败兜底：某块重排失败/超时按 RRF 序折算分数（最高 6 分），保证结果不空、不阻塞

三档模式（50 题评测集实测结果见 output/本地知识库检索提速与易用化报告.md）：
  --mode fast（默认）：本地 qwen2.5:7b + 30 候选 + 5 路并行两阶段重排，实测约 10.7s
  --mode deep        ：云端 glm-5.2 + 30 候选 + 串行重排（精度对齐 v2，慢且易 429）
  --mode turbo       ：不调任何 LLM（等价 --no-rewrite --no-rerank），本地亚秒~数秒

用法:
    py -3.12 -X utf8 kb_search_v3.py "问题文本" [--mode fast|deep|turbo] [--top 5]
                                      [--cands 12] [--workers 2] [--pool 50]
                                      [--backend local|cloud]
                                      [--no-rewrite] [--no-bm25] [--no-rerank]
                                      [--model 模型名] [--json] [--max-chars 300]

依赖: 与 v2 相同（numpy / requests / jieba / 商汤免费池 SENSENOVA_API_KEY / 本地 Ollama）
"""
import os
import re
import sys
import json
import time
import math
import atexit
import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import numpy as np                                             # noqa: E402
import kb_search_v2 as k2                                      # noqa: E402

# 实测（2026-09-15，商汤免费池）：可用 chat 模型仅 3 个
#   deepseek-v4-flash        2.5 / 5.3 s   ← 最快，快档默认
#   sensenova-6.8-flash-lite 5.5 / 17.6 s  ← 抖动大
#   glm-5.2                  10.8 s + 易 429（v2 默认，深档保留）
FAST_MODEL = 'deepseek-v4-flash'           # 云端快档：改写 + 重排（易 429，仅作备选）
DEEP_MODEL = 'glm-5.2'                     # 云端深档：改写 + 重排（v2 默认）
LLM_TIMEOUT = 25                           # 云端单次调用超时（秒）
LLM_RETRIES = 1

# 本地后端（默认，实测 2026-09-15）：Ollama + GPU，零费用、无网络限速
#   qwen2.5:7b 重排 12 候选分 2 块 → 3.6s（对比云端 429 频发、单次 5-25s 抖动）
OLLAMA_CHAT = 'http://localhost:11434/api/chat'
LOCAL_MODEL = 'qwen2.5:7b'
LOCAL_TIMEOUT = 120
CLOUD_MAX_TOKENS = 1200
LOCAL_MAX_TOKENS = 800

CACHE_DIR = os.path.join(BASE, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
RW_CACHE_PATH = os.path.join(CACHE_DIR, 'rewrite_cache.json')
STATS_PATH = os.path.join(CACHE_DIR, 'v3_stats.jsonl')


# ------------------------------------------------------------------ 缓存层
class JsonCache:
    """极简磁盘 KV 缓存（原子写、条数上限、每 N 次落盘）。"""

    def __init__(self, path, limit=20000):
        self.path = path
        self.limit = limit
        self.lock = threading.Lock()
        try:
            with open(path, encoding='utf-8') as f:
                self.d = json.load(f)
        except Exception:
            self.d = {}
        self._dirty = 0

    def get(self, key):
        return self.d.get(key)

    def set(self, key, val):
        with self.lock:
            self.d[key] = val
            self._dirty += 1
            if self._dirty >= 5:
                self._flush_locked()

    def flush(self):
        with self.lock:
            self._flush_locked()

    def _flush_locked(self):
        try:
            if len(self.d) > self.limit:
                self.d = dict(list(self.d.items())[-self.limit:])
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.d, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = 0
        except Exception:
            pass


_RW_CACHE = JsonCache(RW_CACHE_PATH)
atexit.register(_RW_CACHE.flush)      # 退出前落盘：一次性查询进程也能把改写结果写入缓存
_EMB_MEM = {}
_EMB_LOCK = threading.Lock()


def llm_local(prompt, model=LOCAL_MODEL, timeout=LOCAL_TIMEOUT, num_predict=LOCAL_MAX_TOKENS):
    """本地 Ollama 调用（GPU 加速，无网络限速、零费用）。"""
    r = k2.requests.post(OLLAMA_CHAT, json={
        'model': model, 'stream': False,
        'messages': [{'role': 'user', 'content': prompt}],
        'options': {'temperature': 0.15, 'num_predict': num_predict, 'num_ctx': 8192},
    }, timeout=timeout)
    r.raise_for_status()
    return (r.json().get('message', {}).get('content') or '').strip()


def llm_call(prompt, model, backend='local', retries=None, num_predict=None):
    """统一 LLM 入口：backend=local 走 Ollama，cloud 走商汤免费池。"""
    if backend == 'cloud':
        return k2.llm_chat([{'role': 'user', 'content': prompt}], model=model,
                           max_tokens=num_predict or CLOUD_MAX_TOKENS,
                           timeout=LLM_TIMEOUT,
                           retries=LLM_RETRIES if retries is None else retries)
    return llm_local(prompt, model=model,
                     num_predict=num_predict or LOCAL_MAX_TOKENS)


def _rewrite_once(query, model, backend='local'):
    """单次改写调用（短超时 + 少重试，失败由上层回退原句）。"""
    txt = llm_call(k2.REWRITE_PROMPT.format(q=query), model=model, backend=backend) or ''
    txt = txt.replace('“', '"').replace('”', '"')
    m = re.search(r'\{.*\}', txt, re.S)
    body = m.group(0) if m else None
    if body is None:                                  # 本地小模型偶发截断（缺右括号）
        m2 = re.search(r'\{.*', txt, re.S)
        if m2:
            body = m2.group(0).rstrip().rstrip(',') + '}'
    if not body:
        return {'queries': [], 'keywords': []}
    try:
        d = json.loads(body)
        return {'queries': [str(x) for x in d.get('queries', []) if str(x).strip()][:3],
                'keywords': [str(x) for x in d.get('keywords', []) if str(x).strip()][:8]}
    except Exception:
        return {'queries': [], 'keywords': []}


def rewrite_cached(query, model, backend='local'):
    """查询改写（带磁盘缓存）。返回 (dict, cache_hit)"""
    key = hashlib.md5(('%s|%s|%s' % (backend, model, query)).encode('utf-8')).hexdigest()
    hit = _RW_CACHE.get(key)
    if isinstance(hit, dict):
        return hit, True
    rw = _rewrite_once(query, model, backend=backend)
    if rw.get('queries') or rw.get('keywords'):
        _RW_CACHE.set(key, rw)
    return rw, False


def embed_cached(texts, model=None):
    """向量化（进程内缓存；一次请求批量提交）。

    model=None → 运行时取 k2.EMB_MODEL（2026-09-20 22:40 改，理由同 embed_texts：
    默认参数在定义时绑定，按库切模型会失效）。缓存键含模型名，跨模型不会串。
    """
    model = model or k2.EMB_MODEL
    texts = list(texts)
    keys = [hashlib.md5(('%s|%s' % (model, t)).encode('utf-8')).hexdigest() for t in texts]
    miss = [i for i, k in enumerate(keys) if k not in _EMB_MEM]
    if miss:
        arr = k2.embed_texts([texts[i] for i in miss], model=model)
        with _EMB_LOCK:
            for j, i in enumerate(miss):
                _EMB_MEM[keys[i]] = arr[j]
    return np.array([_EMB_MEM[k] for k in keys], dtype=np.float32)


# ------------------------------------------------------------------ 融合/重排
def _merge_rank(lists):
    """多路 (idx, score) 合并：同 idx 取最高分，按分降序。"""
    best = {}
    for lst in lists or []:
        for idx, sc in lst:
            if sc > best.get(idx, -1e18):
                best[idx] = sc
    return sorted(best.items(), key=lambda kv: -kv[1])


def _build_rerank_prompt(query, cands, meta, preview_chars):
    blocks = []
    for n, (idx, _s) in enumerate(cands, 1):
        m = meta[idx]
        body = (m.get('text') or '')[:preview_chars].replace('\n', ' ')
        blocks.append('[%d] 文件：%s\n内容：%s' % (n, m['rel_path'], body))
    return k2.RERANK_PROMPT.format(q=query, cands='\n\n'.join(blocks))


def _parse_scores(txt, group):
    out = {}
    for line in (txt or '').splitlines():
        mm = re.match(r'\s*\[?(\d+)\]?\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)', line)
        if mm:
            n, s = int(mm.group(1)), float(mm.group(2))
            if 1 <= n <= len(group):
                out[group[n - 1][0]] = s
    return out


def rerank_parallel(query, cands, meta, model=LOCAL_MODEL, preview_chars=120,
                    workers=2, retries=LLM_RETRIES, backend='local', log=print):
    """分块并行重排。某块失败/超时 → 该块按 RRF 序折算分数兜底。"""
    if not cands:
        return []
    if workers <= 1 or len(cands) <= 2:
        try:
            txt = llm_call(_build_rerank_prompt(query, cands, meta, preview_chars),
                           model=model, backend=backend, retries=retries)
            scores = _parse_scores(txt, cands)
        except Exception as e:
            log('[重排] 整体失败，按 RRF 序兜底：%s' % str(e)[:80])
            scores = {}
        return _score_fallback(cands, scores)

    groups = [cands[i::workers] for i in range(workers)]
    groups = [g for g in groups if g]

    def work(g):
        try:
            txt = llm_call(_build_rerank_prompt(query, g, meta, preview_chars),
                           model=model, backend=backend, retries=retries)
            return _parse_scores(txt, g)
        except Exception as e:
            log('[重排] 分块失败（%d 条），按 RRF 序兜底：%s' % (len(g), str(e)[:80]))
            return {}

    with ThreadPoolExecutor(max_workers=len(groups)) as ex:
        score_maps = list(ex.map(work, groups))

    out = []
    for g, sc in zip(groups, score_maps):
        out.extend(_score_fallback(g, sc))
    return sorted(out, key=lambda kv: -kv[1])


def _score_fallback(group, scores):
    """块内按 LLM 分数排序；未被打分的条目给 RRF 序折算的保底分（最高 6 分）。"""
    n = len(group)
    out = []
    for pos, (idx, rrf) in enumerate(group, 1):
        if idx in scores:
            out.append((idx, float(scores[idx])))
        else:
            out.append((idx, round(6.0 * (1 - (pos - 1) / float(n)), 2)))
    return out


def fuse_rank(cands, scored, w=0.4):
    """LLM 重排分与 RRF 分线性融合后重排（w 为 LLM 分权重，0 即纯 RRF）。
    实测纯 LLM 排序会丢掉部分命中，融合后 Hit@5 明显回升。"""
    llm = dict(scored)
    rmax = max((s for _, s in cands), default=1.0) or 1.0
    out = []
    for idx, rrf in cands:
        l = (llm.get(idx) or 0.0) / 10.0
        out.append((idx, round(10.0 * (w * l + (1 - w) * (rrf / rmax)), 3)))
    out.sort(key=lambda kv: -kv[1])
    return out


def rerank_two_stage(query, cands, meta, model=LOCAL_MODEL, backend='local',
                     workers=5, preview1=120, preview2=220, w1=0.4, w2=0.8, head=5, log=print):
    """两阶段重排（50 题实测优于单阶段，且耗时仅多约 3s）：
      stage1：全部候选分块并行打分 → 与 RRF 融合（w1）；
      stage2：只对前 head 条精排（单块、更长预览）→ 再融合（w2）微调前 head 内部顺序，
              head 之后顺序不动 —— 保证 Hit@5 不因精排抖动而下降，只改善 Top1 精度。
    """
    if not cands:
        return []
    if len(cands) <= 2:
        return cands
    s1 = rerank_parallel(query, cands, meta, model=model, preview_chars=preview1,
                         workers=workers, backend=backend, log=log)
    fused1 = fuse_rank(cands, s1, w1)
    if head <= 1 or len(fused1) <= head:
        return fused1
    top, tail = fused1[:head], fused1[head:]
    s2 = rerank_parallel(query, top, meta, model=model, preview_chars=preview2,
                         workers=1, backend=backend, log=log)
    return fuse_rank(top, s2, w2) + tail


# ------------------------------------------------------------------ 检索主流程
def retrieve(query, meta=None, vecs=None, top=5, pool=50, cands_n=30, bm25=None,
             use_rewrite=True, use_bm25=True, use_rerank=True,
             model=LOCAL_MODEL, preview_chars=120, workers=5, backend='local',
             two_stage=True, w_fuse=0.4, head2=5, log=print):
    if meta is None or vecs is None:
        meta, vecs = k2.load_store()
    if use_bm25 and bm25 is None:
        bm25 = k2.build_bm25(meta)

    t_all = time.time()
    info = {'query': query, 'stages': {}}

    # ---- 1) 改写（后台）与首轮召回（原句）并行
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=1)
    fut_rw = ex.submit(rewrite_cached, query, model, backend) if use_rewrite else None

    qv0 = embed_cached([query])
    dense_lists = [k2.search_dense(vecs, qv0, topk=pool)]
    bm_lists = []
    if use_bm25:
        bm0 = bm25.search(query, topk=pool)
        if bm0:
            bm_lists.append(bm0)

    queries, keywords, rw_hit = [query], [], None
    if fut_rw is not None:
        try:
            rw, rw_hit = fut_rw.result()
            queries += [q for q in rw.get('queries', []) if q and q != query]
            keywords = rw.get('keywords', [])
        except Exception as e:
            log('[改写] 失败，回退原查询：%s' % str(e)[:100])
    t_rw = time.time() - t0
    ex.shutdown(wait=False)
    if use_rewrite:
        info['stages']['rewrite'] = {'enabled': True, 'seconds': round(t_rw, 2), 'queries': queries,
                                     'keywords': keywords, 'cache_hit': bool(rw_hit)}
        t_recall_extra = 0.0
    else:
        # 未启用改写：这段墙钟其实是「原句 embed + BM25 首轮召回」，归到 recall，避免误读成改写耗时
        info['stages']['rewrite'] = {'enabled': False, 'seconds': 0.0, 'queries': queries,
                                     'keywords': [], 'cache_hit': None}
        t_recall_extra = t_rw

    # ---- 2) 改写查询补充召回 → RRF 融合
    t0 = time.time()
    if len(queries) > 1:
        dense_lists.append(k2.search_dense(vecs, embed_cached(queries[1:]), topk=pool))
    if use_bm25:
        terms = ' '.join(keywords) if keywords else ''
        if terms:
            bm1 = bm25.search(terms, topk=pool)
            if bm1:
                bm_lists.append(bm1)
    dense = _merge_rank(dense_lists)
    bm = _merge_rank(bm_lists)
    fused = k2.rrf_fuse([dense] + ([bm] if bm else []), topk=max(cands_n * 2, cands_n))

    # 每文件最多 1 个片段进入重排（重排是文件级综合判断，候选少时优先覆盖更多文件）
    seen, cands = set(), []
    for idx, sc in fused:
        fp = meta[idx]['file_path']
        if fp in seen:
            continue
        seen.add(fp)
        cands.append((idx, sc))
        if len(cands) >= cands_n:
            break
    info['stages']['recall'] = {'seconds': round(time.time() - t0 + t_recall_extra, 2),
                                'candidates': len(cands), 'dense': len(dense), 'bm25': len(bm)}

    # ---- 3) 重排
    t0 = time.time()
    if use_rerank and cands:
        if two_stage:
            ranked = rerank_two_stage(query, cands, meta, model=model, backend=backend,
                                      workers=workers, preview1=preview_chars,
                                      w1=w_fuse, head=head2, log=log)
        else:
            ranked = fuse_rank(cands, rerank_parallel(query, cands, meta, model=model,
                                                      preview_chars=preview_chars,
                                                      workers=workers, backend=backend, log=log),
                               w_fuse)
    else:
        ranked = cands
    info['stages']['rerank'] = {'seconds': round(time.time() - t0, 2), 'workers': workers,
                                'two_stage': bool(use_rerank and cands and two_stage)}

    # ---- 4) 结果（每文件 1 条）
    final = []
    for idx, sc in ranked:
        final.append({'idx': idx, 'rank_score': sc, 'rel_path': meta[idx]['rel_path'],
                      'file_path': meta[idx]['file_path'], 'chunk_index': meta[idx]['chunk_index'],
                      'has_text': meta[idx]['has_text'], 'text': (meta[idx].get('text') or '')})
        if len(final) >= top:
            break
    info['results'] = final
    info['stages']['total_seconds'] = round(time.time() - t_all, 2)
    return info


# ------------------------------------------------------------------ CLI
def _log_stats(rec):
    try:
        with open(STATS_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description='本地知识库检索 v3（性能优化版）')
    ap.add_argument('query', help='检索问题')
    ap.add_argument('--mode', default='fast', choices=['fast', 'deep', 'turbo'], help='档位')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--pool', type=int, default=50, help='单路召回条数')
    ap.add_argument('--cands', type=int, default=None, help='进入重排的候选数（默认 fast=30 / deep=30）')
    ap.add_argument('--workers', type=int, default=None, help='重排并行块数（默认 fast=5 / deep=1）')
    ap.add_argument('--preview', type=int, default=None, help='重排片段预览字数（默认 fast=120 / deep=220）')
    ap.add_argument('--model', default=None, help='改写/重排模型（默认 fast=qwen2.5:7b 本地 / deep=glm-5.2 云端）')
    ap.add_argument('--backend', default=None, choices=['local', 'cloud'],
                    help='LLM 后端（默认 fast=local 本地 Ollama / deep=cloud 商汤免费池）')
    ap.add_argument('--one-stage', action='store_true', help='关闭两阶段重排（只用 stage1 + 融合）')
    ap.add_argument('--w', type=float, default=None, help='重排分与 RRF 分的融合权重 w（LLM 分占比，默认 0.4）')
    ap.add_argument('--no-rewrite', action='store_true', help='强制关闭查询改写')
    ap.add_argument('--rewrite', action='store_true',
                    help='强制开启查询改写（fast 档默认关闭：本地小模型改写经 50 题实测为负收益）')
    ap.add_argument('--no-bm25', action='store_true')
    ap.add_argument('--no-rerank', action='store_true')
    ap.add_argument('--max-chars', type=int, default=300)
    ap.add_argument('--json', action='store_true', help='只输出 JSON（供脚本调用）')
    args = ap.parse_args()

    cfg = {'fast': dict(model=LOCAL_MODEL, backend='local', cands=30, workers=5, preview=120,
                        two_stage=True, w=0.4, use_rewrite=False),
           'deep': dict(model=DEEP_MODEL, backend='cloud', cands=30, workers=1, preview=220,
                        two_stage=False, w=1.0, use_rewrite=True),
           'turbo': dict(model=LOCAL_MODEL, backend='local', cands=30, workers=1, preview=120,
                         two_stage=False, w=0.4, use_rewrite=False)}[args.mode]
    model = args.model or cfg['model']
    backend = args.backend or cfg['backend']
    two_stage = cfg['two_stage'] and not args.one_stage
    use_rewrite = (cfg['use_rewrite'] or args.rewrite) and not args.no_rewrite
    w_fuse = args.w if args.w is not None else cfg['w']
    cands = args.cands if args.cands is not None else cfg['cands']
    workers = args.workers if args.workers is not None else cfg['workers']
    preview = args.preview if args.preview is not None else cfg['preview']
    quiet = (lambda *a, **k: None) if args.json else print

    meta, vecs = k2.load_store()
    if not args.json:
        print('向量库已加载: %d 条, 维度 %d' % (len(meta), vecs.shape[1]))
    info = retrieve(args.query, meta, vecs, top=args.top, pool=args.pool, cands_n=cands,
                    use_rewrite=use_rewrite,
                    use_bm25=not args.no_bm25,
                    use_rerank=not args.no_rerank and args.mode != 'turbo',
                    model=model, preview_chars=preview, workers=workers, backend=backend,
                    two_stage=two_stage, w_fuse=w_fuse, log=quiet)
    st = info['stages']
    _log_stats({'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'query': args.query, 'mode': args.mode,
                'model': model, 'cands': cands, 'workers': workers,
                'total': st['total_seconds'], 'rewrite': st['rewrite']['seconds'],
                'rerank': st['rerank']['seconds']})

    if args.json:
        print(json.dumps({'query': args.query, 'mode': args.mode,
                          'stages': st,
                          'results': [{k: r[k] for k in ('rank_score', 'rel_path', 'file_path', 'chunk_index')}
                                      for r in info['results']]}, ensure_ascii=False))
        return

    print('\n===== Top %d 检索结果（v3 · %s 档）=====\n' % (args.top, args.mode))
    for rank, r in enumerate(info['results'], 1):
        print('[%d] 重排分 %.1f | 文件: %s' % (rank, r['rank_score'], r['rel_path']))
        print('    路径: %s (片段 %s)' % (r['file_path'], r['chunk_index']))
        if r['has_text']:
            print('    ─ 片段预览 ─')
            print('    ' + r['text'][:args.max_chars].replace('\n', '\n    '))
        else:
            print('    → 该文件仅元数据索引（无提取文本），可打开文件查看')
        print()
    print('耗时：改写 %.1fs | 召回 %.1fs | 重排 %.1fs | 合计 %.1fs（改写缓存命中：%s）'
          % (st['rewrite']['seconds'], st['recall']['seconds'], st['rerank']['seconds'],
             st['total_seconds'], '是' if st['rewrite'].get('cache_hit') else '否'))


if __name__ == '__main__':
    main()
