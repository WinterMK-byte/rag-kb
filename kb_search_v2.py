# -*- coding: utf-8 -*-
"""本地知识库检索脚本 v2（改进版：查询改写 + 混合召回 + LLM 重排）

相对 v1（kb_search.py，已备份为 kb_search.py.bak_20260915）的改进：
  1) 查询改写：商汤 glm-5.2（免费池）把口语问题改写成 2 条检索友好查询 + 关键词表
  2) 混合召回：dense（Ollama qwen3-embedding:0.6b 余弦 Top-N） + BM25 关键词路（jieba 分词）
              两路用 RRF（Reciprocal Rank Fusion, k=60）融合
  3) 重排：glm-5.2 对候选片段逐条相关性打分（0-10），按新分数取 Top-K
  4) 每步均可单独关闭，便于做对照实验：--no-rewrite / --no-bm25 / --no-rerank

用法:
    py -3.12 kb_search_v2.py "问题文本" [--top 5] [--cands 30] [--pool 50]
                             [--no-rewrite] [--no-bm25] [--no-rerank]
                             [--model glm-5.2] [--json] [--max-chars 300]

依赖: numpy, requests, jieba；环境变量 SENSENOVA_API_KEY（商汤免费池）；
      Ollama 已加载 qwen3-embedding:0.6b；向量库同目录 vector_store/
"""
import os, sys, json, argparse, time, pickle, re, math, hashlib
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
VSTORE = os.path.join(BASE, 'vector_store')
# 2026-09-20 22:40 由 nomic-embed-text 换入（768 维 → 1024 维）：
# 中文语义分辨力实测更强，同一 query 的 cos 对比（越低越能区分不同主题）：
#   nomic:  cos(极限, PN结)=0.615  cos(极限, 英语四级)=0.650  → 中文几乎不可分（检索串味的根因）
#   qwen3:  cos(极限, PN结)=0.379  cos(极限, 英语四级)=0.324  → 明显拉开
# ⚠️ 换模型后**必须全量重建**向量库（维度变了，旧向量不兼容）：
#   build_kb_v3.py --root ... --out vector_store_v8   （v7 原样保留可回退）
EMB_MODEL = 'qwen3-embedding:0.6b'
OLLAMA = 'http://localhost:11434/api/embed'

SENSENOVA_BASE = 'https://token.sensenova.cn/v1'
LLM_MODEL = 'glm-5.2'          # 查询改写 / 重排统一使用商汤免费池模型
RRF_K = 60                     # RRF 融合常数
BM25_K1, BM25_B = 1.5, 0.75
BM25_DOC_CHARS = 1000          # 每篇参与 BM25 的正文长度上限（含标题路径）


# ---------------------------------------------------------------- 基础加载
def load_store(vstore=VSTORE):
    meta = json.load(open(os.path.join(vstore, 'metadata.json'), 'r', encoding='utf-8'))
    vecs = np.load(os.path.join(vstore, 'vectors.npy'))
    return meta, vecs


def embed_texts(texts, model=None, timeout=180):
    """向量化。

    model=None → **运行时**取模块变量 EMB_MODEL（2026-09-20 22:40 改）。
    原来是 `model=EMB_MODEL` 的默认参数，Python 在**定义时**就绑死了，
    导致按库自动切换嵌入模型（v7=nomic / v8=qwen3）后参数仍指向旧值。
    """
    model = model or EMB_MODEL
    r = requests.post(OLLAMA, json={'model': model, 'input': list(texts)}, timeout=timeout)
    r.raise_for_status()
    return np.array(r.json()['embeddings'], dtype=np.float32)


def cosine_sim(mat, vec):
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec) + 1e-9
    return np.dot(mat, vec) / norms


# ---------------------------------------------------------------- 稠密召回
def search_dense(vecs, qvecs, topk=50):
    """多查询稠密召回：返回 [(idx, 最高余弦分)]，按分数降序"""
    best = {}
    for qv in qvecs:
        sc = cosine_sim(vecs, qv)
        top = np.argsort(-sc)[:topk]
        for i in top:
            i = int(i)
            if sc[i] > best.get(i, -1):
                best[i] = float(sc[i])
    return sorted(best.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------- BM25 关键词路
_STOP = set('的 了 和 与 及 在 是 有 对 为 中 上 下 一 个 我 你 他 这 那 吗 呢 啊 什么 怎么 如何'.split())


def _tokenize(text):
    try:
        import jieba
        toks = []
        for t in jieba.lcut_for_search(text):
            t = t.strip().lower()
            if not t or t in _STOP:
                continue
            if re.fullmatch(r'[a-z0-9_]{%d,}' % 2, t) or len(t) >= 2:
                toks.append(t)
        return toks
    except Exception:
        # 兜底：中文 2-gram + 英文词
        toks = re.findall(r'[a-zA-Z0-9_]+', text.lower())
        cn = re.findall(r'[\u4e00-\u9fff]+', text)
        for seg in cn:
            toks += [seg[i:i + 2] for i in range(max(len(seg) - 1, 1))]
        return [t for t in toks if t]


class BM25Index:
    def __init__(self, postings, doc_len, avgdl, n_docs):
        self.postings = postings      # term -> {doc_idx: tf}
        self.doc_len = doc_len        # np.ndarray
        self.avgdl = avgdl
        self.n_docs = n_docs

    def search(self, query, topk=50):
        scores = np.zeros(self.n_docs, dtype=np.float32)
        terms = _tokenize(query)
        if not terms:
            return []
        for t in set(terms):
            p = self.postings.get(t)
            if not p:
                continue
            df = len(p)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for d, tf in p.items():
                dl = self.doc_len[d]
                scores[d] += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl))
        top = np.argsort(-scores)[:topk]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def build_bm25(meta, vstore=VSTORE, use_cache=True):
    """构建（或复用缓存）BM25 索引；文档 = rel_path + 正文前 BM25_DOC_CHARS 字"""
    tag = hashlib.sha1(f'{len(meta)}|{os.path.getmtime(os.path.join(vstore, "metadata.json"))}|{BM25_DOC_CHARS}'.encode()).hexdigest()[:12]
    cache = os.path.join(vstore, f'bm25_cache_{tag}.pkl')
    if use_cache and os.path.exists(cache):
        try:
            with open(cache, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    postings, doc_len = {}, np.zeros(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        text = (m.get('rel_path') or '') + ' ' + ((m.get('text') or '')[:BM25_DOC_CHARS])
        toks = _tokenize(text)
        doc_len[i] = len(toks)
        for t in toks:
            postings.setdefault(t, {})
            postings[t][i] = postings[t].get(i, 0) + 1
    idx = BM25Index(postings, doc_len, float(doc_len.mean()) or 1.0, len(meta))
    if use_cache:
        try:
            with open(cache, 'wb') as f:
                pickle.dump(idx, f, protocol=4)
        except Exception:
            pass
    return idx


# ---------------------------------------------------------------- RRF 融合
def rrf_fuse(rank_lists, k=RRF_K, topk=30):
    """rank_lists: [[(idx, score), ...], ...] → [(idx, rrf_score)]"""
    agg = {}
    for lst in rank_lists:
        for rank, (idx, _sc) in enumerate(lst, 1):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:topk]


# ---------------------------------------------------------------- LLM 调用
def _api_key():
    key = os.environ.get('SENSENOVA_API_KEY')
    if not key:
        raise RuntimeError('环境变量 SENSENOVA_API_KEY 未设置，无法调用商汤模型')
    return key


def llm_chat(messages, model=LLM_MODEL, temperature=0.2, max_tokens=1500, timeout=180, retries=4):
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f'{SENSENOVA_BASE}/chat/completions',
                              headers={'Authorization': f'Bearer {_api_key()}', 'Content-Type': 'application/json'},
                              json={'model': model, 'messages': messages,
                                    'temperature': temperature, 'max_tokens': max_tokens},
                              timeout=timeout)
            r.raise_for_status()
            content = (r.json()['choices'][0]['message'].get('content') or '').strip()
            if not content:
                # glm 系推理模型的思考内容会占用 max_tokens，可能挤空正文 → 视为失败重试
                raise RuntimeError('LLM 返回正文为空（推理 token 可能占满 max_tokens）')
            return content
        except Exception as e:      # noqa: PERF203
            last = e
            wait = 8 * (attempt + 1) if '429' in str(e) else 2.0 * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError(f'LLM 调用失败: {last}')


REWRITE_PROMPT = """你是检索查询优化助手。把用户的口语化问题改写成更适合在"本地资料库"中做语义检索与关键词检索的查询。

要求：
1. 输出 2 条改写查询（每条 8-25 字），使用资料中更可能出现的专业术语、学科名词，不要提问句式；
2. 输出 5-8 个关键词（2-6 字的术语或实体名）；
3. 只用资料库中真实可能的术语，不要编造；
4. 严格输出 JSON，格式：{{"queries": ["...", "..."], "keywords": ["...", "..."]}}

用户问题：{q}
只输出 JSON："""


def rewrite_query(query, model=LLM_MODEL):
    txt = llm_chat([{'role': 'user', 'content': REWRITE_PROMPT.format(q=query)}], model=model, max_tokens=1500)
    m = re.search(r'\{.*\}', txt, re.S)
    if not m:
        return {'queries': [], 'keywords': []}
    try:
        d = json.loads(m.group(0))
        return {'queries': [str(x) for x in d.get('queries', []) if str(x).strip()][:3],
                'keywords': [str(x) for x in d.get('keywords', []) if str(x).strip()][:8]}
    except Exception:
        return {'queries': [], 'keywords': []}


RERANK_PROMPT = """你是检索结果相关性评审。下面是用户的检索问题，以及若干候选资料片段（编号 + 文件路径 + 片段内容）。

请对每个片段给出 0-10 的相关性分数，评分标准：
- 9-10：片段直接包含问题的答案（公式、结论、方法原文）；
- 6-8：片段是该主题的讲解/题目，能支撑回答；
- 3-5：同科目但主题不同，只有部分术语相关；
- 0-2：完全无关。
文件名是重要信号：若候选片段所在【文件名】与问题存在明显字面重叠（如问题「高数下册作业答案」、
文件名为《高数下答案.pdf》，即文件名是问题按序出现的子序列），即使片段正文预览不完整或为
扫描/OCR 文本，也应视为强相关证据，按内容相关性给 6 分以上；若文件名直接对应问题所指文件，给 9-10 分。
必须给每个编号都打分，且分数要拉开区分度（允许一位小数），只根据片段内容与文件名判断，不要臆测。

用户问题：{q}

候选片段：
{cands}

严格按如下格式逐行输出（每个编号一行，不要输出其它内容）：
编号:分数
"""


def rerank(query, cands, meta, model=LLM_MODEL, preview_chars=220):
    """cands: [(idx, rrf_score)] → [(idx, llm_score)]（按 llm 分降序）"""
    if not cands:
        return []
    blocks = []
    for n, (idx, _s) in enumerate(cands, 1):
        m = meta[idx]
        body = (m.get('text') or '')[:preview_chars].replace('\n', ' ')
        blocks.append(f'[{n}] 文件：{m["rel_path"]}\n内容：{body}')
    prompt = RERANK_PROMPT.format(q=query, cands='\n\n'.join(blocks))
    txt = llm_chat([{'role': 'user', 'content': prompt}], model=model, max_tokens=3000)
    scores = {}
    for line in txt.splitlines():
        mm = re.match(r'\s*\[?(\d+)\]?\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)', line)
        if mm:
            n, s = int(mm.group(1)), float(mm.group(2))
            if 1 <= n <= len(cands):
                scores[n] = s
    out = []
    for n, (idx, _s) in enumerate(cands, 1):
        out.append((idx, scores.get(n, 0.0)))
    return sorted(out, key=lambda kv: -kv[1])


# ---------------------------------------------------------------- 检索主流程
def retrieve(query, meta=None, vecs=None, top=5, pool=50, cands_n=30, bm25=None,
             use_rewrite=True, use_bm25=True, use_rerank=True, model=LLM_MODEL, log=print):
    if meta is None or vecs is None:
        meta, vecs = load_store()
    t_all = time.time()
    info = {'query': query, 'stages': {}}

    # 1) 查询改写
    queries = [query]
    keywords = []
    t0 = time.time()
    if use_rewrite:
        try:
            rw = rewrite_query(query, model=model)
            queries += rw['queries']
            keywords = rw['keywords']
        except Exception as e:
            log(f'[改写] 失败，回退原查询：{e}')
    info['stages']['rewrite'] = {'seconds': round(time.time() - t0, 2), 'queries': queries, 'keywords': keywords}
    log(f'[改写] {queries}  关键词：{keywords}  ({info["stages"]["rewrite"]["seconds"]}s)')

    # 2) 召回（dense + bm25 → RRF）
    t0 = time.time()
    qvecs = embed_texts(queries)
    dense = search_dense(vecs, qvecs, topk=pool)
    rank_lists = [dense]
    if use_bm25:
        if bm25 is None:
            bm25 = build_bm25(meta)
        bm_terms = ' '.join(keywords) if keywords else query
        bm = bm25.search(bm_terms, topk=pool)
        if not bm:
            bm = bm25.search(query, topk=pool)
        rank_lists.append(bm)
        info['stages']['bm25_hits'] = len(bm)
        log(f'[召回] dense {len(dense)} 条 / BM25 {len(bm)} 条 → RRF 融合')
    else:
        log(f'[召回] dense {len(dense)} 条（未启用 BM25）')
    fused = rrf_fuse(rank_lists, topk=cands_n)
    # 每文件最多保留 2 个 chunk，控制冗余
    seen, cands = {}, []
    for idx, sc in fused:
        fp = meta[idx]['file_path']
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] <= 2:
            cands.append((idx, sc))
    info['stages']['recall'] = {'seconds': round(time.time() - t0, 2), 'candidates': len(cands)}

    # 3) 重排
    t0 = time.time()
    if use_rerank:
        try:
            ranked = rerank(query, cands, meta, model=model)
        except Exception as e:
            log(f'[重排] 失败，回退 RRF 顺序：{e}')
            ranked = cands
    else:
        ranked = cands
    info['stages']['rerank'] = {'seconds': round(time.time() - t0, 2)}

    # 4) 结果（每文件最多 1 条进入最终结果）
    final, used = [], set()
    for idx, sc in ranked:
        fp = meta[idx]['file_path']
        if fp in used:
            continue
        used.add(fp)
        dscore = dict(dense).get(idx, 0.0)
        final.append({'idx': idx, 'rank_score': sc, 'dense_score': dscore,
                      'rel_path': meta[idx]['rel_path'], 'file_path': fp,
                      'chunk_index': meta[idx]['chunk_index'], 'has_text': meta[idx]['has_text'],
                      'text': (meta[idx].get('text') or '')})
        if len(final) >= top:
            break
    info['results'] = final
    info['stages']['total_seconds'] = round(time.time() - t_all, 2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('query', help='检索问题')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--pool', type=int, default=50, help='单路召回条数')
    ap.add_argument('--cands', type=int, default=30, help='进入重排的候选数')
    ap.add_argument('--no-rewrite', action='store_true')
    ap.add_argument('--no-bm25', action='store_true')
    ap.add_argument('--no-rerank', action='store_true')
    ap.add_argument('--model', default=LLM_MODEL)
    ap.add_argument('--max-chars', type=int, default=300)
    ap.add_argument('--json', action='store_true', help='只输出 JSON（供脚本调用）')
    args = ap.parse_args()

    meta, vecs = load_store()
    if not args.json:
        print(f'向量库已加载: {len(meta)} 条, 维度 {vecs.shape[1]}')
    info = retrieve(args.query, meta, vecs, top=args.top, pool=args.pool, cands_n=args.cands,
                    use_rewrite=not args.no_rewrite, use_bm25=not args.no_bm25,
                    use_rerank=not args.no_rerank, model=args.model,
                    log=(lambda *_a, **_k: None) if args.json else print)

    if args.json:
        print(json.dumps({'query': args.query, 'stages': info['stages'],
                          'results': [{k: r[k] for k in ('rank_score', 'dense_score', 'rel_path', 'file_path', 'chunk_index')} for r in info['results']]},
                         ensure_ascii=False))
        return

    print(f'\n===== Top {args.top} 检索结果（改进版）=====\n')
    for rank, r in enumerate(info['results'], 1):
        print(f'[{rank}] 重排分 {r["rank_score"]:.1f} | 向量相似度 {r["dense_score"]:.4f} | 文件: {r["rel_path"]}')
        print(f'    路径: {r["file_path"]} (片段 {r["chunk_index"]})')
        if r['has_text']:
            t = r['text'].replace('\n', '\n    ')
            print('    ─ 片段预览 ─')
            print('    ' + t[:args.max_chars])
        else:
            print('    → 该文件仅元数据索引（无提取文本），可打开文件查看')
        print()
    st = info['stages']
    print(f'耗时：改写 {st["rewrite"]["seconds"]}s | 召回 {st["recall"]["seconds"]}s | '
          f'重排 {st["rerank"]["seconds"]}s | 合计 {st["total_seconds"]}s')


if __name__ == '__main__':
    main()
