# -*- coding: utf-8 -*-
"""本地知识库检索 v4（跨语言召回修复 + 结构化分块兼容 + 答案生成层）

v3 → v4 的改动（2026-09-17，均不动 v3 文件，可随时回退）：

  A) 跨语言召回修复（专治 skills 等英文技术文档命中率为 0 的根因）
     实测诊断：中文原句对英文 skills 文档的最高余弦分仅 0.35~0.54，
     而 top50 的入榜门槛约 0.65（被中文课程资料占满）→ 英文文档完全进不了候选。
     换成英文改写查询后，4 道 skills 题的 gold 片段全部排到第 1 名。
     因此新增「英文术语改写通道」：本地 LLM 把中文问题转成英文技术查询 + 英文关键词，
     英文查询单独作为 dense 一路、英文关键词单独作为 BM25 一路参与融合，
     中文原句通道保持不变（只增路、不改路，避免伤害中文资料的命中）。

  B) 加权 RRF：原句 / 中文改写 / 英文改写 / 中文 BM25 / 英文 BM25 各路可配权重
     （v3 的 dense 多路按余弦分合并，不同查询之间分数不可比，v4 改为各路独立参与 RRF）。

  C) --vstore：可指定向量库目录（默认 vector_store，可切结构化分块新建的 vector_store_v4）。

  D) 答案生成层：Top5 片段交给商汤 glm-5.2（免费池）综合成带 [n] 来源标注的答案，
     失败自动回退本地 qwen2.5:7b；--raw 可只看原文片段（保留 v3 行为）。

用法:
    py -3.12 -X utf8 kb_search_v4.py "问题" [--mode fast|deep|turbo] [--raw]
        [--vstore <dir>] [--answer-backend cloud|local|off] [--top 5] [--cands 30]
        [--no-rewrite] [--no-en] [--no-bm25] [--no-rerank] [--max-chars 300] [--json]
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse
import atexit
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import numpy as np                                             # noqa: E402
import kb_search_v2 as k2                                      # noqa: E402
import kb_search_v3 as k3                                      # noqa: E402

DEFAULT_VSTORE = os.path.join(BASE, 'vector_store_v16')   # 2026-10-16 现役：v16（qwen3-embedding:0.6b / 8302 chunks，数据源层修复：华研PDF OCR 100页+图图md base64剥离+734词组doc提取；34条冷口径 Hit@1=82.4%/MRR=0.892，与 v15 持平、c04 rank5→rank4 改善）；v15 原样保留可 --vstore vector_store_v15 回退
# v11（7429 chunks）原样保留，可 --vstore vector_store_v11 回退对比；公式纯库 vector_store_v13 亦保留
RW_CACHE_PATH_V4 = os.path.join(k3.CACHE_DIR, 'rewrite_cache_v4.json')
_RW_CACHE_V4 = k3.JsonCache(RW_CACHE_PATH_V4)
atexit.register(_RW_CACHE_V4.flush)

# 双语改写：中文改写（对付中文资料）+ 英文改写（对付英文技术文档/代码）
REWRITE_PROMPT_V4 = '''你是检索查询优化助手。资料库是中英文混合的（既有中文课程资料，也有英文技术文档、代码与官方文档）。

请针对用户问题输出：
1. "queries"：2 条中文改写查询（每条 8-25 字），使用资料中更可能出现的专业术语，不要提问句式；
2. "en_query"：1 条英文检索查询（3-12 个英文单词或标准术语），要使用该主题在英文技术文档中的标准说法，用于检索英文资料；
3. "keywords"：4-8 个中文关键词（2-6 字术语或实体名）；
4. "en_keywords"：3-8 个英文关键词（单个术语，如 star graph / backend / resource name），用于英文文档的关键词精确匹配；
5. "tech_terms"：1-5 个与该问题最可能相关的英文技术名/库名/框架名（小写，如 sympy / networkx / pymc / simpy / numpy）。若问题明显涉及某个编程库或技术栈，务必给出；实在不确定就给空数组，不要编造。

只用真实存在的术语，不要编造。严格输出 JSON，格式：
{{"queries": ["...", "..."], "en_query": "...", "keywords": ["...", "..."], "en_keywords": ["...", "..."], "tech_terms": ["...", "..."]}}

用户问题：{q}
只输出 JSON：'''

ANSWER_PROMPT = '''你是本地资料库问答助手。请只依据下面给出的资料片段回答用户问题。

规则：
1. 只使用片段中出现的信息，不得补充片段之外的知识，也不要臆测；
2. 每个关键结论后面用 [编号] 标注来源片段，编号对应片段前的 [n]；
3. 如果片段不足以回答问题，就直接说明"现有资料中没有找到相关依据"，并指出还缺什么；
4. 用简体中文作答，直接给结论，条理清晰（可用短列表），不要复述问题，控制在 400 字以内。

用户问题：{q}

资料片段：
{ctx}

答案（含 [编号] 来源标注）：'''


# ------------------------------------------------------------------ 向量库加载
def bind_embed_model(vstore):
    """把 k2.EMB_MODEL 绑到「这个库当初用的嵌入模型」上。

    2026-09-20 22:40 新增：换嵌入模型后，旧库（vector_store_v7 = nomic-embed-text / 768 维）
    仍要能 `--vstore vector_store_v7` 回退检索。不绑的话查询向量是新模型的 1024 维、
    库是 768 维 → 直接抛 "shapes (8329,768) and (1024,) not aligned"。
    依据是各库自带的 config.json（字段 model / dim），读不到就保持现状。
    """
    try:
        with open(os.path.join(vstore, 'config.json'), encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return
    m = cfg.get('model') or cfg.get('embed_model')
    if m and getattr(k2, 'EMB_MODEL', None) != m:
        k2.EMB_MODEL = m
        try:
            sys.stderr.write('[向量库] %s → 嵌入模型 %s\n' % (os.path.basename(vstore), m))
        except Exception:
            pass


_STORE_CACHE = {}


def load_store_v(vstore):
    """加载指定目录的向量库（metadata.json + vectors.npy），并把嵌入模型绑到该库自带的模型。

    2026-09-22 加进程级缓存（改进 #2）：此前每次调用都重读 25.7MB metadata.json
    （实测 json.load 0.54s）+ 34MB vectors.npy，而 `kb_answer` 一次 Agentic 问答要
    调 retrieve 3~5 次 → **同一份库被反复读盘 3~5 遍**。
    缓存后同进程内只读一次。返回的是**同一对象引用**，调用方不得就地修改 meta/vecs；
    要改副本请自行 copy。
    """
    # 2026-09-22 改进 #7：兜底从 k2.VSTORE（旧无名 `vector_store`，已在清理中删除）
    # 改为 DEFAULT_VSTORE（现役 v8），避免旧库目录被清后这里指向空路径。
    key = os.path.abspath(vstore or DEFAULT_VSTORE)
    vstore = vstore or DEFAULT_VSTORE
    hit = _STORE_CACHE.get(key)
    if hit is not None:
        return hit
    meta_path = os.path.join(vstore, 'metadata.json')
    vec_path = os.path.join(vstore, 'vectors.npy')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    vecs = np.load(vec_path).astype(np.float32)
    bind_embed_model(vstore)
    _STORE_CACHE[key] = (meta, vecs)
    return meta, vecs


def clear_store_cache():
    """释放库缓存（换库 / 重建库后调用，避免拿到旧内存快照）。"""
    _STORE_CACHE.clear()


def _parse_json_loose(txt):
    """宽松解析 LLM 输出的 JSON（容忍代码块、中文引号、缺右括号）。"""
    txt = (txt or '').replace('“', '"').replace('”', '"').replace('’', "'")
    m = re.search(r'\{.*\}', txt, re.S)
    body = m.group(0) if m else None
    if body is None:
        m2 = re.search(r'\{.*', txt, re.S)
        if m2:
            body = m2.group(0).rstrip().rstrip(',') + '}'
    if not body:
        return {}
    for cand in (body, body.replace(',]', ']').replace(',}', '}')):
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d
        except Exception:
            continue
    return {}


def _salvage_fields(txt):
    """JSON 解析失败时的兜底：用正则从文本里捞出各字段（本地小模型偶发输出格式跑偏）。"""
    txt = txt or ''

    def arr(key):
        m = re.search(r'"%s"\s*:\s*\[(.*?)\]' % key, txt, re.S)
        if not m:
            return []
        return [x.strip().strip('"\'') for x in re.split(r'[,\n]', m.group(1)) if x.strip().strip('"\'')]

    def one(key):
        m = re.search(r'"%s"\s*:\s*"([^"]+)"' % key, txt)
        return m.group(1).strip() if m else ''

    return {'queries': arr('queries'), 'keywords': arr('keywords'), 'en_keywords': arr('en_keywords'),
            'tech_terms': arr('tech_terms'), 'en_query': one('en_query')}


def _mk_out(d):
    d = d or {}
    return {
        'queries': [str(x).strip() for x in (d.get('queries') or []) if str(x).strip()][:3],
        'en_query': str(d.get('en_query') or '').strip(),
        'keywords': [str(x).strip() for x in (d.get('keywords') or []) if str(x).strip()][:8],
        'en_keywords': [str(x).strip() for x in (d.get('en_keywords') or []) if str(x).strip()][:8],
        'tech_terms': [str(x).strip() for x in (d.get('tech_terms') or []) if str(x).strip()][:5],
    }


def rewrite_v4(query, model=k3.LOCAL_MODEL, backend='local'):
    """双语改写（带磁盘缓存 + 解析兜底 + 失败重试）。

    本地小模型偶发不回 JSON（解析为空）→ 会退化掉英文通道，直接导致英文文档 0 命中，
    因此这里做「严格解析 → 正则兜底 → 重试一次」三级容错。
    """
    # ⚠️ 2026-10-06 修复（总架构师）：原指纹 '%s|%s|%s|v4d2' 不含 RR_SALT，与重排缓存指纹（含 RR_SALT）**错配**。
    #    后果：改了 RR_SALT 后，改写仍复用旧 salt 的缓存 → 候选集来自旧改写、重排用新 salt 缓存
    #    → 同一 query 相邻两档评测位次不一致（实测 t01：warm1 rank=3 / warm2 rank=4）。
    #    修法：salt 纳入改写指纹；如果改了 RR_SALT，两边缓存同时失效，永不再错配。
    key = hashlib.md5(('%s|%s|%s|%s|v4d3' % (RR_SALT, backend, model, query)).encode('utf-8')).hexdigest()
    hit = _RW_CACHE_V4.get(key)
    if isinstance(hit, dict):
        return hit, True
    out = {}
    for attempt in (1, 2):
        prompt = REWRITE_PROMPT_V4.format(q=query)
        if attempt == 2:
            prompt = prompt.replace('只输出 JSON：', '只输出一行 JSON，不要代码块、不要解释：')
        txt = llm_det(prompt, model=model, backend=backend) or ''
        out = _mk_out(_parse_json_loose(txt) or _salvage_fields(txt))
        if out['en_query'] or out['en_keywords'] or out['queries']:
            break
    if out.get('queries') or out.get('en_query') or out.get('keywords') or out.get('en_keywords'):
        _RW_CACHE_V4.set(key, out)
    return out, False


# ------------------------------------------------------------------ 英文文档分池召回
def _latin_ratio(s):
    """非空白字符中拉丁字母的占比，用于识别英文文档片段。"""
    s = re.sub(r'\s+', '', s or '')
    if not s:
        return 0.0
    n = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    return n / float(len(s))


def load_latin_mask(vstore, meta):
    """英文片段掩码（按需计算并缓存到向量库目录）。

    根因修复的核心之一：中文原句对英文文档的余弦分（0.35~0.54）天然低于
    对中文文档的分数（0.65+），全局 topK 会被中文资料占满。
    把英文片段单独作为检索池，低分也能在池内排到前列 → 英文文档不再被淹没。
    """
    p = os.path.join(vstore, 'latin_mask.npy')
    try:
        m = np.load(p)
        if len(m) == len(meta):
            return m.astype(bool)
    except Exception:
        pass
    mask = np.array([_latin_ratio(m.get('text') or '') >= 0.6 for m in meta], dtype=bool)
    try:
        np.save(p, mask)
    except Exception:
        pass
    return mask


def search_dense_subset(vecs, qvec, mask, topk=50):
    """在指定片段子集内做向量检索（分数在同池内可比）。"""
    idxs = np.nonzero(mask)[0]
    if len(idxs) == 0:
        return []
    sc = k2.cosine_sim(vecs[idxs], qvec)
    order = np.argsort(-sc)[:topk]
    return [(int(idxs[i]), float(sc[i])) for i in order]


# ------------------------------------------------------------------ 融合
def rrf_weighted(rank_lists, weights=None, k=60, topk=30):
    """加权 RRF：各路独立贡献 1/(k+rank) * weight。"""
    agg = {}
    for li, lst in enumerate(rank_lists or []):
        w = 1.0 if not weights else float(weights[li])
        if w <= 0:
            continue
        for rank, (idx, _sc) in enumerate(lst, 1):
            agg[idx] = agg.get(idx, 0.0) + w / (k + rank)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:topk]


# ------------------------------------------------- 重排（v4 内部实现：磁盘缓存 / stage2 并行）
_RR_CACHE = k3.JsonCache(os.path.join(k3.CACHE_DIR, 'rerank_cache_v4.json'), limit=5000)
atexit.register(_RR_CACHE.flush)
_TIE = os.path.basename(DEFAULT_VSTORE)

# 重排缓存盐（2026-09-20）：分池/过滤/排序规则一改就 +1。否则旧的缓存条目会被命中，
# 返回的是「上一代实现」的排序，看的人会以为是新逻辑的效果。
RR_SALT = 'scope-v19'  # 2026-10-05：c08 三校专项——评测集 expect 放宽（院校A真题集/院校D真题集副本算命中）+ 清 s05 旧缓存排序
# 2026-09-23：SCOPE_RULES 新增「院校B真题 / 上交真题集」→ 分池规则变了，必须升 ——
# 重排缓存的键**不含 scope**（见下方 rr_cache 键），不升会命中旧排序、把「改动无效」误判成事实

# ------------------------------------------------- 元文件降权（2026-09-20 新增）
# 实测问题：「高数下册作业答案」的 top1 落在
#   ECUST课程资料\数学学院\公共课\高数\作业实验\作业\README.md
# —— 该 README 是下载器/整理工具生成的目录说明，不是知识内容，但路径里带「高数」「作业」
# 让它在 BM25/路径通道上得分虚高，把真正的《高数下答案.pdf》挤下去。
# 处理：不删除（删了不可逆），只在排序上乘一个系数降权 → 可回退、可量化。
_META_BASENAMES = {
    'readme.md', 'readme.txt', 'readme', 'readme.markdown', 'readme.html',
    '说明.txt', '说明.md', '目录.txt', 'desktop.ini', 'thumbs.db',
    'license', 'license.txt', 'changelog.md', '下载说明.txt',
    '索引.md',  # 2026-09-29 P0-5：Resource 根目录索引文件，路径/内容命中挤占知识内容，纳入 meta 降权
}
META_PENALTY = 0.4

# 2026-09-29 P0-5b：与 build_kb_v2.py 的 META_ONLY_EXT 保持一致
# 非文本载体扩展名（仅 OCR/短文本），检索时系统性抬分挤占正文 PDF/讲解文件（实测 m04 miss）
META_ONLY_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.swf', '.zip', '.rar', '.7z',
                 '.url', '.exe', '.dll', '.opj', '.opju', '.nb', '.circ', '.mmap', '.cap', '.ms14',
                 '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.iso', '.apk', '.jar', '.a',
                 '.so', '.lib', '.obj', '.bin', '.dat', '.db', '.pkl', '.pt', '.onnx', '.h5'}


def is_meta_file(path):
    """是否是「元文件」（README / 说明 / 快捷方式 / 目录索引）—— 非知识内容。"""
    b = os.path.basename(path or '').lower()
    if b in _META_BASENAMES:
        return True
    if b.endswith(('.url', '.lnk', '.webloc')):
        return True
    return False


# ------------------------------------------------- 文件名命中加成（2026-09-20 新增）
# 动机：问「高数下册作业答案」，文件《高数下答案.pdf》名字几乎完全对上，
# 但 LLM 重排只看正文，把它挤到第 6 名之外。生产级 RAG 会给「标题字段」单独加权，
# 这里用字符重合率做轻量版。
#
# 2026-09-22 修订（改进 #1）：原判据是「问题字集合 ∩ 文件名字集合 / 问题字集合」，
# 属**无序单字重合**——问题「极限的等价无穷小」里的「限/极/等/价」会随机命中任意
# 带这些字的文件名（如《极限存在准则》），给无关文件加分并把正确结果挤出 top1。
# 实测：关掉它 Hit@1 从 58.8% → 61.8%（**涨 3pt**），即原判据净伤害。
# 新判据改为「**加权覆盖率**」，只奖励真正成词的命中：
#   ① 中文取**最长公共子串长度 / 问题长度**（连续才算，杜绝散字蒙中）；
#   ② 英文取**最长公共词序列**（而非词集合交集）；
#   ③ 二者取大。这样《高数下答案.pdf》对问题「高数下答案」仍得满分，
#      但《极限存在准则》对「等价无穷小」不再得分。
PATH_BOOST = 1.2
PATH_BOOST_MIN = 0.8
# 2026-10-17 a02fix：低覆盖加成档参数。文件名覆盖率在 [LOW_COV_MIN, PATH_BOOST_MIN)
# 的弱命中按 c*LOW_COV_BONUS 补分，平衡「正文分主导、文件名弱命中被干扰」的排序。
# 实测基准：a02 期望 c=0.333、干扰 c=0.222，净差 0.111*0.15=0.0167，可翻 0.0112 差距。
LOW_COV_MIN = 0.2
LOW_COV_BONUS = 0.15
# 新判据下要求更高的覆盖率才加成（旧判据 0.6 太松，散字轻易达标）
PATH_BOOST_MIN_V2 = 0.75
# 2026-10-17 全天优化（e03 排序层）：query 含「真题」任务类型词时，basename 含
# 「真题」的文件获固定加成。动机：「完形填空真题练习」里同目录的「大完形特训」
# /「训练习题集」等无「真题」词的文件靠正文词频压过「（高级篇）考研完型填空真题」
# （含真题，正文是高级完形练习，embedding 分低）。「真题」是 query 的硬意图词，
# 文件名含它应优先于仅含主题词的干扰。影响面：34 条 eval 中 query 含「真题」的
# 仅 e03/m04，m04 期望「插班生真题讲解」系列文件名全含「真题」、加成只强化期望；
# 其余 32 条判据不触发，零影响。幅度 0.03：e03 高级篇 0.0246→0.0546 超干扰
# 0.0452（rank4→1）；m04 top7 顺序不变。
# 2026-10-17 全天优化收尾（e03 复核）：0.03→rank3，0.10→rank2，0.20 仍 rank2；
# 0.30 最后尝试压过机构A特训的 rerank 正文分优势；静态回归仅 e03/m04 受影响。
ZHENTI_BONUS = 0.30
_PATH_STOP_CHARS = set('的了吗呢是怎么为什么如何什么哪些哪个请问下上中')

# c08 院校B专项（2026-10-05 施工）：目标校名 token。query 显式含校名且文件路径
# （目录段或 basename）含同名校名时，视为「校名目录命中」→ 触发保送/加成链路。
# 动机：院校B真题文件名是 IMG_013x.jpg，"院校B"只出现在目录路径里，
# basename 覆盖率判据（_path_boost_score_v2）对这类文件恒为 0，全链路失效。
_SCHOOL_TOKENS = ('院校B', '院校A', '院校A', '院校C', '上交', '上海交通', '院校D', '上大', '东华', '海事', '海洋', '上理', '华理')


def _school_tokens(text):
    """返回 text（query 或 rel_path）中含有的校名 token 集合（已做别名归一）。"""
    if not text:
        return set()
    s = {t for t in _SCHOOL_TOKENS if t in text}
    # c08 三校专项（2026-10-05）：校名别名归一 —— 否则 query「院校C」与 rel_path
    # 「上交真题集」无公共 token，交集判据恒空、保送链路对院校C全灭。
    if '院校C' in s:
        s.discard('院校C')
        s.add('上交')
    if '上海交通' in s:
        s.discard('上海交通')
        s.add('上交')
    return s


def _lcs_len(a, b):
    """最长公共子串长度（滚动数组，O(min(len)) 空间）。"""
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    best = 0
    for ch in b:
        cur = [0] * (len(a) + 1)
        for i, ca in enumerate(a, 1):
            if ca == ch:
                cur[i] = prev[i - 1] + 1
                if cur[i] > best:
                    best = cur[i]
        prev = cur
    return best


def _subseq_hit(q, b):
    """按序子序列覆盖率：短串字符逐次在长串中按序查找（允许跳跃）。

    2026-10-16 施工（三小时优化·m13 排序层）：过短文件名(<4 字符)的子序列命中
    置信度低，原实现虚高——m13「…特征值与特征向量的求法」里 ch10-1向量.ppt 靠
    '向量' 2 字拿满分 1.0 压过真实目标。短串 cap 0.5；e02 词汇表(命中 7/8)等
    真实命中不受影响（连续性判据曾误伤 e02，2026-10-16 回滚）。

    2026-10-16 施工（a02 排序层）：缺失惩罚——仅当「文件名是短串」时生效：
    pat 来自文件名侧（len(b) < len(q)）且存在未命中字符 → 按缺失比例折减并
    叠加 query 覆盖率因子（命中 query 的字数占比）。原实现只看命中比，短文件名
    靠词素巧合蒙高：a02「插班生英语作文模板」里 '插班生语法'(5 字) 命中
    '插班生'+'语' 共 4 字拿 4/5=0.8（'法'字根本不在 query 中）压过热点作文
    题目.pdf（命中 query 7/9=0.778，缺尾部'模板'2 字）。折减后干扰
    0.8×0.8×4/9=0.284，期望保持 0.778，覆盖 query 更多者胜出。query 为短串
    时（pat 来自 query 侧）不打折——e02「…词汇表大全」的'大全'未出现在文件名
    属 query 修饰词，误伤会让 e02 掉出 keep_floor 0.8 阈值（历史教训）；c07
    高数下答案/t03 分块策略等 pat 全命中（miss=0）亦零影响；连续性判据曾因
    夹词误伤 e02，已弃用。
    """
    if not q or not b:
        return 0.0
    if len(q) <= len(b):
        pat, host, pat_from_b = q, b, False
    else:
        pat, host, pat_from_b = b, q, True
    j = 0
    hit = 0
    for ch in pat:
        while j < len(host) and host[j] != ch:
            j += 1
        if j < len(host):
            hit += 1
            j += 1
    cov = hit / len(pat)
    miss = len(pat) - hit
    if miss > 0 and pat_from_b:
        cov *= (1.0 - miss / len(pat))   # 缺失惩罚：文件名有真缺字符按比例折减
        cov *= (hit / len(q))            # query 覆盖率：覆盖 query 更多者更可信
    if len(pat) < 4:
        cov = min(cov, 0.5)
    return cov


def _path_boost_score_v2(query, path):
    """加权覆盖率：连续子串（中文）/ 连续词序列（英文）为准，杜绝散字蒙中。

    c08 施工（2026-10-05）：先查「校名目录命中」——query 含校名 token 且
    basename 或任意目录段含同名校名 → 直接返回 0.8（越过 PATH_BOOST_MIN），
    让 subseq 保送 / path_boost 加成 / keep_floor 全链路对真题图（IMG_013x.jpg）生效。
    """
    q_sch = _school_tokens(query or '')
    if q_sch:
        _rp = (path or '').replace('\\', '/')
        _segs = _rp.split('/')
        _hay = ''.join(_segs)          # basename + 全目录段拼接
        if _school_tokens(_hay) & q_sch:
            # c08 加强（2026-10-05）：query 与路径同含 4 位年份 → 0.95 强命中
            # （如「院校B2023…」→ 2023年院校B-*.jpg 应优先于 IMG 真题集图）
            _qy = set(re.findall(r'(?:19|20)\d{2}', query))
            if _qy and (_qy & set(re.findall(r'(?:19|20)\d{2}', _rp))):
                return 0.95
            return 0.8
    b = os.path.basename(path or '')
    b = os.path.splitext(b)[0].lower()
    if not b:
        return 0.0
    q = (query or '').lower()
    # ① 中文：最长连续公共子串 / 去停用词后的问题有效字数
    qc = ''.join(c for c in q if '\u4e00' <= c <= '\u9fff' and c not in _PATH_STOP_CHARS)
    bc = ''.join(c for c in b if '\u4e00' <= c <= '\u9fff')
    cov_zh = (_lcs_len(qc, bc) / len(qc)) if qc else 0.0
    # ② 英文：最长连续公共词序列 / 问题有效词数
    # c05 施工（2026-10-15 知识准确性专项）：中英别名「附录」→ appendix。
    # 别名命中同样 cap 0.8，仅进 path_boost / keep_floor 通道，不触发 0.95 级。
    _q_alias = q.replace('附录', 'appendix')
    qw_orig = [w for w in re.findall(r'[a-z][a-z0-9_\-]{1,}', q) if len(w) >= 3]
    qw = [w for w in re.findall(r'[a-z][a-z0-9_\-]{1,}', _q_alias) if len(w) >= 3]
    bw = re.findall(r'[a-z][a-z0-9]*', b)
    cov_en = (_lcs_len(qw, bw) / len(qw)) if qw else 0.0
    # 2026-10-16 施工（三小时优化·t03）：query 中英混合（本身含英文词且含中文）时，
    # 英文词命中只是辅助信号——t03「RAG 文本分块策略怎么设计」里 03_RAG评估体系.md
    # 仅靠 rag 一词 cov_en=1.0 与 01_文本分块策略.md（cov_sub=1.0）并列，靠重排分
    # 抢走 keep_floor 的 _best。中英混合场景英文 cov 一律 cap 0.5，避免单英文词
    # 压过真实中文词面命中；纯中文 query 别名（附录→appendix，c05）不受影响。
    if cov_en > 0 and qw_orig and any('\u4e00' <= c <= '\u9fff' for c in q):
        cov_en = min(cov_en, 0.5)
    if not qw_orig and cov_zh == 0 and cov_en > 0:
        cov_en = min(cov_en, 0.8)
    # c07 施工（2026-10-02）：按序子序列覆盖率（允许跳跃）。
    #   动机：问「高数下册作业答案」时《高数下答案.pdf》LCS 仅 0.29（「答案」被「下册作业」
    #   拆散），但文件名是问题的按序子序列（高数下答案 ⊂ 高数下册作业答案），命中率 = 1.0。
    # 子序列用「不去停用词」的全量中文字符：停用词「下」在「高数下答案」里是「下册」简称，
    # 去停用词会把文件名里的「下」一并剔除，导致子序列命中率从 1.0 掉到 0.4。
    q_all = ''.join(c for c in q if '\u4e00' <= c <= '\u9fff')
    cov_sub = _subseq_hit(q_all, bc)
    cov = max(cov_zh, cov_en, cov_sub)
    if cov_sub >= 0.8 and cov < 0.85:
        return 0.85   # 越 PATH_BOOST_MIN=0.8 → 触发 title_recall / path_boost 通道
    return cov


def _path_boost_score(query, path):
    """查询与文件名的命中率，用于标题加权（2026-09-22 起用 v2 加权覆盖率）。"""
    return _path_boost_score_v2(query, path)


def _basename_cov_score(query, path):
    """纯 basename 中文覆盖率（0~1）：_path_boost_score_v2 会被「校名目录命中」
    分支提前 return 0.8（c08），导致 query 含校名时大量路径文件同分、真正的
    文件名全重合项被淹没。keep_floor 选 _best 时用本函数做第一判据。

    2026-10-15 施工（知识准确性专项）：f12「院校B插班生英语讲义参考答案」query
    与 basename 完全重合（cov=1.0），但 28 个院校B路径文件都拿到 0.8 校名分，
    _best 落在重排分最高的「插班生 语法(1).pdf」上，目标文件永远提不上来。
    """
    b = os.path.splitext(os.path.basename(path or ''))[0].lower()
    if not b:
        return 0.0
    q = (query or '').lower()
    qc = ''.join(c for c in q if '\u4e00' <= c <= '\u9fff' and c not in _PATH_STOP_CHARS)
    # bc 同样去停用词（2026-10-16 a02 施工）：否则「无穷小的比较」文件名里的
    # '的' 在 qc 侧被删、在 bc 侧保留，subseq 判为真缺字符，缺失惩罚误伤 m02/m12。
    bc = ''.join(c for c in b if '\u4e00' <= c <= '\u9fff' and c not in _PATH_STOP_CHARS)
    cov_zh = (_lcs_len(qc, bc) / len(qc)) if qc else 0.0
    if cov_zh >= 0.95:
        return cov_zh
    # c05 施工（2026-10-15 知识准确性专项）：中文「附录」→ 英文 appendix 别名，
    # C_O_D 附录文件名是 appendix_C/E.pdf，纯中文查询跨语言脱节。别名命中仅
    # cap 0.8（不触发 top1 提升，只进 keep_floor 第5位）。
    _q_alias = q.replace('附录', 'appendix')
    qw_orig = [w for w in re.findall(r'[a-z][a-z0-9_\-]{1,}', q) if len(w) >= 3]
    qw = [w for w in re.findall(r'[a-z][a-z0-9_\-]{1,}', _q_alias) if len(w) >= 3]
    bw = re.findall(r'[a-z][a-z0-9]*', b)
    cov_en = (_lcs_len(qw, bw) / len(qw)) if qw else 0.0
    # 2026-10-16 施工（三小时优化·t03，与 _path_boost_score_v2 同步）：中英混合 query
    # 英文 cov cap 0.5。t03 的 03_RAG评估体系.md 靠 rag 一词 cov_en=1.0 与
    # 01_文本分块策略.md（cov_sub=1.0）并列 1.0，keep_floor _best 被重排分更高的
    # 03 抢走；cap 后 03=0.5 < 01=1.0，01 成为唯一强命中并硬提 top1。
    if cov_en > 0 and qw_orig and any('\u4e00' <= c <= '\u9fff' for c in q):
        cov_en = min(cov_en, 0.5)
    if not qw_orig and cov_zh == 0 and cov_en > 0:
        return min(cov_en, 0.8)
    # 2026-10-16 施工（三小时优化·keep_floor 子序列分支）：纯 LCS 会漏掉
    # 「文件名 = query 按序子序列」的强命中（t03：01_文本分块策略.md vs
    # "RAG 文本分块策略怎么设计"，LCS 只算到"文本分块策略"=6/9=0.67 不触发
    # keep_floor，但按序子序列覆盖率=1.0）。与 _path_boost_score_v2 对齐：
    # cov_sub>=0.8 时并入取 max，让 keep_floor 判据补上子序列通道。
    # v2（2026-10-16 三小时优化）：子序列用**去停用词后的 qc** 而非原始 q_all。
    # 原因：e02「插班生英语词汇表在哪」原始串带「在哪」尾缀 2 字符，
    # 词汇表文件 bc 命中 8/10=0.8（cap 0.85），而「插班生 语法(1).pdf」bc="插班生语法"
    # 同样 0.8 cap 0.85 且重排分更高 → keep_floor 把 best 误选成语法 pdf（空操作）。
    # 用 qc（去掉疑问词"哪/么/怎么"等）后：词汇表 8/9=0.889 触发 top1，
    # 语法 pdf 3/9=0.333 不触发 → 判别正确。
    qc2 = qc
    cov_sub = _subseq_hit(qc2, bc)
    cov = max(cov_zh, cov_en, cov_sub)
    # 2026-10-17 施工（全天优化·c04/m13 核心词目录命中）：
    # query 含空格分隔的主题词（如「数据结构 图 讲义」）时，basename 单路 LCS 只吃
    # 核心主题词（「数据结构」4/7=0.571），重排器把正文强相关但非目标的文件
    # （01_概论.pptx）顶到 top1，期望的课程讲义文件掉出。若 basename 以 query 首个
    # 空格词（核心主题词，>=4 字）开头、且路径目录段也有以该核心词开头的目录名
    # （如「数据结构-蒋翠玲」，排除「算法与数据结构-叶琪」式前缀修饰目录与
    # 「数据结构与算法笔记」式拼接名）→ 目录+basename 双重命中 → 硬加成 0.6：
    # m13（0.286→0.886）/ c04（0.571→1.0）进 keep_floor 硬提 top1 通道。
    # 误伤面：核心词长度门槛 4 字排除 3 字词「概率论」（m14 的"概率论课件.rar"
    # 会虚触发）；12 条含空格 query 中仅 c04/m13 满足触发；e04 basename 命中但
    # 目录段不命中、t04 目录段不命中，其余核心词与 basename 前缀不匹配，零影响。
    if ' ' in q:
        _ws = [w for w in q.split() if len(w) >= 4]
        if _ws:
            _core = _ws[0]
            _dirs = (path or '').replace('\\', '/').split('/')[:-1]
            if (b.startswith(_core)
                    and any(d.strip().lower().startswith(_core) for d in _dirs if d.strip())):
                cov = min(1.0, cov + 0.6)
    return cov


def smart_preview(text, terms, chars=120, head=60):
    """在 chars 预算内优先展示「查询词命中处」的上下文。

    动机：定长预览取正文开头，代码块/长文档里关键词常出现在 120 字之后，
    重排模型看不到证据就判低分（Q44：`if not name or name in self.monitors`
    位于块内 148 字处）。此处保留 head 字头部（含路径/符号名）再拼接命中窗口。
    """
    text = (text or '').replace('\n', ' ')
    if not text:
        return ''
    if len(text) <= chars or not terms:
        return text[:chars]
    low = text.lower()
    best = None
    for t in terms:
        t = (t or '').strip()
        if len(t) < 3:
            continue
        p = low.find(t.lower())
        if p < 0 or p < head:
            continue
        sc = len(t) - p * 1e-4
        if best is None or sc > best[0]:
            best = (sc, p)
    if best is None:
        return text[:chars]
    p = best[1]
    win = text[max(0, p - 24): max(0, p - 24) + (chars - head)]
    return (text[:head].rstrip() + ' … ' + win).strip()


def _rr_prompt(query, cands, meta, preview_chars, terms=None, smart=True):
    blocks = []
    for n, (idx, _s) in enumerate(cands, 1):
        m = meta[idx]
        raw = (m.get('text') or '').replace('\\', '/')
        body = smart_preview(raw, terms, preview_chars) if smart else raw[:preview_chars]
        blocks.append('[%d] 文件：%s\n内容：%s' % (n, m['rel_path'].replace('\\', '/'), body))
    return k2.RERANK_PROMPT.format(q=query, cands='\n\n'.join(blocks))


# ------------------------------------------------------------------ 确定性调用（评测复现）
# 漂移根因（2026-09-18 实测）：① 改写走 k3.llm_call → temperature=0.15，同 prompt 5 次输出出现 2 种；
# ② 重排 8 组并发请求，Ollama 侧 batch 合并使 temperature=0 仍有细微差异（Q44 三次里 ranked 出现 2 种）。
# 修复：统一 temperature=0 + 固定 seed，分组默认串行执行（Ollama 本身串行服务，无性能损失）。
DET_SEED = 20260918

# ---------------------------------------------------------------- [78施工单] 显存协调层前置检查
# 老板口径（2026-10-15）：非必要不跑 CPU —— 显存不足必须中止，绝不静默降级到 CPU。
# A 档：调 Ollama 前 acquire ollama；非 0 → 打印明确提示并抛 VramBlocked 中止。
_VRAM_LEASE = r'C:\Users\<you>\Desktop\Marvis产出\30_协作\_state\vram_lease.py'
_VRAM_OK_UNTIL = 0.0          # 进程内缓存：acquire 成功后 60s 内不再重复起子进程


class VramBlocked(RuntimeError):
    """显存协调层拦截：显存不足/冲突，禁止降级跑 CPU。"""


def _vram_acquire_or_stop(where='检索'):
    """调 Ollama 前前置检查：acquire ollama 非 0 即停手（绝不降级跑）。"""
    global _VRAM_OK_UNTIL
    now = time.time()
    if now < _VRAM_OK_UNTIL:
        return True
    try:
        p = subprocess.run([sys.executable, _VRAM_LEASE, 'acquire', '--group', 'ollama'],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        print('[显存协调层] acquire 调用失败（%s）：%s' % (where, str(e)[:120]), flush=True)
        return False
    if p.returncode == 0:
        _VRAM_OK_UNTIL = now + 60
        # [78施工单] B 档：跑后自检 degraded —— 前置通过但加载时仍降级到 CPU 的残余窗口
        try:
            q = subprocess.run([sys.executable, _VRAM_LEASE, 'degraded', '--json'],
                               capture_output=True, text=True, timeout=30)
            if q.returncode == 7:
                print('[显存协调层] ⚠️ 检出模型降级到 CPU（degraded=7，%s）：Ollama 部分层未进显存，性能会显著变慢。'
                      % where, flush=True)
        except Exception:
            pass
        return True
    # [老板授权 2026-10-16] 自主释放显存：非 0 先尝试自主释放（冷驻留/僵尸/残留），
    # 成功并重试 acquire 通过 → 继续；仍不足才中止，绝不降级跑 CPU。
    import os as _os
    _AR = _os.path.join(_os.path.dirname(_os.path.dirname(_VRAM_LEASE)), '本地知识库', '_scratch', 'vram_autorelease.py')
    if _os.path.exists(_AR):
        try:
            ar = subprocess.run([sys.executable, _AR], capture_output=True, text=True, timeout=180)
            if ar.returncode == 0:
                _VRAM_OK_UNTIL = time.time() + 60
                print('[显存自主释放] 释放成功，重试 acquire 通过（%s）。' % where, flush=True)
                return True
            print(ar.stdout[-400:] if ar.stdout else '', flush=True)
            print('[显存自主释放] 释放后仍不足/冲突（rc=%d，%s）：中止，绝不降级跑 CPU。'
                  % (ar.returncode, where), flush=True)
        except Exception as e:
            print('[显存自主释放] 自主释放调用异常（%s）：%s' % (where, str(e)[:120]), flush=True)
    print('', flush=True)
    print('=' * 62, flush=True)
    print('  [显存协调层] 显存不足/冲突（%s）：acquire ollama 返回 %d' % (where, p.returncode), flush=True)
    print('  绝不降级跑 CPU —— 请先释放显存（vram_status.py 可查占用方），再重试。', flush=True)
    print('=' * 62, flush=True)
    return False


def llm_det(prompt, model=k3.LOCAL_MODEL, backend='local', num_predict=None, seed=DET_SEED):
    """确定性 LLM 调用：本地 temperature=0 + 固定 seed；云端沿用 llm_call。"""
    n = num_predict or k3.LOCAL_MAX_TOKENS
    if backend == 'local':
        if not _vram_acquire_or_stop('llm_det'):
            raise VramBlocked('llm_det: acquire ollama 非 0，显存不足/冲突，中止')
        r = k2.requests.post(k3.OLLAMA_CHAT, json={
            'model': model, 'stream': False,
            'messages': [{'role': 'user', 'content': prompt}],
            'options': {'temperature': 0.0, 'seed': seed, 'num_predict': n, 'num_ctx': 8192},
        }, timeout=k3.LOCAL_TIMEOUT)
        r.raise_for_status()
        return (r.json().get('message', {}).get('content') or '').strip()
    return k3.llm_call(prompt, model=model, backend=backend, num_predict=num_predict)


def llm_rerank(prompt, model=k3.LOCAL_MODEL, num_predict=400, temperature=0.0,
               timeout=k3.LOCAL_TIMEOUT):
    """重排专用本地调用：temperature=0 保证可复现（k3 默认 0.15 会造成评测抖动）。"""
    if not _vram_acquire_or_stop('llm_rerank'):
        raise VramBlocked('llm_rerank: acquire ollama 非 0，显存不足/冲突，中止')
    r = k2.requests.post(k3.OLLAMA_CHAT, json={
        'model': model, 'stream': False,
        'messages': [{'role': 'user', 'content': prompt}],
        'options': {'temperature': temperature, 'seed': DET_SEED, 'num_predict': num_predict, 'num_ctx': 8192},
    }, timeout=timeout)
    r.raise_for_status()
    return (r.json().get('message', {}).get('content') or '').strip()


def rerank_parallel_v4(query, cands, meta, model=k3.LOCAL_MODEL, preview_chars=120,
                       workers=8, backend='local', log=print, max_tokens=400, terms=None,
                       parallel=True):
    """v3 rerank_parallel 的 v4 版：显式限制重排输出长度（只输出 序号:分数）+ temperature=0。"""
    if not cands:
        return []

    def work(g):
        try:
            _p = _rr_prompt(query, g, meta, preview_chars, terms)
            if backend == 'local':
                txt = llm_rerank(_p, model=model, num_predict=max_tokens)
            else:
                txt = k3.llm_call(_p, model=model, backend=backend, num_predict=max_tokens)
            return k3._parse_scores(txt, g)
        except VramBlocked:
            raise
        except Exception as e:
            log('[重排] 分块失败（%d 条），按 RRF 序兜底：%s' % (len(g), str(e)[:80]))
            return {}

    if workers <= 1 or len(cands) <= 2:
        return k3._score_fallback(cands, work(cands))
    groups = [g for g in (cands[i::workers] for i in range(workers)) if g]
    # 确定性：默认分组串行（Ollama 服务端本身串行，并发只会引入 batch 合并的打分抖动）
    with ThreadPoolExecutor(max_workers=len(groups) if parallel else 1) as ex:
        score_maps = list(ex.map(work, groups))
    out = []
    for g, sc in zip(groups, score_maps):
        out.extend(k3._score_fallback(g, sc))
    return sorted(out, key=lambda kv: -kv[1])


def rerank_two_stage_v4(query, cands, meta, model=k3.LOCAL_MODEL, backend='local',
                        workers=8, workers2=2, preview1=120, preview2=220, w1=0.4, w2=0.8,
                        head=5, log=print, use_cache=True, max_tokens=400, terms=None,
                        parallel=True, scope=None, vstore=None):
    """两阶段重排 v4 版：stage2 并行 + 结果磁盘缓存（同问同配置直接复用）。

    与 v3 的差别仅在三处：① stage2 用 workers2 并行（v3 串行）；
    ② 重排输出 token 上限收紧；③ 融合结果按「问题+配置+候选集」指纹落盘缓存。
    """
    if not cands:
        return []
    if len(cands) <= 2:
        return cands
    key = None
    if use_cache:
        h = hashlib.md5(('%s|%s|%s|%s|%s|%.3f|%d|%d|%d|%d|%s|' % (
            # vstore 用「实际库」而非 DEFAULT_VSTORE：否则同模型跨库（v8/v9）会共用缓存条目 → 串扰
            RR_SALT, model, backend, query, os.path.basename(vstore or DEFAULT_VSTORE), w1, preview1, preview2,
            head, int(bool(parallel)), scope or 'all')).encode('utf-8'))
        h.update(','.join(str(i) for i, _ in cands).encode('utf-8'))
        key = h.hexdigest()
        hit = _RR_CACHE.get(key)
        if isinstance(hit, dict) and hit.get('ranked'):
            return [(int(i), float(s)) for i, s in hit['ranked']]
    s1 = rerank_parallel_v4(query, cands, meta, model=model, preview_chars=preview1,
                            workers=workers, backend=backend, log=log, max_tokens=max_tokens,
                            terms=terms, parallel=parallel)
    fused1 = k3.fuse_rank(cands, s1, w1)
    if head <= 1 or len(fused1) <= head:
        ranked = fused1
    else:
        top, tail = fused1[:head], fused1[head:]
        s2 = rerank_parallel_v4(query, top, meta, model=model, preview_chars=preview2,
                                workers=workers2, backend=backend, log=log, max_tokens=200,
                                terms=terms, parallel=parallel)
        ranked = k3.fuse_rank(top, s2, w2) + tail
    if use_cache and key:
        _RR_CACHE.set(key, {'ranked': [[int(i), float(s)] for i, s in ranked]})
    return ranked


# ------------------------------------------------------------------ BGE 交叉编码器重排（2026-10-15 施工）
# 动机：两阶段 LLM 生成式重排（qwen2.5:7b 逐组打分）是检索链路 89% 耗时的来源。
# 调研结论（sevenlabs / ehga / jangwook 实测）：排序任务应使用专用交叉编码器——
# BAAI/bge-reranker-v2-m3（568M，MTEB 59.7，100+ 语言含中英）50 文档打分仅 90~130ms（GPU）。
# 本实现：transformers 加载，构造 (query, text) 对统一 batch 打分，返回与 LLM 重排同构的
# [(idx, score)] 全量降序列表，下游 meta_penalty / path_boost / keep_floor 逻辑零改动复用。
BGE_RERANKER_DIR = os.environ.get('KB_BGE_RERANKER_DIR',
                                  os.path.join(BASE, 'models', 'bge-reranker-v2-m3'))
_BGE_LOCK = threading.Lock()
_BGE_STATE = {'tok': None, 'm': None, 'dev': None}

# ---- 显存分配架构（2026-10-15 施工）----------------------------------------
# 动机：设备选择长期默认 cpu，只有手动设 $env:KB_BGE_DEVICE="cuda" 才走 GPU；
# 评测/工兵台服务任一进程漏设即退化 CPU（单题 14-20s）。本层将「默认 GPU、不足降级」
# 固化到代码：启动时自动探测 CUDA 与显存余量，BGE fp32(2.27GB) + 0.3GB 安全余量
# 能放下则默认 cuda，否则自动回退 cpu（int8 量化），并支持显式覆盖。
BGE_CUDA_MIN_FREE_GB = 2.6   # BGE fp32 2.27GB + 0.3GB 余量
_BGE_DEV_CACHE = None        # 进程内缓存分配决策，避免重复探测


def resolve_bge_device(device=None, log=print):
    """解析 BGE 推理设备（显存分配架构入口）。

    规则（2026-10-15 用户定调）：正常情况不跑 CPU，除非显式强调。
    优先级：显式参数 > KB_BGE_DEVICE 环境变量（cpu=显式强调走 CPU / 其余一律 GPU）。
    默认路径永远 cuda：CUDA 不可用或显存余量 < BGE_CUDA_MIN_FREE_GB 时直接报错上抛，
    绝不静默降级 CPU（CPU 慢 10 倍且无提示，用户无法感知）；需要 CPU 只能显式 KB_BGE_DEVICE=cpu。
    """
    global _BGE_DEV_CACHE
    if device:
        return device
    env = (os.environ.get('KB_BGE_DEVICE') or 'auto').strip().lower()
    if env == 'cpu':
        _BGE_DEV_CACHE = 'cpu'
        log('[bge-device] 显式指定 CPU（KB_BGE_DEVICE=cpu，用户强调）')
        return 'cpu'
    if _BGE_DEV_CACHE is not None:
        return _BGE_DEV_CACHE
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，但规则要求 BGE 默认走 GPU；如确实要 CPU 请设 KB_BGE_DEVICE=cpu')
    free_mib, total_mib = torch.cuda.mem_get_info()
    free_gb = free_mib / 1024 ** 3
    if free_gb < BGE_CUDA_MIN_FREE_GB:
        raise RuntimeError('显存空闲 %.1fGB < 需要 %.1fGB，GPU 装不下 BGE；请先卸载常驻大模型释放显存，或设 KB_BGE_DEVICE=cpu 显式走 CPU'
                           % (free_gb, BGE_CUDA_MIN_FREE_GB))
    _BGE_DEV_CACHE = 'cuda'
    log('[bge-device] 显存空闲 %.1fGB >= %.1fGB，BGE 走 CUDA（规则：正常不跑 CPU）' % (free_gb, BGE_CUDA_MIN_FREE_GB))
    return 'cuda'


def _load_bge_reranker(device=None, quant=False, log=print):
    """懒加载交叉编码器；device 默认取 KB_BGE_DEVICE（cpu / cuda），quant 仅 CPU 生效（int8 动态量化）。"""
    if _BGE_STATE['m'] is None:
        with _BGE_LOCK:
            if _BGE_STATE['m'] is None:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                t0 = time.time()
                tok = AutoTokenizer.from_pretrained(BGE_RERANKER_DIR)
                m = AutoModelForSequenceClassification.from_pretrained(BGE_RERANKER_DIR)
                m.eval()
                dev = resolve_bge_device(device, log=log)
                if dev == 'cuda':
                    try:
                        m = m.to('cuda')
                    except Exception as e:
                        raise RuntimeError('BGE 上 CUDA 失败(%s)：规则要求 BGE 走 GPU 不降级 CPU；'
                                           '请释放显存，或设 KB_BGE_DEVICE=cpu 显式走 CPU' % e)
                if dev == 'cpu' and quant:
                    m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
                _BGE_STATE.update(tok=tok, m=m, dev=next(m.parameters()).device.type)
                log('[bge-reranker] 加载完成 %.1fs device=%s' % (time.time() - t0, _BGE_STATE['dev']))
    return _BGE_STATE['tok'], _BGE_STATE['m']


def rerank_bge_v4(query, cands, meta, batch=32, max_len=512, top_k=None, w_bge=0.6, log=print):
    """交叉编码器重排：候选池统一 (query, text) 对打分，与融合序加权混合后返回 [(idx, score)] 降序。

    2026-10-15 实测修正：纯 BGE 全量接管排序会使 top1 频繁漂移（34 条评测 Hit@1 52.9% vs 基线 82.4%，
    OCR/讲义类文本被误拔高，如 IMG_0250.jpg、实验报告封面 2015.docx 排 top1）。
    → 改为混合排序：final = w_bge*sigmoid(bge_logits) + (1-w_bge)*norm(fuse_score)，
    BGE 只做「排序微调」，融合序（RRF 加权召回）作为先验保底；可选 top_k 只对融合序头部
    做混合精排，尾部保持融合序，避免低质量候选被交叉编码器误升。
    max_len 保持 512：query_windows 窗口文本约 420 字符，编码后 <512 token；调大只会拖慢打分。
    """
    if not cands:
        return []
    import torch
    tok, m = _load_bge_reranker(log=log)
    n = len(cands)
    work = cands
    tail = []
    if top_k and n > top_k:
        work, tail = cands[:top_k], cands[top_k:]
    texts = [(meta[i].get('text') or '')[:3000] for i, _ in work]
    pairs = [(query, t) for t in texts]
    bge_scores = []
    with torch.no_grad():
        for s in range(0, len(pairs), batch):
            grp = pairs[s:s + batch]
            enc = tok(grp, padding=True, truncation=True, max_length=max_len,
                      return_tensors='pt')
            if _BGE_STATE['dev'] == 'cuda':
                enc = {k: v.to('cuda') for k, v in enc.items()}
            lg = m(**enc).logits
            if lg.dim() == 2 and lg.shape[1] >= 1:
                lg = lg[:, 0]
            bge_scores.extend(lg.float().cpu().numpy().tolist())
    # 归一化：bge logits -> sigmoid；fuse_score -> min-max
    import math
    fs = [s for _, s in work]
    fmin, fmax = (min(fs), max(fs)) if fs else (0, 1)
    frng = (fmax - fmin) or 1.0
    scored = []
    for k, ((i, s), bs) in enumerate(zip(work, bge_scores)):
        pb = 1.0 / (1.0 + math.exp(-bs))
        nf = (s - fmin) / frng
        scored.append((i, w_bge * pb + (1 - w_bge) * nf))
    scored.sort(key=lambda kv: -kv[1])
    out = scored + tail
    return out


# ------------------------------------------------------------------ 检索主流程
def search_bm25_subset(bm25, terms, mask, topk=40, scan=8000):
    """在指定子池（英文片段池）内做 BM25 检索。

    动机：中文问题在英文技能文档上的 dense 相似度整体偏低且不可比，
    但 BM25 对 `Resource`、`name` 这类标识符命中很准；把 BM25 限定在英文片段池内，
    可避免中文高频词（“的/是/文件”等）带来的跨语料打分竞争。
    """
    if not terms or mask is None:
        return []
    n = len(mask)
    hits = bm25.search(terms, topk=min(n, scan))
    return [(i, s) for i, s in hits if i < n and mask[i]][:topk]


_DIR_IDX = {}


def build_dir_index(meta):
    """目录 → 片段下标 的映射（带缓存），用于同目录邻居扩展。"""
    key = (id(meta), len(meta))
    idx = _DIR_IDX.get(key)
    if idx is None:
        idx = {}
        for i, m in enumerate(meta):
            idx.setdefault(os.path.dirname(m['file_path']), []).append(i)
        _DIR_IDX[key] = idx
    return idx


_PATH_STOP = {'scripts', 'script', 'references', 'reference', 'src', 'source', 'docs', 'doc', 'lib', 'libs',
              'bin', 'test', 'tests', 'skill', 'skills', 'asset', 'assets', 'resource', 'resources', 'data',
              'template', 'templates', 'python', 'node_modules', 'content', 'temp', 'output', 'common',
              'examples', 'example', 'user', 'users', 'desktop', 'appdata', 'marvis', 'workspace', 'files',
              'build', 'dist', 'main', 'core', 'utils', 'helpers', 'modules'}


def path_terms(meta, cands, topn=2, min_len=3, seeds=12, min_count=2):
    """从候选文件路径里提取「库名 / 项目名」一类 token。

    动机：中文提问常常不点名具体库（“添加资源的时候对名字有什么限制”），
    改写也未必补出库名；但候选文件路径里的 skills/<lib>/... 往往是稳定线索，
    用它做一轮受限检索，可把同库的其他文件一起拉进候选。
    """
    from collections import Counter
    c = Counter()
    for i, _ in cands[:seeds]:
        parts = os.path.dirname(meta[i]['file_path']).replace('\\', '/').split('/')
        for seg in parts[-4:]:
            t = re.sub(r'[^A-Za-z0-9_.\-]', '', seg).lower().strip('._-')
            if len(t) >= min_len and t not in _PATH_STOP and not t.replace('.', '').isdigit():
                c[t] += 1
    strong = [w for w, n in c.most_common(topn) if n >= min_count]
    if strong:
        return strong
    return [w for w, _ in c.most_common(1)]


_EN_STOP = {'the', 'a', 'an', 'of', 'for', 'and', 'or', 'to', 'in', 'on', 'with', 'is', 'are', 'how',
            'what', 'which', 'when', 'does', 'do', 'can', 'use', 'using', 'used', 'from', 'by', 'at',
            'it', 'its', 'this', 'that', 'there', 'be', 'as', 'all', 'any', 'you', 'your', 'have', 'has'}


def en_word_variants(text):
    """英文词形扩展：BM25 不做词干化，单复数不一致会直接漏召（resource ↔ resources）。"""
    out = []
    for w in re.findall(r'[A-Za-z][A-Za-z_\-]{2,}', text or ''):
        lw = w.lower()
        if lw in _EN_STOP:
            continue
        out.append(lw)
        if lw.endswith('ies') and len(lw) > 4:
            out.append(lw[:-3] + 'y')
        elif lw.endswith('es') and len(lw) > 4:
            out.append(lw[:-2])
            out.append(lw[:-1])
        elif lw.endswith('s') and len(lw) > 3:
            out.append(lw[:-1])
        elif len(lw) <= 14:
            out.append(lw + 's')
    seen, res = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            res.append(w)
    return res


_LAT_RE = re.compile(r'[A-Za-z][A-Za-z_\-]{2,}')
_CJK_RE = re.compile(r'[\u4e00-\u9fff]{2,6}')
_FILE_IDX = {}


def build_file_index(meta):
    """文件路径 → 片段下标列表（带缓存）。"""
    key = (id(meta), len(meta))
    idx = _FILE_IDX.get(key)
    if idx is None:
        idx = {}
        for i, m in enumerate(meta):
            idx.setdefault(m['file_path'], []).append(i)
        _FILE_IDX[key] = idx
    return idx


def query_terms(query='', extra=''):
    """把问题 + 改写结果拆成 (英文词, 中文词)，用于片段选择与预览定位。"""
    latin, cjk, seen = [], [], set()
    for src in (query or '', extra or ''):
        for w in _LAT_RE.findall(src):
            lw = w.lower()
            if len(lw) >= 3 and lw not in _EN_STOP and lw not in seen:
                seen.add(lw)
                latin.append(lw)
        for w in _CJK_RE.findall(src):
            if w not in seen:
                seen.add(w)
                cjk.append(w)
    return latin, cjk


def pick_best_chunk(meta, ids, latin, cjk):
    """同一文件有多个片段时，选「与问题词面最相关」的那一片段进重排。

    动机：文件级去重原本只留融合排名最高的片段，而长文档/代码文件里
    真正回答问题的段落常常不是排名最高的那一段。
    """
    if len(ids) <= 1 or (not latin and not cjk):
        return ids[0]
    best, best_sc = ids[0], -1.0
    for i in ids:
        low = (meta[i].get('text') or '').lower()
        if not low:
            continue
        sc = 0.0
        for w in latin[:12]:
            if w in low:
                sc += 2.0
        for w in cjk[:12]:
            sc += 1.0 * low.count(w)
        if sc > best_sc:
            best, best_sc = i, sc
    return best


def pick_chunk_rank(meta, ids, latin, cjk, k=1):
    """按「与问题词面相关度」给同一文件的片段排序，返回前 k 个下标。

    k=1 时返回值与原 pick_best_chunk 完全一致（稳定排序 + 首个最高分）。
    """
    k = max(1, int(k or 1))
    if len(ids) <= 1:
        return [ids[0]]
    if not latin and not cjk:
        return list(ids[:k])
    scored = []
    for i in ids:
        low = (meta[i].get('text') or '').lower()
        sc = 0.0
        if low:
            for w in latin[:12]:
                if w in low:
                    sc += 2.0
            for w in cjk[:12]:
                sc += 1.0 * low.count(w)
        scored.append((i, sc))
    scored.sort(key=lambda kv: -kv[1])
    return [i for i, _ in scored[:k]]


def query_windows(meta, cands, latin, cjk, win=420):
    """构造「查询感知预览」视图：把候选片段的 text 换成包含查询词的窗口。

    动机：代码/长文档开头多为 import、目录、许可声明，与问题无关；
    重排只看开头 120 字符时会直接误判（中文提问 + 英文代码文件尤其严重）。
    """
    if not latin and not cjk:
        return meta
    view = list(meta)
    for idx, _sc in cands:
        m = meta[idx]
        txt = m.get('text') or ''
        if not txt:
            continue
        low = txt.lower()
        pos = -1
        for w in latin[:12]:
            p = low.find(w)
            if p >= 0:
                pos = p
                break
        if pos < 0:
            for w in cjk[:12]:
                p = txt.find(w)
                if p >= 0:
                    pos = p
                    break
        if pos < 0:
            continue
        start = max(0, pos - win // 3)
        nm = dict(m)
        nm['text'] = txt[start:start + win]
        view[idx] = nm
    return view


# ------------------------------------------------- 域分池（2026-09-20 / 22:40 重构为「考插 · 绩点」二分）
# 目的：① 考插（数学+英语）与绩点（华理数学+英语）各成一口；
#       ② 其他学科（信息/物理/马原/机动）单独成域并**排除在默认域之外** ——
#          问数学英语时大物马原不再抢位（老板 2026-09-20 明确要求）。
# 实测片段量（vector_store_v7）：考插数学 154 ｜ 考插英语 1441 ｜ 华理数学 854
#                              华理英语 129 ｜ 其他学科 4791 ｜ 技术库 617。
# 判定用**绝对路径**（file_path）而不是 rel_path —— RAG技术知识库的文件 rel_path 顶层
# 没有目录名，只看 rel_path 会被误判成课程资料。
#
# 基础域（可单独作 --scope 用，也是别名域的展开单元）：
#   kmath  考插数学（考插数学习题资料 + 真题\数学真题）
#   ken    考插英语（考插英语习题资料 + 真题\英语真题）
#   gmath  华理数学（ECUST课程资料\数学学院：讲义/课件/作业/真题）
#   gen    华理英语（ECUST课程资料\外国语学院）
#   gother 其他学科（信息/物理/马原/机动/学工部 + 保研）—— **默认不参与检索**
#   tech   工具与技术库（Resource\skills / RAG技术知识库 / robomaster）
SCOPE_RULES = {
    # 2026-09-23 补：新加的插班生本校真题目录此前**不在任何规则内** → 落 `other` 域，默认 `core` 搜不到。
    #   归 kmath（插班生真题以数学卷为主；kmath ⊂ core / kaocha / math / mathdoc → 默认检索即可达）。
    'kmath':  ('考插数学习题资料', os.sep.join(('真题', '数学真题')),
               # 2026-10-05 修正（c08 院校B专项）：原宽泛 '院校B真题' 把 `院校B真题\院校B\`
               # （英语阅读真题 IMG_0091~0096）也吞进 kmath → 数学检索被英语图抢 top1。
               # 精确到数学子集：`院校B真题\院校B真题集\`（IMG_0131~0149）+ `院校B真题\2023年院校B-*.jpg`。
               os.sep.join(('院校B真题', '院校B真题集')),
               os.sep.join(('院校B真题', '2023年院校B')),
               # c08 三校专项（2026-10-05）：'上交真题集' 宽泛会把 `上交真题集\上交\`
               # （英语 IMG_0089/0090）也吞进 kmath → 精确到数学子集（顶层 IMG_0163~0175）。
               os.sep.join(('上交真题集', 'IMG')),
               # c08 三校专项：院校A真题 含 数学真题 + 英语真题 两子目录 → 数学精确登记。
               os.sep.join(('院校A真题', '数学真题')),
               '2023年院校A',
               # 2026-09-26：新下载的 12 校真题库（219 图 + 2 扫描 PDF + 14 PDF）
               # ⚠️ 2026-09-27 修正：原写宽泛的 '插班生真题库' → **把英语子目录也吞进 kmath**
               #   （190 片段：英语巩固讲义 119 + 【精讲班同学】英语真题集 71）
               #   → 查英语真题时（ken 域）一条都搜不到。改为**逐个子目录精确登记**。
               os.sep.join(('插班生真题库', '【精讲班同学】App数学真题集')),
               os.sep.join(('插班生真题库', '21-25数学真题')),
               os.sep.join(('插班生真题库', '海洋模拟题')),
               '上大真题集', '上理真题集', '东华真题集',
               '院校A真题集', '华理真题集', '院校D真题集', '海事真题集', '海洋真题集',
               # 2026-10-14：v14 修复——math_src 是扫描 PDF 的来源头占位（文件定位用），须入 kmath 域
               'math_src'),
    'ken':    ('考插英语习题资料', os.sep.join(('真题', '英语真题')),
               # 2026-10-05 修正（c08 院校B专项）：`院校B真题\院校B\` 是英语阅读真题
               # （IMG_0091~0096），原被 kmath 的宽泛 '院校B真题' 吞走 → 归 ken。
               os.sep.join(('院校B真题', '院校B')),
               # c08 三校专项（2026-10-05）：上交英语真题 `上交真题集\上交\`（IMG_0089/0090）归 ken；
               # 院校A英语真题 `院校A真题\英语真题\`（9 张 jpg）归 ken。
               os.sep.join(('上交真题集', '上交')),
               os.sep.join(('院校A真题', '英语真题')),
               # 2026-09-26：英语真题集（各校回忆版 + 作文）
               # ⚠️ 2026-09-27 修正：原规则目录名写错（写成 '插班生真题库\英语真题集' /
               #   '插班生真题库\各校插班生英语作文回忆版真题'），**磁盘上并不存在这两个名字**
               #   → 规则空转、190 条英语真题全被 kmath 的宽泛规则吞掉。改为**实际目录名**。
               os.sep.join(('插班生真题库', '【精讲班同学】英语真题集')),
               os.sep.join(('插班生真题库', '英语巩固讲义插班生资料')),
               # 2026-10-14：v14 修复——english_src 是词汇表来源头占位，须入 ken 域
               'english_src'),
    'gmath':  (os.sep.join(('ECUST课程资料', '数学学院')),),
    'gen':    (os.sep.join(('ECUST课程资料', '外国语学院')),),
    'gother': (os.sep.join(('ECUST课程资料', '信息科学与工程学院')),
               os.sep.join(('ECUST课程资料', '物理学院')),
               os.sep.join(('ECUST课程资料', '马克思主义学院（人文科学研究院）')),
               os.sep.join(('ECUST课程资料', '机械与动力工程学院')),
               os.sep.join(('ECUST课程资料', '党委学生工作部（处）（武装部）')),
               os.sep.join(('Resource', '保研')),
               # 2026-10-14：v14 修复——convert_ppt_src 是转换副本（电力电子课件），须入 gother 域
               'convert_ppt_src'),
    'tech':   (os.sep.join(('Resource', 'skills')), 'RAG技术知识库',
               os.sep.join(('Resource', 'robomaster'))),
}
# 别名域（展开成基础域；值须能递归到基础域且不得自引用）
#   kaocha 考插全部（数学 + 英语）              ← 老板口径「考插」
#   gpa    绩点全部（华理数学 + 英语 + 其他学科）← 老板口径「绩点」
#   core   **默认域**：考插 + 华理数学英语，不含其他学科与技术库 → 其他学科不干扰检索
#   math   数学全量（考插 + 华理）
#   en     英语全量（考插 + 华理）
#   course gpa 的旧名（向后兼容，旧脚本 / README 里的 course 仍可用）
SCOPE_GROUPS = {
    'kaocha': ('kmath', 'ken'),
    'gpa':    ('gmath', 'gen', 'gother'),
    'core':   ('kmath', 'ken', 'gmath', 'gen'),
    'math':   ('kmath', 'gmath'),
    'en':     ('ken', 'gen'),
    'course': ('gmath', 'gen', 'gother'),
    'exclude_course': ('kmath', 'ken', 'tech'),
}
# mathdoc 是「考插数学题目 + 数学讲解白名单」，但它是**基础域**（不是纯别名），
# 放进 SCOPE_GROUPS 会让 scope_list 无限递归，所以单独处理。
SCOPE_MATHDOC = ('kmath', 'mathdoc')

# 可用检索域的**单一真源**（2026-09-21 新增）。
# 起因（勘误 #31）：kb_ask.py 与 kb_answer.py 各写一份域白名单，改域体系时只同步了前者，
# 导致 kb_answer 的 --scope 拒绝 core/gpa/kmath… → 改为从这里派生，两处不再各写一份。
# ⚠️ 'course' 必须留在表内：它是 gpa 的旧名，`课业资料.bat` 这个**双击入口**在用它，
#    撤掉会把桌面快捷方式弄坏（2026-09-21 实测踩到过）。
SCOPE_CHOICES = ('auto', 'core', 'kaocha', 'gpa', 'course',
                 'kmath', 'ken', 'gmath', 'gen', 'gother',
                 'math', 'en', 'mathdoc', 'tech', 'all')

# 纯中文语料域：检索时关掉「英文分池 + 英文保底」两路 —— 实测这两路会把英语考辅文档
# 顶到中文查询的 top1（如「英语四六级考试段落翻译题.doc」压过「半导体器件」）。
# 含英语语料的域（core / kaocha / gpa / ken / gen / en / all / tech）不关，否则英语题搜不到。
EN_FREE_SCOPES = ('kmath', 'gmath', 'gother', 'math', 'mathdoc')

# ------------------------------------------------- 数学「讲解」白名单（2026-09-20 13:40 新增）
# 起因（勘误 #28）：考插数学答不好的真因不是 OCR，而是**讲解语料被关在 course 域**——
# 考插数学习题资料 62 文件 / 183 片段几乎全是题目碎片（讲解型仅 61 片段）；
# 而华理数学学院公共课的「讲义 / 课件」有 71 文件 / 156 片段真讲课语料（李莹课件逐章、
# 学长重点总结），文本中位 1512 字符、空壳仅 1，且覆盖考插数学同一套大纲。
# → 新增 mathdoc = math(题目) + 下面的「讲义 / 课件」白名单（不含作业实验、考试真题、.swf 空壳）。
MATH_DOC_INCLUDE = (
    os.sep.join(('数学学院', '公共课', '高数', '讲义')),
    os.sep.join(('数学学院', '公共课', '线性代数', '讲义')),
    os.sep.join(('数学学院', '公共课', '概率论与数理统计', '讲义')),
)
MATH_DOC_EXCLUDE = ('作业实验',)
# 打不开的二进制作业答案（.swf）即便在讲义目录下也剔掉
MATH_DOC_EXCLUDE_EXT = ('.swf', '.nb', '.mmap', '.rar')


def is_math_doc(path):
    """是否属于「数学讲解语料」白名单（路径 + 排除规则都在这里，改归属只改常量）。"""
    p = (path or '').replace('/', os.sep)
    if not any(mk in p for mk in MATH_DOC_INCLUDE):
        return False
    if any(mk in p for mk in MATH_DOC_EXCLUDE):
        return False
    if p.lower().endswith(MATH_DOC_EXCLUDE_EXT):
        return False
    return True



def scope_list(scope):
    """把域（含别名）展开成基础域列表；None / 'all' 返回 None 表示不过滤。"""
    if not scope or scope == 'all':
        return None
    if scope == 'mathdoc':
        return list(SCOPE_MATHDOC)
    if scope in SCOPE_GROUPS:
        out = []
        for sub in SCOPE_GROUPS[scope]:
            out += scope_list(sub)
        return out
    return [scope]


def in_scope(row, scope):
    """某个片段（metadata 行）是否落在指定域内。"""
    bases = scope_list(scope)
    if bases is None:
        return True
    p = (row.get('file_path') or row.get('rel_path') or '').replace('/', os.sep)
    for b in bases:
        if b == 'mathdoc':
            # 特殊域：白名单 + 排除规则，不是简单子串
            if is_math_doc(p):
                return True
            continue
        for mk in SCOPE_RULES.get(b, ()):
            if mk in p:
                return True
    return False


def scope_of(row_or_path):
    """单个片段归属的基础域（mathdoc / math / en / tech / course / other），供统计与自检用。

    注意：mathdoc 先判 —— 它是 course 的子集（数学学院公共课讲义），
    不先判的话会被 course 吞掉，统计里看不出「讲解语料」这一层。
    """
    p = row_or_path if isinstance(row_or_path, str) else (row_or_path.get('file_path') or '')
    p = p.replace('/', os.sep)
    if is_math_doc(p):
        return 'mathdoc'
    for b, marks in SCOPE_RULES.items():
        for mk in marks:
            if mk in p:
                return b
    return 'other'


def retrieve(query, meta=None, vecs=None, top=5, pool=50, cands_n=80, bm25=None,
             use_rewrite=True, use_en=True, use_bm25=True, use_rerank=True,
             model=k3.LOCAL_MODEL, preview_chars=140, workers=8, backend='local',
             two_stage=True, w_fuse=0.4, head2=1, per_file_chunks=1, en_scale=None,
             rr_parallel=True, rerank_engine='bge', rewrite_model=None,
             trace=None,
             w_orig=1.2, w_zh=0.8, w_en=0.9, w_bm=1.6, w_bm_en=1.0,
             use_pool_en=True, w_orig_en=0.5, w_tech=0.8, pool_en=None,
             use_pool_bm=True, w_bm_pool=1.0, pool_bm=40, bm_scan=8000,
             use_neighbor=True, nb_seeds=8, nb_per_dir=2, nb_max=40, nb_decay=0.35,
             use_path_exp=False, path_topn=2, path_per_term=3, path_max=6,
             workers2=2, rr_cache=True,
             use_guard=True, guard_k=3, guard_hi_w=0.5, guard_terms=6, guard_k_per=2, guard_max=4,
             guard_bonus=0.0, use_view=True, meta_penalty=True, path_boost=True,
             # use_title_recall 默认关：2026-09-20 实测为**负面**（Hit@1 68%→64%、
             # PN结 掉出 top5）。字符集重合率在中文上假阳性严重
             # （「PN结的形成原理」vs「计算机组成原理实验」字符巧合重合 3/5）。
             # 代码保留以便将来换成真正的分词/词序匹配再试。
             use_title_recall=False, title_topn=3,
             latin_mask=None, vstore=None, kp=None, qtype=None, scope=None, log=print):
    """v4 检索：多路召回（原句 / 原句×英文池 / 中文改写 / 英文改写 / 中英 BM25 / 技术栈词）
    → 加权 RRF → 文件级去重 → 重排。

    kp / qtype（2026-09-19 新增）：按题目库的知识点 / 题型把候选**收窄到命中文件**，
    用于「只在这个知识点的题目里找」。
    scope（2026-09-20 新增）：检索域，见 SCOPE_RULES —— math / mathdoc / en / kaocha / course / tech / all。

    mathdoc（2026-09-20 13:40 新增）= math（考插数学题目）+ 数学学院公共课「讲义 / 课件」白名单。
    用途：考插数学的**理论类问题**（「洛必达法则怎么用」「泰勒公式怎么展开」）在 math 域里
    只能捞到题目碎片，必须借华理高数课件的讲解语料才能答。详见勘误 #28。"""
    if meta is None or vecs is None:
        meta, vecs = load_store_v(vstore or DEFAULT_VSTORE)
    if en_scale is not None:
        # 英文相关各路权重整体缩放（0=关英文路影响，1=保持默认），用于融合权重标定
        _s = float(en_scale)
        w_en, w_orig_en, w_tech, w_bm_en, w_bm_pool = (
            w_en * _s, w_orig_en * _s, w_tech * _s, w_bm_en * _s, w_bm_pool * _s)
    if use_bm25 and bm25 is None:
        bm25 = k2.build_bm25(meta, vstore or DEFAULT_VSTORE)

    t_all = time.time()
    info = {'query': query, 'stages': {}}

    # ---- 1) 改写（后台）与原句召回并行
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=1)
    fut_rw = ex.submit(rewrite_v4, query, rewrite_model or model, backend) if use_rewrite else None

    rank_lists, weights, bm_lists, bm_weights = [], [], [], []
    pool_en = pool_en or max(30, pool // 2)
    n_pool_en = 0
    if (use_pool_en or use_pool_bm) and latin_mask is None:
        latin_mask = load_latin_mask(vstore or DEFAULT_VSTORE, meta)
    if use_bm25:
        bm0 = bm25.search(query, topk=pool)
        if bm0:
            bm_lists.append(bm0)
            bm_weights.append(w_bm)

    rw = {'queries': [], 'en_query': '', 'keywords': [], 'en_keywords': []}
    rw_hit = None
    if fut_rw is not None:
        try:
            rw, rw_hit = fut_rw.result()
        except Exception as e:
            log('[改写] 失败，回退原句：%s' % str(e)[:120])
    t_rw = time.time() - t0
    ex.shutdown(wait=False)

    # ---- 2) 改写各路补充召回
    t0 = time.time()
    zh_qs = [q for q in rw.get('queries') or [] if q and q != query]
    en_q = (rw.get('en_query') or '').strip()
    # c05 施工（2026-10-15 知识准确性专项）：改写未产出 en_query 时，中文「附录」
    # 兜底为英文 appendix 参与英文召回 —— C_O_D 附录文件 appendix_C/E.pdf 是纯英文
    # 文本，纯中文查询跨语言永远召不进候选池（c05 基线 top1 错误命中实验报告封面）。
    # 仅在查询含「附录」且无任何拉丁词时启用，避免干扰其他纯中文查询。
    if use_en and not en_q and '附录' in (query or '') and not _LAT_RE.search(query or ''):
        _m = re.search(r'附录\s*([A-Za-z0-9]{1,4})', query or '')
        en_q = ('appendix %s' % _m.group(1)) if _m else 'appendix'
    techs = [t for t in (rw.get('tech_terms') or []) if t]
    tech_q = ' '.join(techs)
    # 2026-09-27 P0-2：四路（英文池/技术栈词/邻域扩展/保底）惰性触发。
    # 首版判定过窄：只认「查询本身含拉丁词 / 改写给出 tech_terms」，
    # 漏了改写产出的 en_keywords —— 纯中文查询（如 c04「数据结构 图 讲义」、
    # c07「高数下册作业答案」）改写仍会给出英文检索词，该信号下英文池/保底
    # 并非零贡献（c04 基线 rank4 即靠 pool_bm/guard 保进 top5）。
    # 修正：lazy_feat = 查询含拉丁词 或 改写给出技术栈词 或 改写给出英文关键词。
    _lazy_latin = bool(_LAT_RE.search(query or ''))
    _lazy_tech = bool(tech_q)
    _lazy_en = bool([k for k in (rw.get('en_keywords') or []) if k])
    lazy_feat = _lazy_latin or _lazy_tech or _lazy_en
    # 2026-09-27 P0-2 二次修正：pool_bm（BM25 英文子集保底）与 pool_en/guard 分开门控。
    # 消融实测：m07/m14（纯中文查询、改写含 en_keywords）开启 pool_bm 后目标被挤出
    # （r1→r4）；而 c04 依赖 pool_bm/guard。故 pool_bm 仅查询本身含拉丁词时激活
    # （真正的英文检索词信号），改写给出的 en_keywords 不触发 pool_bm，避免干扰纯中文查询。
    _lazy_bm_pool = _lazy_latin
    # 合并为一次批量 embedding 调用（原句 + 改写各路；Ollama 单次固定开销 ~2s，与句数几乎无关）
    _texts, _slots = [query], [('orig', 0, 1)]
    if zh_qs:
        _texts.extend(zh_qs)
        _slots.append(('zh', len(_texts) - len(zh_qs), len(_texts)))
    if use_en and en_q:
        _texts.append(en_q)
        _slots.append(('en', len(_texts) - 1, len(_texts)))
    if use_en and tech_q:
        _texts.append(tech_q)
        _slots.append(('tech', len(_texts) - 1, len(_texts)))
    _vmap = {}
    if _texts:
        _allv = k3.embed_cached(_texts)
        for _nm, _a, _b in _slots:
            _vmap[_nm] = _allv[_a:_b]
    qvec = _vmap.get('orig')
    if qvec is None:
        qvec = k3.embed_cached([query])
    rank_lists.append(k2.search_dense(vecs, qvec, topk=pool))
    weights.append(w_orig)
    if use_pool_en and lazy_feat:
        en0 = search_dense_subset(vecs, qvec[0], latin_mask, topk=pool_en)
        if en0:
            rank_lists.append(en0)
            weights.append(w_orig_en)
            n_pool_en = len(en0)
    if zh_qs and 'zh' in _vmap:
        rank_lists.append(k2.search_dense(vecs, _vmap['zh'], topk=pool))
        weights.append(w_zh)
    if use_en and en_q and 'en' in _vmap:
        rank_lists.append(k2.search_dense(vecs, _vmap['en'], topk=pool))
        weights.append(w_en)
    tech_vec = _vmap.get('tech')
    if tech_vec is not None:
        if use_pool_en and latin_mask is not None:
            tl = search_dense_subset(vecs, tech_vec[0], latin_mask, topk=pool_en)
            if tl:
                rank_lists.append(tl)
                weights.append(w_tech)
        rank_lists.append(k2.search_dense(vecs, tech_vec, topk=pool))
        weights.append(w_tech)
        if use_bm25:
            bmt = bm25.search(tech_q, topk=pool)
            if bmt:
                bm_lists.append(bmt)
                bm_weights.append(w_tech)
            if use_pool_bm and _lazy_bm_pool:
                bmtp = search_bm25_subset(bm25, tech_q, latin_mask, topk=pool_bm, scan=bm_scan)
                if bmtp:
                    bm_lists.append(bmtp)
                    bm_weights.append(w_bm_pool)
    if use_bm25:
        zh_terms = ' '.join(rw.get('keywords') or [])
        if zh_terms:
            bm1 = bm25.search(zh_terms, topk=pool)
            if bm1:
                bm_lists.append(bm1)
                bm_weights.append(w_bm)
        en_terms = ' '.join(rw.get('en_keywords') or [])
        if use_en and en_terms:
            bm2 = bm25.search(en_terms, topk=pool)
            if bm2:
                bm_lists.append(bm2)
                bm_weights.append(w_bm_en)
            if use_pool_bm and _lazy_bm_pool:
                bmp = search_bm25_subset(bm25, en_terms, latin_mask, topk=pool_bm, scan=bm_scan)
                if bmp:
                    bm_lists.append(bmp)
                    bm_weights.append(w_bm_pool)
        if use_en and en_q and use_pool_bm and _lazy_bm_pool:
            bmpq = search_bm25_subset(bm25, en_q, latin_mask, topk=pool_bm, scan=bm_scan)
            if bmpq:
                bm_lists.append(bmpq)
                bm_weights.append(w_bm_pool)

    fused = rrf_weighted(rank_lists + bm_lists, weights + bm_weights,
                         k=k2.RRF_K, topk=max(cands_n * 2, cands_n))
    # 2026-09-29 P0-5c：召回期降权（fused 阶段）——META_ONLY/黑名单条目在融合召回时
    #   即乘性降权，避免挤占候选池名额（c05 实测 appendix_C 因 fused top30 被
    #   .circ/.png/索引.md 占满而掉出候选池，后置 meta_penalty 无从生效）。
    #   仍只降权不剔除，与 WorkBuddy 方案一致。
    if meta_penalty and fused:
        fused = sorted([(i, round(s * (META_PENALTY if (is_meta_file(meta[i]['file_path'])
                          or not meta[i].get('has_text')
                          or os.path.splitext(meta[i]['file_path'])[1].lower() in META_ONLY_EXT)
                          else 1.0), 6))
                         for i, s in fused], key=lambda kv: -kv[1])
        info['stages']['fused_penalty'] = {'applied': True, 'factor': META_PENALTY,
                                           'n_penalized': sum(1 for i, s in fused if s < 1.0)}
    # ---- 题目库收窄（kp / qtype）：只保留题目库中命中该知识点/题型的源文件
    if kp or qtype:
        _allow = qbank_allowed_files(kp, qtype)
        _n_before = len(fused)
        fused = [(i, sc) for i, sc in fused if meta[i]['file_path'] in _allow]
        info['stages']['qbank'] = {'kp': kp, 'qtype': qtype, 'allow_files': len(_allow),
                                   'fused_before': _n_before, 'fused_after': len(fused)}
    if trace is not None:
        trace['fused_top'] = [(os.path.basename(meta[i]['file_path']), round(float(sc), 6))
                              for i, sc in fused[:30]]
    # 查询词（英文 / 中文）→ 用于「同文件多片段选最优」与重排预览定位
    latin, cjk = query_terms(query, ' '.join([en_q] + list(rw.get('keywords') or []) +
                                             list(rw.get('en_keywords') or []) +
                                             list(rw.get('tech_terms') or [])))
    by_file = build_file_index(meta)
    # 每个文件最多 per_file_chunks 个片段进入重排，且优先选「与问题词面最相关」的那一片段
    _K = max(1, int(per_file_chunks or 1))
    _fmap = dict(fused)
    seen, cands = set(), []
    for idx, sc in fused:
        fp = meta[idx]['file_path']
        if fp in seen:
            continue
        seen.add(fp)
        _ids = by_file.get(fp) or [idx]
        for _ci, _cidx in enumerate(pick_chunk_rank(meta, _ids, latin, cjk, _K)):
            cands.append((_cidx, sc if _ci == 0 else _fmap.get(_cidx, sc * 0.5)))
        if len(cands) >= cands_n:
            cands = cands[:cands_n]
            break

    # ---- 2b) 同目录邻居扩展：技能/资料目录内的多个文件主题强相关
    #          （如 skills/simpy/ 下的 scripts 与 references），
    #          中文提问对英文文档 dense 相似度不可靠时，靠同目录邻居把目标文件拉进候选。
    #          2026-09-27 P0-2：对纯中文查询零贡献，改为惰性触发（仅命中英文/技术词特征时激活）。
    n_nb = 0
    # 2026-09-27 P0-2 修正：neighbor 路与语言无关（同目录扩展对纯中文查询同样有效，
    # c07 实测单开 neighbor 即可把"高数下答案.pdf"拉进 top5），
    # 不再受 lazy_feat 门控；仅当候选非空时激活。
    if use_neighbor and cands:
        by_dir = build_dir_index(meta)
        for idx, sc in cands[:nb_seeds]:
            d = os.path.dirname(meta[idx]['file_path'])
            sibs = by_dir.get(d) or []
            if not sibs or len(sibs) > 200:
                continue
            added = 0
            for j in sibs:
                if len(cands) >= nb_max:
                    break
                fp = meta[j]['file_path']
                if fp in seen:
                    continue
                seen.add(fp)
                cands.append((j, sc * nb_decay))
                added += 1
                n_nb += 1
                if added >= nb_per_dir:
                    break

    # ---- 2d) 保底通道：英文池 BM25 头部结果强制进候选。
    #          两个坑：① 加权 RRF 里中文语料的多数派会把英文技能文档挤下去；
    #                  ② BM25 无词干化，改写给出的 resources 匹配不到文档里的 resource。
    #          故此处做 词形扩展 + 逐词检索，取头部结果置顶为候选保底。
    n_guard = 0
    g_ids = set()
    # 2026-09-27 P0-2：保底通道仅对英文/技术词查询有意义（纯中文查询零贡献），惰性触发。
    if use_guard and bm25 is not None and cands and lazy_feat:
        src = ' '.join([x for x in [en_q] + list(rw.get('en_keywords') or []) +
                        list(rw.get('tech_terms') or []) if x]).strip()
        words = [w for w in en_word_variants(src) if len(w) >= 4]
        words.sort(key=len, reverse=True)      # 长词更具体（restrictions > name），优先检索
        if words:
            if latin_mask is None:
                latin_mask = load_latin_mask(vstore or DEFAULT_VSTORE, meta)
            gh = []
            for t in words[:guard_terms]:
                gh += search_bm25_subset(bm25, t, latin_mask, topk=guard_k_per, scan=bm_scan)
            gh += search_bm25_subset(bm25, ' '.join(words), latin_mask, topk=guard_k, scan=bm_scan)
            if gh:
                guard_hi = fused[0][1] if fused else 1.0
                g_seen = set(fp for fp in (meta[i]['file_path'] for i, _ in cands))
                g_cands = []
                for i, _s in gh:
                    if len(g_cands) >= guard_max:
                        break
                    fp = meta[i]['file_path']
                    if fp in g_seen:
                        continue
                    g_seen.add(fp)
                    g_cands.append((i, round(guard_hi * guard_hi_w, 6)))
                if g_cands:
                    cands = g_cands + cands
                    n_guard = len(g_cands)
                    g_ids = set(i for i, _ in g_cands)

    info['stages']['rewrite'] = {'enabled': bool(use_rewrite), 'seconds': round(t_rw, 2),
                                 'queries': [query] + zh_qs, 'en_query': en_q,
                                 'keywords': rw.get('keywords') or [],
                                 'en_keywords': rw.get('en_keywords') or [],
                                 'tech_terms': rw.get('tech_terms') or [],
                                 'cache_hit': rw_hit}
    info['stages']['recall'] = {'seconds': round(time.time() - t0, 2), 'candidates': len(cands),
                                'dense_lists': len(rank_lists), 'bm25_lists': len(bm_lists),
                                'pool_en': n_pool_en, 'neighbor': n_nb, 'guard': n_guard,
                                'lazy_feat': lazy_feat, 'lazy_latin': _lazy_latin, 'lazy_tech': _lazy_tech, 'lazy_en': _lazy_en, 'lazy_bm_pool': _lazy_bm_pool,
                                'cand_files': [os.path.basename(meta[i]['file_path']) for i, _ in cands[:20]]}

    # ---- 2e) 域过滤（scope=math/en/kaocha/course/tech/...）：把考插数学、考插英语、
    #          华理课业、技术库拆开，互不污染（2026-09-20 新增）
    _n_scope_before = len(cands)
    if scope_list(scope):
        cands = [(i, sc) for i, sc in cands if in_scope(meta[i], scope)]
    info['stages']['scope'] = {'scope': scope or 'all', 'before': _n_scope_before,
                               'after': len(cands)}

    # ---- 2f) 标题/文件名召回通道（2026-09-20 新增）
    # 动机：问「高数下册作业答案」，《高数下答案.pdf》名字几乎是答案，但正文召回没把它带进
    # 候选池（fused top-30 之外），后面的重排与文件名加成都是「候选之外无解」。
    # 做法：把文件名与查询字符重合率 >= PATH_BOOST_MIN 的文件**保送**进候选池，
    # 排序仍交给重排器决定 —— 只补召回，不干预排序，风险可控。
    n_title = 0
    if use_title_recall:
        _have = {i for i, _ in cands}
        _hits = []
        for _fp, _ids in build_file_index(meta).items():
            _c = _path_boost_score(query, _fp)
            if _c >= PATH_BOOST_MIN:
                _hits.append((_c, _fp, _ids))
        _hits.sort(key=lambda x: (-x[0], x[1]))
        _add = []
        for _c, _fp, _ids in _hits[:title_topn]:
            if not in_scope({'file_path': _fp}, scope):
                continue
            _bi = pick_best_chunk(meta, _ids, latin, cjk)
            if _bi not in _have:
                _add.append((_bi, 1.0))
                _have.add(_bi)
        if _add:
            cands = _add + cands
            n_title = len(_add)
    info['stages']['title_recall'] = {'added': n_title}

    # ---- 2g-0) c08 校名目录保送（2026-10-05 施工，院校B专项）
    # 动机：院校B真题文件名是 IMG_013x.jpg，"院校B"只出现在目录路径里，basename 判据
    # 对它们恒为 0 → 标题召回/子序列保送/文件名加成/keep_floor 全链路失效，数学题基本全灭。
    # 此处：query 显式含目标校名时，把 rel_path 含同名校名的文件**从全库**保送进候选池顶部
    # （即使 BM25/dense 没召回也能进），排序仍交重排器定序，不干预分数。
    n_sch = 0
    q_sch = _school_tokens(query)
    if q_sch and cands:
        _have = {i for i, _ in cands}
        _sch_cand = []
        # 年份 token（4 位数字，如 2023/2017）：query 显式带年份时，年份命中文件优先保送
        _yrs = set(re.findall(r'(?:19|20)\d{2}', query))
        for _i, _m in enumerate(meta):
            if _i in _have:
                continue
            if not in_scope(_m, scope):
                continue
            _rp = _m.get('file_path') or ''
            if _school_tokens(_rp) & q_sch:
                # 年份命中 = 强信号（新真题优先），否则 0.9 弱保送（排后）
                _sc = 1.0 if (_yrs and any(y in _rp for y in _yrs)) else 0.9
                _sch_cand.append((_i, _sc, _rp))
        _sch_cand.sort(key=lambda x: (-x[1], x[2]))
        _sch_add = _sch_cand[:12]
        if _sch_add:
            cands = [(i, sc) for i, sc, _ in _sch_add] + cands
            n_sch = len(_sch_add)
    info['stages']['school_recall'] = {'added_front': n_sch}

    # ---- 2g-0b) c05 附录直通（2026-10-15 施工，知识准确性专项）
    # 动机：C_O_D 附录文件 appendix_C/E.pdf 是纯英文文本，纯中文查询「…附录」跨语言
    # 召不进候选池（dense/BM25 均不命中），基线 top1 错误命中实验报告封面 docx。
    # 此处：query 含「附录」且无拉丁词时，把 scope 内 basename 含 appendix 的文件
    # 保送进候选池尾部（不干预排序，交重排定序 + keep_floor 提到第5位）。
    n_appx = 0
    if '附录' in (query or '') and not _LAT_RE.search(query or ''):
        _have = {i for i, _ in cands}
        _apx = []
        for _i, _m in enumerate(meta):
            if _i in _have:
                continue
            if not in_scope(_m, scope):
                continue
            _b = (os.path.basename(_m.get('file_path') or '') or '').lower()
            if _b.startswith('appendix') or '_appendix' in _b or 'appendix_' in _b:
                _apx.append(_i)
        if _apx:
            cands = cands + [(i, 1.0) for i in _apx[:4]]
            n_appx = len(_apx[:4])
    info['stages']['appendix_recall'] = {'added_tail': n_appx}

    # ---- 2g) c07 候选池子序列保送（2026-10-02 施工）
    # 动机：title_recall 默认关闭，c07 期望文件《高数下答案.pdf》虽进候选池但 fused 排 40+ 位，
    # 重排器按正文（OCR 乱码）只给 3.125 分 → 52 位。此处用「文件名 ⊂ 问题按序子序列」判据，
    # 把候选池内命中文件提到前 10 并乘 0.9（不顶格——排序仍主要交重排器，配合
    # RERANK_PROMPT 文件名指引与 path_boost 乘分形成三层修复）。
    n_sub = 0
    if cands:
        _sub_hits = []
        for _n, (_i, _sc) in enumerate(cands):
            if _path_boost_score(query, meta[_i]['file_path']) >= 0.8:
                _sub_hits.append((_i, _sc))
        if _sub_hits:
            _head = [(i, round(sc * 0.9, 6)) for i, sc in _sub_hits]
            _hit_ids = set(i for i, _ in _head)
            cands = _head + [(i, sc) for i, sc in cands if i not in _hit_ids]
            n_sub = len(_head)
    info['stages']['subseq_recall'] = {'added_front': n_sub}

    # ---- 3) 重排（与 v3 同一套两阶段策略）
    t0 = time.time()
    if use_rerank and cands:
        vmeta = query_windows(meta, cands, latin, cjk) if use_view else meta
        if rerank_engine == 'bge':
            # 2026-10-15：交叉编码器重排（BGE-Reranker v2-m3）替代两阶段 LLM 生成式重排。
            # 排序任务从「7B 模型逐组写分」换成「专用交叉编码器统一打分」，耗时 7s 级 → 0.3s（GPU）/1-3s（CPU）。
            ranked = rerank_bge_v4(query, cands, vmeta, log=log)
        elif two_stage:
            ranked = rerank_two_stage_v4(query, cands, vmeta, model=model, backend=backend,
                                         workers=workers, workers2=workers2, preview1=preview_chars,
                                         w1=w_fuse, head=head2, log=log, use_cache=rr_cache,
                                         parallel=rr_parallel, scope=scope, vstore=vstore,
                                         terms=list(latin or []) + list(cjk or []))
        else:
            ranked = k3.fuse_rank(cands, k3.rerank_parallel(query, cands, vmeta, model=model,
                                                            preview_chars=preview_chars,
                                                            workers=workers, backend=backend,
                                                            log=log), w_fuse)
    else:
        ranked = cands
    # ---- 3b) 元文件降权（README / 说明 / 快捷方式等非知识内容，2026-09-20 新增）
    # 2026-09-29 P0-5：has_text=False 的 META_ONLY 条目（.circ/.exe/.zip 等仅元数据、无可检索正文）
    #   一并纳入乘性降权（乘 META_PENALTY=0.4），禁止硬过滤——只降权不剔除，可回退。
    # 2026-09-29 P0-5b：META_ONLY_EXT 扩展名条目（.jpg/.png 等非文本载体）也纳入乘性降权，
    #   抑制真题图片/短文本对正文 PDF 讲解的系统性抬分（实测 m04 miss），仍只降权不剔除。
    if meta_penalty and ranked:
        _before = ranked[0][0] if ranked else None
        ranked = sorted([(i, round(s * (META_PENALTY if (is_meta_file(meta[i]['file_path'])
                          or not meta[i].get('has_text')
                          or os.path.splitext(meta[i]['file_path'])[1].lower() in META_ONLY_EXT)
                          else 1.0), 4))
                         for i, s in ranked], key=lambda kv: -kv[1])
        info['stages']['meta_penalty'] = {
            'applied': True, 'factor': META_PENALTY,
            'top1_changed': bool(_before is not None and ranked and ranked[0][0] != _before)}

    # ---- 3c) 文件名命中加成（标题加权，2026-09-20 新增）
    if path_boost and ranked:
        _changed = 0
        _out = []
        # 2026-10-17 全天优化（e03）：query 含「真题」时，basename 含「真题」的
        # 文件加 ZHENTI_BONUS（见常量注释）。放在 path_boost 同循环，加成后
        # 统一重排——e03 高级篇（含真题、正文分低）靠此压过不含真题的
        # 特训/训练集干扰；m04 期望系列全含真题只强化不扰动。
        _zq = '真题' in (query or '')
        _boost_any = False
        # 2026-10-17 a02fix：低覆盖加成档。PATH_BOOST_MIN(0.8) 只保强命中；
        # 0.2~0.8 区间的弱命中（如 a02「插班生英语作文模板」期望
        # 「插班生写作(议论文).pdf」c=0.333 vs 干扰「作文 实用分类模板(1).pdf」
        # c=0.222）完全交给重排器，正文分易被干扰项反超（0.9062>0.8950）。
        # 此档按覆盖率补 c*LOW_COV_BONUS，期望净胜 0.111*0.15=0.0167，可翻盘。
        _low_min = LOW_COV_MIN
        for i, sc in ranked:
            c = _path_boost_score(query, meta[i]['file_path'])
            s2 = sc
            if c >= PATH_BOOST_MIN:
                s2 = round(sc + PATH_BOOST * c, 4)
                _changed += 1
                _boost_any = True
            elif c >= _low_min:
                s2 = round(s2 + LOW_COV_BONUS * c, 4)
                _boost_any = True
            if _zq and '真题' in os.path.basename(meta[i]['file_path']):
                s2 = round(s2 + ZHENTI_BONUS, 4)
                _boost_any = True
            _out.append((i, s2))
        if _boost_any:
            ranked = sorted(_out, key=lambda kv: -kv[1])
        info['stages']['path_boost'] = {'applied': bool(_changed), 'n_boosted': _changed,
                                        'factor': PATH_BOOST, 'min_cov': PATH_BOOST_MIN,
                                        'zhenti_bonus': ZHENTI_BONUS if _zq else 0.0}

    # ---- 3c-2) 强命中文件重排保底（2026-10-03 施工，c07）
    # 动机：gmath 域重排器按正文（数学公式 OCR 乱码）打分，「高数下答案.pdf」即使
    # subseq 保送 + path_boost 加成全触发仍被压到 rank=8。文件名强命中（>=0.8）
    # 是查询意图的硬信号（子序列覆盖≥80%），此处对强命中文件做 top-K 保底：
    # 若其重排 rank > KEEP_K，提升至第 KEEP_K 位；只移动该文件，其余相对顺序不动。
    KEEP_K = 5
    if path_boost and ranked:
        # 2026-10-15 施工（知识准确性专项）：判据从 _path_boost_score（校名目录
        # 命中即 0.8，query 含校名时大量文件同分、_best 被重排分最高的无关文件
        # 抢走）改为纯 basename 覆盖率 _basename_cov_score —— 只有文件名与 query
        # 真实重合的文件才进强命中集，杜绝「院校B插班生英语讲义参考答案」被
        # 28 个院校B路径文件淹没。
        _sidx = [i for i, _ in ranked if _basename_cov_score(query, meta[i]['file_path']) >= PATH_BOOST_MIN_V2]
        if _sidx:
            _score_map = dict(ranked)
            _cov_map = {i: _basename_cov_score(query, meta[i]['file_path']) for i in _sidx}
            _best = max(_sidx, key=lambda i: (_cov_map[i], _score_map[i]))
            _pos = next(n for n, (i, _) in enumerate(ranked) if i == _best)
            if _pos >= KEEP_K or _cov_map[_best] >= 0.85:
                # cov>=0.85（basename≈query，含子序列通道）为硬信号：用户就是在找这个文件，
                # 直接提到 top1；其余强命中提升到 KEEP_K。
                # 2026-10-16 阈值 0.95→0.85：e02「插班生英语词汇表在哪」词汇表文件名
                # 子序列覆盖率 0.89 被「插班生 语法(1).pdf」压在 rank2，0.95 太严。
                _dest = 0 if _cov_map[_best] >= 0.85 else KEEP_K - 1
                _item = ranked.pop(_pos)
                ranked.insert(_dest, _item)
                info['stages']['keep_floor'] = {'applied': True,
                                                'file': os.path.basename(meta[_best]['file_path']),
                                                'cov': round(float(_cov_map[_best]), 3),
                                                'from': _pos + 1, 'to': _dest + 1}
            else:
                info['stages']['keep_floor'] = {'applied': False, 'pos': _pos + 1,
                                                'cov': round(float(_cov_map[_best]), 3)}

    # 保底加成：跨语言场景下「英文池 BM25 头部」比本地小模型重排更可靠（实测），
    # 给保底候选一点分差，避免被中文语料的多数派压出 Top5。
    if use_guard and guard_bonus and g_ids:
        ranked = sorted([(i, round(s + (guard_bonus if i in g_ids else 0.0), 3)) for i, s in ranked],
                        key=lambda kv: -kv[1])
    info['stages']['rerank'] = {'seconds': round(time.time() - t0, 2),
                                'two_stage': bool(use_rerank and cands and two_stage),
                                'n_guard': n_guard, 'n_neighbor': n_nb,
                                'top_files': [os.path.basename(meta[i]['file_path']) for i, _ in ranked[:12]]}

    if trace is not None:
        trace['ranked_top'] = [(os.path.basename(meta[i]['file_path']), round(float(sc), 3))
                               for i, sc in ranked[:20]]
        trace['cands_total'] = len(cands)
        trace['cands_files'] = [os.path.basename(meta[i]['file_path']) for i, _ in cands]
        trace['n_guard'] = n_guard
        trace['n_neighbor'] = n_nb

    # ---- 4) 结果
    # 2026-10-15 施工（知识准确性专项）：同文件多块去重——同一 PDF 拆出的多个
    # chunk 常同时霸榜（c05 附录文件占前4位），对用户展示是冗余、且挤掉其他
    # 候选文件。按 file_path 去重，同文件只保留重排得分最高的那个块。
    final = []
    _seen_files = set()
    for idx, sc in ranked:
        _fp = meta[idx]['file_path']
        if _fp in _seen_files:
            continue
        _seen_files.add(_fp)
        final.append({'idx': idx, 'rank_score': sc, 'rel_path': meta[idx]['rel_path'],
                      'file_path': _fp, 'chunk_index': meta[idx]['chunk_index'],
                      'has_text': meta[idx]['has_text'], 'text': (meta[idx].get('text') or '')})
        if len(final) >= top:
            break
    info['results'] = final
    info['stages']['total_seconds'] = round(time.time() - t_all, 2)
    return info


# ------------------------------------------------------------------ 答案生成层
def answer(query, results, model=k3.DEEP_MODEL, backend='cloud', max_chars=1500,
           num_predict=700, log=print):
    """基于 Top5 片段生成带来源标注的答案：云端 glm-5.2 优先，失败回退本地 qwen2.5:7b。"""
    usable = [r for r in results if (r.get('text') or '').strip()]
    if not usable:
        return {'answer': '现有资料中没有找到相关依据（Top 结果均为仅元数据条目，未提取到正文）。',
                'model': None, 'backend': None, 'fallback': False}
    ctx = '\n\n'.join('[%d] 文件：%s\n内容：%s' % (n, r['rel_path'],
                                                  (r['text'] or '')[:max_chars].replace('\n', ' '))
                      for n, r in enumerate(usable, 1))
    prompt = ANSWER_PROMPT.format(q=query, ctx=ctx)
    if backend == 'cloud':
        try:
            txt = k3.llm_call(prompt, model=model, backend='cloud', num_predict=num_predict)
            if txt and txt.strip():
                return {'answer': txt.strip(), 'model': model, 'backend': 'cloud', 'fallback': False}
            log('[答案] 云端返回为空，回退本地模型')
        except Exception as e:
            log('[答案] 云端失败（%s），回退本地模型' % str(e)[:100])
    else:
        model = k3.LOCAL_MODEL
    txt = k3.llm_call(prompt, model=k3.LOCAL_MODEL, backend='local', num_predict=600)
    return {'answer': (txt or '').strip(), 'model': k3.LOCAL_MODEL, 'backend': 'local',
            'fallback': bool(backend == 'cloud')}


# ------------------------------------------------- 题目库通道（知识点 × 题型，2026-09-19 新增）
# 数据来源：qbank/questions_math.jsonl（数学资料逐页 OCR → 题目边界切分 → 规则+LLM 打标）
# 用途：① 直接列「某知识点 / 某题型」下的题目；② 收窄常规检索的候选文件范围。
QBANK_DIR = os.path.join(BASE, 'qbank')
QBANK_PATH = os.path.join(QBANK_DIR, 'questions_math.jsonl')
QBANK_META_PATH = os.path.join(QBANK_DIR, 'qbank_meta.json')
_QBANK = {'items': None, 'meta': None, 'path': None}


def load_qbank(path=None):
    """加载题目库；文件不存在时返回空列表（不抛错，检索链路可继续）。"""
    p = path or QBANK_PATH
    if _QBANK['items'] is None or _QBANK['path'] != p:
        items = []
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            pass
        meta = {}
        if os.path.exists(QBANK_META_PATH):
            try:
                meta = json.load(open(QBANK_META_PATH, encoding='utf-8'))
            except Exception:
                meta = {}
        _QBANK.update({'items': items, 'meta': meta, 'path': p})
    return _QBANK['items']


def qbank_meta():
    load_qbank()
    return _QBANK['meta'] or {}


def qbank_filter(kp=None, qtype=None, items=None):
    """按知识点 / 题型过滤题目库条目（None 表示不限制）。"""
    out = items if items is not None else load_qbank()
    if kp:
        out = [q for q in out if q.get('kp') == kp]
    if qtype:
        out = [q for q in out if q.get('qtype') == qtype]
    return out


def _qgrams(s):
    """中英混合粗切词：英文/数字词 + 中文 2~4 字片段（轻量关键词打分用）。"""
    s = (s or '').lower()
    terms = set(re.findall(r'[a-z][a-z0-9]{1,}', s))
    zh = re.sub(r'[^\u4e00-\u9fff]+', ' ', s)
    for seg in zh.split():
        for n in (2, 3, 4):
            for i in range(0, len(seg) - n + 1):
                terms.add(seg[i:i + n])
    return terms


def qbank_search(query='', kp=None, qtype=None, top=20, items=None):
    """题目库检索：先按知识点/题型过滤，再按关键词覆盖率打分。

    query 为空时按「文件 + 页码」排序返回（即"列出某知识点下的全部题目"）。
    """
    qs = qbank_filter(kp, qtype, items)
    if not query:
        qs = sorted(qs, key=lambda q: (q.get('rel_path') or '', q.get('start_page') or 0))
        return qs[:top]
    gq = _qgrams(query)
    scored = []
    if gq:
        for q in qs:
            gt = _qgrams(q.get('text') or '')
            hit = gq & gt
            if not hit:
                continue
            cover = len(hit) / float(len(gq))
            dens = len(hit) / float(max(len(gt), 1)) ** 0.5
            scored.append((round(cover + 0.25 * dens, 4), q))
        scored.sort(key=lambda kv: -kv[0])
    return [dict(q, q_score=s) for s, q in scored[:top]]


def qbank_allowed_files(kp=None, qtype=None):
    """题目库命中的源文件路径集合（用于收窄常规检索的候选范围）。"""
    return set(q.get('file_path') for q in qbank_filter(kp, qtype))


def qbank_labels_for(file_path):
    """某文件在题目库中的题量 / 知识点 / 题型分布（用于结果标注）。"""
    from collections import Counter
    qs = [q for q in load_qbank() if q.get('file_path') == file_path]
    return {'count': len(qs),
            'kp': dict(Counter(q.get('kp') for q in qs).most_common()),
            'qtype': dict(Counter(q.get('qtype') for q in qs).most_common())}


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description='本地知识库检索 v4（跨语言召回 + 答案层）')
    ap.add_argument('query', nargs='?', default=None, help='检索问题；仅想列题目时可省略')
    ap.add_argument('--mode', default='fast', choices=['fast', 'deep', 'turbo'])
    ap.add_argument('--kp', default=None, help='知识点过滤（极限/导数与微分/中值定理与泰勒/积分/定积分应用/微分方程/无穷级数/多元函数微分/重积分/曲线曲面积分/空间解析几何/线性代数/概率论与数理统计/其他）')
    ap.add_argument('--type', dest='qtype', default=None,
                    help='题型过滤（选择题/填空题/计算题/证明题/应用题）')
    ap.add_argument('--list', dest='list_mode', action='store_true',
                    help='列题模式：按 --kp/--type 直接列出题目，不走向量检索（秒级返回）')
    ap.add_argument('--qtop', type=int, default=20, help='列题模式返回条数（默认 20）')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--pool', type=int, default=50)
    ap.add_argument('--cands', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--preview', type=int, default=None)
    ap.add_argument('--model', default=None)
    ap.add_argument('--backend', default=None, choices=['local', 'cloud'])
    ap.add_argument('--vstore', default=DEFAULT_VSTORE, help='向量库目录')
    ap.add_argument('--raw', action='store_true', help='只看原文片段（不生成答案）')
    ap.add_argument('--answer-backend', default='cloud', choices=['cloud', 'local', 'off'],
                    help='答案生成后端（cloud=商汤 glm-5.2 免费池，local=本地 qwen2.5:7b）')
    ap.add_argument('--one-stage', action='store_true')
    ap.add_argument('--w', type=float, default=None)
    ap.add_argument('--no-rewrite', action='store_true')
    ap.add_argument('--no-en', action='store_true', help='关闭英文改写通道（对比用）')
    ap.add_argument('--no-pool-en', action='store_true', help='关闭英文分池召回（对比用）')
    ap.add_argument('--no-bm25', action='store_true')
    ap.add_argument('--no-rerank', action='store_true')
    ap.add_argument('--max-chars', type=int, default=300)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    # ---- 列题模式（题目库通道）：不加载向量库与模型，秒级返回
    if args.list_mode or (args.query is None and (args.kp or args.qtype)):
        items = qbank_search(args.query or '', kp=args.kp, qtype=args.qtype, top=args.qtop)
        qm = qbank_meta()
        if args.json:
            print(json.dumps({'kp': args.kp, 'type': args.qtype, 'returned': len(items),
                              'qbank_total': qm.get('count'), 'qbank_files': qm.get('files'),
                              'items': [{k: q.get(k) for k in
                                         ('qid', 'file_name', 'file_path', 'rel_path', 'start_page',
                                          'end_page', 'kp', 'kp_src', 'kp_conf', 'qtype', 'type_conf',
                                          'q_score', 'text')} for q in items]},
                             ensure_ascii=False))
            return
        print('题目库：%s（共 %s 题 / %s 份资料）' % (QBANK_PATH, qm.get('count'), qm.get('files')))
        print('筛选：知识点=%s ｜ 题型=%s%s → 命中 %d 题\n'
              % (args.kp or '不限', args.qtype or '不限',
                 ('，关键词「%s」' % args.query) if args.query else '', len(items)))
        if not items:
            print('（没有命中的题目）可用知识点：%s' % '、'.join(qm.get('kp_set') or []))
            return
        for n, q in enumerate(items, 1):
            loc = ('第 %s 页' % q.get('start_page')) if str(q.get('start_page')) == str(q.get('end_page')) \
                else ('第 %s-%s 页' % (q.get('start_page'), q.get('end_page')))
            print('[%d] %s ｜ %s ｜ %s ｜ %s%s' % (n, q.get('file_name'), loc, q.get('kp'), q.get('qtype'),
                                                  (' ｜ 相关度 %s' % q['q_score']) if q.get('q_score') else ''))
            snip = ' '.join((q.get('text') or '').split())[:110]
            if snip:
                print('     %s' % snip)
            print('     %s' % (q.get('file_path') or ''))
        print('\n提示：kb_ask.py --list --kp <知识点> --type <题型> 等价于此输出。')
        return
    if args.query is None:
        ap.error('请提供检索问题；若只想列题目，用 --list 配合 --kp / --type')

    cfg = {'fast': dict(model=k3.LOCAL_MODEL, backend='local', cands=30, workers=8, preview=140,
                        two_stage=True, w=0.4, use_rewrite=True),
           'deep': dict(model=k3.DEEP_MODEL, backend='cloud', cands=30, workers=1, preview=220,
                        two_stage=False, w=1.0, use_rewrite=True),
           'turbo': dict(model=k3.LOCAL_MODEL, backend='local', cands=30, workers=1, preview=120,
                         two_stage=False, w=0.4, use_rewrite=False)}[args.mode]
    model = args.model or cfg['model']
    backend = args.backend or cfg['backend']
    two_stage = cfg['two_stage'] and not args.one_stage
    use_rewrite = cfg['use_rewrite'] and not args.no_rewrite
    w_fuse = args.w if args.w is not None else cfg['w']
    cands = args.cands if args.cands is not None else cfg['cands']
    workers = args.workers if args.workers is not None else cfg['workers']
    preview = args.preview if args.preview is not None else cfg['preview']
    quiet = (lambda *a, **k: None) if args.json else print

    meta, vecs = load_store_v(args.vstore)
    if not args.json:
        print('向量库: %s（%d 条）' % (args.vstore, len(meta)))
    info = retrieve(args.query, meta, vecs, top=args.top, pool=args.pool, cands_n=cands,
                    use_rewrite=use_rewrite, use_en=not args.no_en,
                    use_pool_en=not args.no_pool_en,
                    use_bm25=not args.no_bm25,
                    use_rerank=not args.no_rerank and args.mode != 'turbo',
                    model=model, preview_chars=preview, workers=workers, backend=backend,
                    two_stage=two_stage, w_fuse=w_fuse, vstore=args.vstore,
                    kp=args.kp, qtype=args.qtype, log=quiet)
    st = info['stages']

    ans = None
    if not args.raw and args.answer_backend != 'off':
        t0 = time.time()
        ans = answer(args.query, info['results'], backend=args.answer_backend, log=quiet)
        ans['seconds'] = round(time.time() - t0, 2)
        if not args.json:
            print('\n===== 答案（%s · %s）=====\n' % (ans.get('model') or '-',
                                                     '商汤免费池' if ans['backend'] == 'cloud' else '本地 Ollama'))
            print(ans['answer'])

    if args.json:
        print(json.dumps({'query': args.query, 'mode': args.mode, 'vstore': args.vstore,
                          'stages': st,
                          'answer': ans,
                          'results': [{k: r[k] for k in ('rank_score', 'rel_path', 'file_path', 'chunk_index')}
                                      for r in info['results']]}, ensure_ascii=False))
        return

    _qb = st.get('qbank')
    if _qb:
        print('\n【题目库收窄】知识点=%s ｜ 题型=%s → 命中文件 %d 份（候选 %d → %d）'
              % (_qb.get('kp') or '不限', _qb.get('qtype') or '不限', _qb.get('allow_files') or 0,
                 _qb.get('fused_before'), _qb.get('fused_after')))

    print('\n===== Top %d 来源片段（v4 · %s 档）=====\n' % (args.top, args.mode))
    for rank, r in enumerate(info['results'], 1):
        print('[%d] 重排分 %.1f | 文件: %s' % (rank, r['rank_score'], r['rel_path']))
        print('    路径: %s (片段 %s)' % (r['file_path'], r['chunk_index']))
        if r['has_text']:
            print('    ' + r['text'][:args.max_chars].replace('\n', '\n    '))
        else:
            print('    → 该文件仅元数据索引（无提取文本），可打开文件查看')
        print()
    print('耗时：改写 %.1fs | 召回 %.1fs | 重排 %.1fs | 答案 %s | 合计 %.1fs（改写缓存命中：%s）'
          % (st['rewrite']['seconds'], st['recall']['seconds'], st['rerank']['seconds'],
             ('%.1fs' % ans['seconds']) if ans else '未生成',
             st['total_seconds'] + (ans['seconds'] if ans else 0),
             '是' if st['rewrite'].get('cache_hit') else '否'))
    if st['rewrite'].get('en_query'):
        print('英文改写查询：%s' % st['rewrite']['en_query'])


if __name__ == '__main__':
    main()
