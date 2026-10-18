# -*- coding: utf-8 -*-
"""本地知识库 · 问答入口（模范 RAG：Agentic 多轮检索 + 带引用答案）

和 kb_ask.py 的区别：
    kb_ask.py     给你「5 条相关片段」，你自己读
    kb_answer.py  给你「一段能直接看的答案 + 来源编号」，并附上原始片段备查

流程（对标 2024-2025 主流 RAG 的标准做法）：
    1) 第 1 轮检索：改写多查询 → 混合召回(BM25+向量) → 域过滤 → 两阶段重排
    2) 证据体检（不看 LLM，纯启发式，确定性、零成本）
    3) 证据不足 → 第 2 轮 Agentic 升级（最多 3 档，逐档加重）：
       档 A 换召回方式：关掉改写与重排，纯召回一次（救被重排误杀的候选）
       档 B 扩检索域    ：kmath→mathdoc→kaocha→core→gpa→all（每轮跳一格）
                           （mathdoc = math + 数学讲解语料，2026-09-20 加，详见勘误 #28；
                             链补全见 2026-09-21 勘误 #31）
       档 C 同文件深挖  ：在第 1 轮命中的最佳文件里，按查询词再挑更多块
    4) 两轮结果按「文件+块号」去重合并，保留最高分
    5) 生成 grounded 答案：云端 glm-5.2 → 失败回退本地 qwen2.5:7b → 再失败只给片段
    6) 引用校验：答案里越界的 [编号] 一律剔除；正文无引用则补「来源清单」
    7) 落答案缓存（对抗商汤免费池 429 限流；同问同上下文直接复用）

用法：
    py -3.12 -X utf8 kb_answer.py "电子的能带结构是怎么形成的"
    py -3.12 -X utf8 kb_answer.py "极限的定义" --scope math
    py -3.12 -X utf8 kb_answer.py "四级段落翻译技巧" --scope en --json
    py -3.12 -X utf8 kb_answer.py "PN结的形成原理" --scope gother --no-answer   # 只要检索，不要答案
"""
import os
import sys
import json
import time
import hashlib
import argparse

KB = os.path.dirname(os.path.abspath(__file__))
if KB not in sys.path:
    sys.path.insert(0, KB)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import kb_search_v2 as k2      # noqa: E402
import kb_search_v3 as k3      # noqa: E402
import kb_search_v4 as k4      # noqa: E402

VSTORE = k4.DEFAULT_VSTORE
# 可用检索域的单一真源（2026-09-21 修，勘误 #31）：原来本文件自己写死一份旧 8 域，
# 与 kb_ask.py 的 14 域不一致，导致 --scope core 被 argparse 拒绝。
SCOPE_CHOICES = k4.SCOPE_CHOICES

# 进程内缓存：交互模式下只加载一次索引与 BM25（否则每问一句都重新加载）
_STORE = {}


def _load(vstore):
    if vstore not in _STORE:
        meta, vecs = k4.load_store_v(vstore)
        _STORE[vstore] = (meta, vecs, k2.build_bm25(meta, vstore))
    return _STORE[vstore]

# 答案缓存：同问 + 同上下文 + 同模型 → 直接复用，避免反复打云端（免费池限流严重）
_ANS_CACHE = k3.JsonCache(os.path.join(k3.CACHE_DIR, 'answer_cache.json'), limit=2000)
import atexit                # noqa: E402
atexit.register(_ANS_CACHE.flush)

# --------------------------------------------------------------- 提示词
GEN_PROMPT = '''你是老板的私人学习助理，只依据下面给出的「资料片段」回答问题。

硬规则（违反即作废）：
1. 只允许使用片段中确实出现的信息。片段里没有的，一律不许补充、不许凭常识推断。
2. 每个结论后面必须标注来源编号，形如 [1] 或 [1][3]，编号对应片段前的 [n]。
3. 如果片段不足以回答，**必须直说**「资料里没有找到相关依据」，并指出还缺什么，不要硬编。
4. 如果片段之间相互矛盾，把矛盾点指出来，不要替它们统一口径。
5. 用简体中文，直接给结论；可以用短列表；不要复述问题，不要写"根据提供的资料"这类废话开头。
6. 总长度控制在 500 字以内；**宁可少说也要说完**：最多列 3 条，每条一句话。公式用普通文本写法（如 x^2、lim、∑），不要用 LaTeX。
7. 结尾必须是完整句子（以句号结束），不要留下半句话。

用户问题：{q}

资料片段：
{ctx}

现在输出答案（每个结论带 [编号]）：'''

VERIFY_PROMPT = '''下面是一份「问答答案」和它引用的资料片段。请检查答案是否严格依据片段。

只输出 JSON，不要任何解释：
{{"ungrounded": ["答案里没有片段依据的句子，逐句列出，最多3条"],
  "verdict": "ok 或 partially_unsupported 或 unsupported"}}

答案：
{a}

片段：
{ctx}
只输出 JSON：'''


# --------------------------------------------------------------- 工具
def _parse_json_loose_v(txt):
    """宽松解析自校验模型输出（容忍代码块围栏、中文引号、前后废话）。

    复用 k4._parse_json_loose 的宽容度，但这里只需要 dict，且要能捞到
    `verdict` / `ungrounded` 两个键；解析失败一律返回 {}（调用方据此跳过提示）。
    """
    try:
        d = k4._parse_json_loose(txt)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    out = {}
    t = txt or ''
    m = re.search(r'"verdict"\s*:\s*"([^"]+)"', t)
    if m:
        out['verdict'] = m.group(1).strip()
    m2 = re.search(r'"ungrounded"\s*:\s*\[(.*?)\]', t, re.S)
    if m2:
        out['ungrounded'] = [x.strip().strip('"\'') for x in m2.group(1).split(',') if x.strip().strip('"\'')]
    return out


def _terms(q):
    """查询词（中文 2-4 字片段 + 英文/数字词），用于证据体检与同文件深挖。

    用 k4._qgrams 而不是 k4.query_terms —— 后者按标点切成 2-6 字长块，
    要整块命中才算覆盖，判据过严（几乎永远判「证据不足」）。
    """
    try:
        return set(t for t in k4._qgrams(q) if len(t) >= 2)
    except Exception:
        latin, cjk = k4.query_terms(q)
        return set(latin) | set(cjk)


def _evidence(results, q):
    """证据体检：返回 dict（usable / best / cover / sufficient）。纯启发式，不调 LLM。

    判据用「词面覆盖」为主、分数为辅 —— 实测发现高分≠答对：
    问「电子的能带结构」，本地 7B 重排能给示波器实验报告打 7.89 分，
    但该文档对查询词的覆盖率只有 9%，实际是错的。
    """
    usable = [r for r in results if len((r.get('text') or '').strip()) >= 30]
    best = max([r.get('rank_score') or 0.0 for r in results], default=0.0)
    terms = _terms(q)
    cover = 0.0
    if terms:
        pool = ' '.join((r.get('text') or '')[:600] for r in results[:3]).lower()
        hit = sum(1 for t in terms if t.lower() in pool)
        cover = hit / max(1, len(terms))
    sufficient = (len(usable) >= 2) and (cover >= 0.28 or best >= 8.5)
    return {'usable': len(usable), 'best': round(best, 3), 'cover': round(cover, 3),
            'sufficient': bool(sufficient)}


# 扩域链（2026-09-21 副驾修，勘误 #31）：**必须有「数学讲解」这一跳**
#
# 背景：09-20 22:40 域体系 v2 重构后，_widen 仍是 v1 时代的窄链，kmath 直接跳 all，
# 中间**跳过了 mathdoc**。后果 = 勘误 #28 当年的病复发：考插数学的理论类问题
#（「中值定理怎么证明」）在 kmath 域只能捞到题目碎片，一跳 all 后 8329 片段里
# 讲义被噪声淹没。实测：kmath 域 top5 无一条讲义；mathdoc 域 top1 即
# Ch3-2微分中值定理.pdf（李莹课件）。
#
# 新链（按老板 09-21 口径）：
#   kmath   → mathdoc   考插数学题目 → 加「数学讲解」白名单（**关键一跳**）
#   mathdoc → kaocha    讲解 → 考插全量（数学 + 英语）
#   kaocha  → core      考插 → 考插 + 华理数学英语
#   core    → gpa       核心 → 加其他学科（物理/信息/马原/机动）
#   gpa     → all       绩点 → 全库（含 tech 技能技术库）
#   math/en/tech 等不分科目的域 → 直接进 core（科目内已无更多可扩）
_WIDEN_NEXT = {
    'kmath':   'mathdoc',
    'mathdoc': 'kaocha',
    'kaocha':  'core',
    'core':    'gpa',
    'gpa':     'all',
    # 不分科目的域：一跳进 core（gpa 已含 gother，会稀释数学/英语查询）
    'math': 'core',
    'en':   'core',
    'ken':  'core',
    'gmath': 'core',
    'gen':   'core',
    'tech':  'core',
    # 其他学科本身已是终点候选，直接全库
    'gother': 'all',
    # 旧域向后兼容
    'course': 'all',
    'auto':   'all',
}


def _widen(scope):
    """扩域链的**下一跳**（第 3 轮只能跳一格，多轮才走完整条链）。

    2026-09-21 修（勘误 #31）：补回「数学讲解」跳，链改为
    kmath → mathdoc → kaocha → core → gpa → all（表见 _WIDEN_NEXT）。
    旧链 kmath → all 会跳过 mathdoc，使考插数学理论问题拿不到讲义语料。
    """
    return _WIDEN_NEXT.get(scope or 'all', 'all')


def _deepen(results, meta, vecs, q, per_file=2, log=print):
    """同文件深挖：在已命中文件里，按查询词再挑若干块（small-to-big 的便宜版）。"""
    terms = _terms(q)
    if not terms or not results:
        return []
    out = []
    seen_files = []
    for r in results[:2]:
        fp = r.get('file_path')
        if fp and fp not in seen_files:
            seen_files.append(fp)
    for fp in seen_files:
        cand = [i for i, m in enumerate(meta) if m.get('file_path') == fp]
        scored = []
        for i in cand:
            txt = (meta[i].get('text') or '')
            if len(txt.strip()) < 30:
                continue
            low = txt[:600].lower()
            s = sum(1 for t in terms if t.lower() in low)
            if s:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        for s, i in scored[:per_file]:
            if not any(r.get('idx') == i for r in results):
                out.append({'idx': i, 'rank_score': 0.0, 'rel_path': meta[i]['rel_path'],
                            'file_path': meta[i]['file_path'],
                            'chunk_index': meta[i].get('chunk_index'), 'text': meta[i].get('text') or '',
                            'from': 'deepen'})
    if out:
        log('  [深挖] 在 %d 个已命中文件里补到 %d 块' % (len(seen_files), len(out)))
    return out


def _merge(*groups):
    """按 (文件, 块号) 去重合并，同一条取最高分。"""
    best = {}
    for g in groups:
        for r in g or []:
            key = (r.get('file_path'), r.get('chunk_index'))
            if key not in best or (r.get('rank_score') or 0) > (best[key].get('rank_score') or 0):
                best[key] = r
    return sorted(best.values(), key=lambda r: -(r.get('rank_score') or 0))


def _mk_ctx(results, max_chars=1500, n=6):
    rows = [r for r in results if len((r.get('text') or '').strip()) >= 30][:n]
    ctx = '\n\n'.join('[%d] 文件：%s（第 %s 段）\n内容：%s'
                      % (i, r['rel_path'], r.get('chunk_index'),
                         (r.get('text') or '')[:max_chars].replace('\n', ' '))
                      for i, r in enumerate(rows, 1))
    return rows, ctx


def _fix_citations(ans, n_max):
    """引用校验：剔除越界的 [编号]（模型偶尔编 7、8 这种不存在的编号）。"""
    import re
    bad = []
    for m in re.finditer(r'\[(\d{1,2})\]', ans):
        if int(m.group(1)) > n_max or int(m.group(1)) < 1:
            bad.append(m.group(0))
    for b in set(bad):
        ans = ans.replace(b, '')
    return ans, sorted(set(bad))


# --------------------------------------------------------------- 主流程
def ask(q, scope='auto', top=6, backend='cloud', agentic=True, want_answer=True,
        vstore=VSTORE, log=print, verify=True):
    t_all = time.time()
    meta, vecs, bm25 = _load(vstore)
    _scope = scope
    # 用户显式指定了域 = 「我只要这些资料」→ 不许自动扩域，否则会莫名跑出别的域的东西
    allow_widen = (scope == 'auto')
    if _scope == 'auto':
        import kb_ask as ka
        _scope = ka.auto_scope(q)

    rounds = []

    # ---- 第 1 轮：完整链路（改写 + 重排）
    t0 = time.time()
    kw = dict(use_rewrite=True, use_rerank=True, scope=_scope)
    if _scope in k4.EN_FREE_SCOPES:
        kw.update(use_en=False, use_pool_en=False)
    r1 = k4.retrieve(q, meta=meta, vecs=vecs, bm25=bm25, top=top, log=log, **kw)
    res1 = r1['results']
    ev1 = _evidence(res1, q)
    rounds.append({'round': 1, 'scope': _scope, 'profile': 'full(llm改写+重排)',
                   'seconds': round(time.time() - t0, 1), 'evidence': ev1,
                   'top': [os.path.basename(x['file_path']) for x in res1[:3]]})
    log('  第1轮(%s｜完整链路)：候选证据 usable=%d best=%.2f 覆盖=%.0f%% 用时%.1fs'
        % (_scope, ev1['usable'], ev1['best'], ev1['cover'] * 100, rounds[-1]['seconds']))

    res = res1
    if agentic and not ev1['sufficient']:
        log('  证据不足 → 进入 Agentic 升级')

        # ---- 第 2 轮 · 档 A：换召回方式（纯召回，救被重排误杀的候选）
        t0 = time.time()
        r2 = k4.retrieve(q, meta=meta, vecs=vecs, bm25=bm25, top=top,
                         use_rewrite=False, use_rerank=False, scope=_scope, log=lambda *a, **k: None)
        res = _merge(res1, r2['results'])
        ev2 = _evidence(res, q)
        rounds.append({'round': 2, 'scope': _scope, 'profile': 'plain(纯向量+BM25)',
                       'seconds': round(time.time() - t0, 1), 'evidence': ev2,
                       'top': [os.path.basename(x['file_path']) for x in res[:3]]})
        log('  第2轮(档A 纯召回)：usable=%d best=%.2f 覆盖=%.0f%% → 合并 %d 条'
            % (ev2['usable'], ev2['best'], ev2['cover'] * 100, len(res)))

        # ---- 第 3 轮 · 档 B：扩检索域（仅 auto 模式；显式指定域时不动，尊重用户意图）
        if not ev2['sufficient'] and allow_widen:
            wide = _widen(_scope)
            if wide != _scope:
                t0 = time.time()
                kw3 = dict(use_rewrite=True, use_rerank=True, scope=wide)
                # 纯中文语料域：关英文分池与英文保底（沿用 kb_search_v4.EN_FREE_SCOPES 口径，
                # 不再硬编码 —— 2026-09-21 修，原来只判 ('course','math','mathdoc')，
                # 新的 kmath/kaocha/core 走不到这里，英文考辅文档会顶掉中文结果）
                if wide in k4.EN_FREE_SCOPES:
                    kw3.update(use_en=False, use_pool_en=False)
                r3 = k4.retrieve(q, meta=meta, vecs=vecs, bm25=bm25, top=top,
                                 log=lambda *a, **k: None, **kw3)
                res = _merge(res, r3['results'])
                ev3 = _evidence(res, q)
                rounds.append({'round': 3, 'scope': wide, 'profile': 'widen(扩域)',
                               'seconds': round(time.time() - t0, 1), 'evidence': ev3,
                               'top': [os.path.basename(x['file_path']) for x in res[:3]]})
                log('  第3轮(档B 扩域 %s→%s)：usable=%d best=%.2f 覆盖=%.0f%% → 合并 %d 条'
                    % (_scope, wide, ev3['usable'], ev3['best'], ev3['cover'] * 100, len(res)))

        # ---- 档 C：同文件深挖
        deep = _deepen(res, meta, vecs, q, log=log)
        if deep:
            res = _merge(res, deep)

    res = res[:max(top, 8)]
    ev = _evidence(res, q)
    used_rows, ctx = _mk_ctx(res)

    out = {'query': q, 'scope': _scope, 'agentic': bool(agentic and len(rounds) > 1),
           'rounds': rounds, 'evidence': ev, 'elapsed': 0.0,  # 末尾统一回填
           'results': res, 'answer': None, 'answer_model': None,
           'answer_backend': None, 'citations_fixed': [], 'hits_used': len(used_rows)}

    if not want_answer:
        return out

    if not used_rows:
        out['answer'] = '资料里没有找到相关依据（命中的条目都没有可用正文）。'
        out['answer_backend'] = 'none'
        out['elapsed'] = round(time.time() - t_all, 1)
        return out

    # ---- 答案缓存
    # 缓存盐：改了提示词 / 答案规则就 +1（与重排缓存的 RR_SALT 同一个道理）
    ANS_SALT = 'ans-v7'    # 2026-09-27：SCOPE_RULES 修正插班生真题库英/数分域（与 RR_SALT scope-v12 同步）
    sig = hashlib.md5(('%s|%s|%s|%s|%d|' % (ANS_SALT, q, _scope, backend, len(used_rows))).encode('utf-8'))
    sig.update(('|'.join('%s#%s' % (r.get('file_path'), r.get('chunk_index')) for r in used_rows))
               .encode('utf-8'))
    sig.update(('|'.join(hashlib.md5((r.get('text') or '')[:400].encode('utf-8')).hexdigest()
                         for r in used_rows)).encode('utf-8'))
    ckey = sig.hexdigest()
    hit = _ANS_CACHE.get(ckey)
    if isinstance(hit, dict) and hit.get('answer'):
        out['answer'] = hit['answer']
        out['answer_model'] = hit.get('model')
        out['answer_backend'] = (hit.get('backend') or '') + '(cached)'
        log('  [答案] 命中缓存')
        return out

    # ---- 生成答案：云端 → 本地 → 只给片段
    prompt = GEN_PROMPT.format(q=q, ctx=ctx)
    ans, mdl, bkd = None, None, None
    if backend == 'cloud':
        t0 = time.time()
        try:
            ans = k3.llm_call(prompt, model=k3.DEEP_MODEL, backend='cloud',
                              retries=2, num_predict=3000)
            mdl, bkd = k3.DEEP_MODEL, 'cloud'
            log('  [答案] 云端 %s 生成完成（%.1fs）' % (mdl, time.time() - t0))
        except Exception as e:
            log('  [答案] 云端失败（%s）→ 回退本地' % str(e)[:90])
    if not ans:
        try:
            import kb_ask as ka
            if ka.ensure_ollama(wait=20):
                t0 = time.time()
                ans = k3.llm_call(prompt, model=k3.LOCAL_MODEL, backend='local', num_predict=900)
                mdl, bkd = k3.LOCAL_MODEL, 'local'
                log('  [答案] 本地 %s 生成完成（%.1fs）' % (mdl, time.time() - t0))
        except Exception as e:
            log('  [答案] 本地也失败：%s' % str(e)[:90])
    if not ans:
        out['answer'] = None
        out['answer_backend'] = 'none'
        out['note'] = '云端与本地模型都不可用（免费池限流 / Ollama 未启动），下面是原始片段，请自行阅读。'
        out['elapsed'] = round(time.time() - t_all, 1)
        return out

    ans, fixed = _fix_citations(ans, len(used_rows))
    truncated = bool(ans) and ans.rstrip()[-1:] not in '。！？；）】》.!?'
    out.update({'answer': ans, 'answer_model': mdl, 'answer_backend': bkd,
                'citations_fixed': fixed, 'answer_truncated': truncated})

    # ---- 答案自校验（改进 #3，2026-09-22 接线）
    # 背景：VERIFY_PROMPT 早已写好但从未被调用（死代码）。RAGAS 一类框架把
    #       Faithfulness（答案是否严格来自片段）列为生成侧头号指标，这里补上。
    # 策略：**只提示、不替换**——把「无依据的句子」回显给老板，由人判断；
    #       绝不用校验结果去改写答案（小模型改写风险大于收益）。
    # 成本控制：仅云端生成时启用（本地模型自校验又慢又不可靠），且失败静默。
    if verify and bkd == 'cloud' and ans:
        try:
            t0 = time.time()
            vtxt = k3.llm_call(VERIFY_PROMPT.format(a=ans[:1200], ctx=ctx[:2400]),
                               model=k3.DEEP_MODEL, backend='cloud', retries=1, num_predict=400)
            vd = _parse_json_loose_v(vtxt)
            verdict = str(vd.get('verdict') or '').strip()
            ungrounded = [str(x).strip() for x in (vd.get('ungrounded') or []) if str(x).strip()][:3]
            if verdict and verdict != 'ok':
                out['verify'] = {'verdict': verdict, 'ungrounded': ungrounded}
                log('  [自校验] %s，疑似无依据 %d 句（%.1fs）'
                    % (verdict, len(ungrounded), time.time() - t0))
            elif verdict == 'ok':
                out['verify'] = {'verdict': 'ok', 'ungrounded': []}
                log('  [自校验] ok（%.1fs）' % (time.time() - t0))
        except Exception as e:
            log('  [自校验] 跳过（%s）' % str(e)[:70])

    out['elapsed'] = round(time.time() - t_all, 1)
    try:
        _ANS_CACHE.set(ckey, {'answer': ans, 'model': mdl, 'backend': bkd})
    except Exception:
        pass
    return out


def render(out, top=6):
    lines = []
    lines.append('【问题】%s    ｜ 检索域 %s ｜ 用时 %.1f 秒'
                 % (out['query'], out['scope'], out['elapsed']))
    if out.get('agentic'):
        lines.append('  [Agentic] 跑了 %d 轮：' % len(out['rounds']))
        for r in out['rounds']:
            lines.append('     第%d轮 %-24s %-28s 用时%.1fs  best=%.2f 覆盖=%.0f%%'
                         % (r['round'], r['profile'], r['scope'], r['seconds'],
                            r['evidence']['best'], r['evidence']['cover'] * 100))
    if out.get('citations_fixed'):
        lines.append('  [引用校验] 已剔除越界编号：%s' % ' '.join(out['citations_fixed']))
    _v = out.get('verify') or {}
    if _v.get('verdict') and _v['verdict'] != 'ok':
        _tag = {'partially_unsupported': '部分语句缺依据',
                'unsupported': '整体缺依据'}.get(_v['verdict'], _v['verdict'])
        lines.append('  [答案自校验] %s —— 以下句子在片段中找不到依据，请自行判断：' % _tag)
        for _s in (_v.get('ungrounded') or [])[:3]:
            lines.append('       · %s' % _s[:110])
    if not (out.get('evidence') or {}).get('sufficient') and not out.get('agentic'):
        lines.append('  [提示] 本地证据偏弱；若确定资料库里有，可换 --scope auto 让检索自动扩域')
    elif not (out.get('evidence') or {}).get('sufficient') and out.get('scope') != 'all':
        lines.append('  [提示] 跑了多轮仍偏弱；显式指定域时不会自动扩域，可试 --scope auto 或 --scope all')
    lines.append('')
    if out.get('answer'):
        lines.append('─' * 62)
        lines.append(out['answer'])
        lines.append('─' * 62)
        if out.get('answer_truncated'):
            lines.append('  ！答案被长度上限截断（可加 --backend local 或重问一次）')
        lines.append('  （答案由 %s 生成｜%s）' % (out.get('answer_model'), out.get('answer_backend')))
    else:
        lines.append('（本次未生成答案%s）' % ('：' + out['note'] if out.get('note') else ''))
    lines.append('')
    lines.append('【依据片段】')
    for i, r in enumerate(out['results'][:top], 1):
        tag = '（深挖补充）' if r.get('from') == 'deepen' else ''
        lines.append('  [%d] %s  第%s段  相关度 %.2f %s'
                     % (i, r['rel_path'], r.get('chunk_index'), r.get('rank_score') or 0, tag))
        txt = (r.get('text') or '').strip().replace('\n', ' ')
        if txt:
            lines.append('      %s' % txt[:180])
    return '\n'.join(lines)


HELP = """
可用操作：
  直接输入问题 + 回车     给出一段带 [编号] 引用的答案 + 依据片段
  s <域>                 切换检索域（可选：%s）
  m cloud|local          切换答案模型（cloud=云端 glm-5.2，local=本地 qwen2.5:7b）
  r <问题>               只检索、不生成答案
  h                      本帮助
  q                      退出
""" % ' / '.join(SCOPE_CHOICES)


def interactive(scope, top, backend):
    print('\n本地知识库问答已就绪 —— 答案模型 %s ｜ 检索域 %s' % (backend, scope))
    print('直接输入问题回车；h 帮助，q 退出。\n')
    while True:
        try:
            q = input('问> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n已退出。')
            return
        if not q:
            continue
        if q.lower() in ('q', 'exit', 'quit'):
            print('已退出。')
            return
        if q.lower() == 'h':
            print(HELP)
            continue
        if q.lower().startswith('s ') or q.startswith('域'):
            v = (q.split() + [''])[1].strip().lower() if q.lower().startswith('s ') else q[1:].strip()
            if v in SCOPE_CHOICES:
                scope = v
                print('已切换检索域 → %s' % scope)
            else:
                print('检索域只能选 %s' % ' / '.join(SCOPE_CHOICES))
            continue
        if q.lower().startswith('m '):
            v = q.split()[1].strip().lower()
            if v in ('cloud', 'local'):
                backend = v
                print('已切换答案模型 → %s' % backend)
            else:
                print('答案模型只能选 cloud / local')
            continue
        want = True
        if q.lower().startswith('r '):
            q = q[2:].strip()
            want = False
        print()
        try:
            out = ask(q, scope=scope, top=top, backend=backend, want_answer=want)
            print(render(out, top))
        except Exception as e:
            print('出错：%s' % str(e)[:300])
        print()


def main():
    ap = argparse.ArgumentParser(description='本地知识库问答（Agentic 检索 + 带引用答案）')
    ap.add_argument('query', nargs='?', default=None, help='要问的问题；不填进交互模式')
    ap.add_argument('--scope', default='auto',
                    choices=list(SCOPE_CHOICES))
    ap.add_argument('--top', type=int, default=6, help='进入答案的片段数（默认 6）')
    ap.add_argument('--backend', default='cloud', choices=['cloud', 'local'],
                    help='答案生成用云端 glm-5.2（默认）还是本地 qwen2.5:7b')
    ap.add_argument('--no-agentic', action='store_true', help='关掉多轮升级，只跑一轮')
    ap.add_argument('--no-answer', action='store_true', help='只检索，不生成答案')
    ap.add_argument('--no-verify', action='store_true',
                    help='关掉答案自校验（默认开，仅云端生成时生效）')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    # 检索本身就依赖 Ollama（本地嵌入 + 重排），先自检；没起会自动拉起
    import kb_ask as ka
    if not ka.ensure_ollama():
        sys.exit(2)

    if not args.query:
        interactive(args.scope, args.top, args.backend)
        return

    out = ask(args.query, scope=args.scope, top=args.top, backend=args.backend,
              agentic=not args.no_agentic, want_answer=not args.no_answer,
              verify=not args.no_verify)
    if args.json:
        slim = {k: v for k, v in out.items() if k != 'results'}
        slim['results'] = [{kk: vv for kk, vv in r.items() if kk != 'text'} for r in out['results']]
        print(json.dumps(slim, ensure_ascii=False))
    else:
        print(render(out, args.top))


if __name__ == '__main__':
    main()
