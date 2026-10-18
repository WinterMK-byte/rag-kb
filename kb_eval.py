# -*- coding: utf-8 -*-
"""本地知识库 · 检索评测（Hit@1/3/5、MRR、越域率、耗时）

为什么要有它：没有评测集就只能靠"感觉好像准了"下结论。
体检报告里的 P7「无评测集」就是缺这块 —— 本脚本补上。

评测集：qbank/eval_set.jsonl，每条 {id, q, scope, expect[], expect_scope?, note}
    expect        期望命中的文件关键词（任一命中即算 hit）
    expect_scope  仅 auto 条目：期望被路由到的域

用法：
    py -3.12 -X utf8 kb_eval.py                 # 跑全部（条数见 --list-count，现为 34 条）
    py -3.12 -X utf8 kb_eval.py --cold          # 【口径标准】跑前清重排缓存（磁盘+内存），保证冷口径
    py -3.12 -X utf8 kb_eval.py --both          # 连跑冷/热两档并排打印（⚠️ 冷档可信度见 #194）
    py -3.12 -X utf8 kb_eval.py --no-rerank     # 消融：关重排
    py -3.12 -X utf8 kb_eval.py --no-rewrite    # 消融：关改写
    py -3.12 -X utf8 kb_eval.py --limit 8       # 只跑前 8 条（快速体检）
    py -3.12 -X utf8 kb_eval.py --json out.json # 存结果便于对比

🔴 为什么要有 --cold（2026-09-27 定案，务必先读）：
    kb_eval 的分数**依赖重排缓存命中**（重排缓存 key 含 RR_SALT，kb_search_v4 L494）。
    同码对照实测：清空 `cache\rerank_cache_v4.json` 重跑，Hit@1 会差约 3pt。
    → 所以：**报数字必须标口径**，跨版本比较必须同口径。
      ① `RR_SALT` 一升 → 缓存必然全失效 → 指标必掉 ~3pt，**属口径差不是退化**；
      ② **口诀：指标掉了先别改代码 —— 先清缓存重跑一遍。**

    ⚠️ **2026-09-27 #148 更正（比上面更重要）**：
       现役 `scope-v12` 下 `--both` 实测 **冷/热差 +0.0pt**（都是 70.6%）。
       旧记的「热 76.5% / 冷 73.5%」是 **v11 旧缓存存活期**读数，盐升后**已不可复现**。
       且 **70.6% 不是退化** —— 拆开看：09-26 新增的 8 条 mathdoc 讲义题（m07~m14）
       只有 3/8 rank=1，**其余 26 条老条目 = 80.8%**。
       → **判 Hit@1 要先按 ID 组/新老条目拆，再看总分。**

    🔴 **2026-10-01 #194 更正（推翻上面两条的可靠性，最优先读）**：
       实测发现 `clear_rr_cache()` 旧实现**只清磁盘、不清内存**（`_RR_CACHE.d` 是
       import 时加载的模块级单例，kb_search_v3.JsonCache L86-90），导致：
       ① `--cold` **同进程内不生效** → 冷档实际命中导入时的旧缓存，属「假冷」；
       ② `--both` 的"冷↔热差"**结构性恒为 +0.0pt**（warm 复用 cold 写回内存的结果，
          与 LLM 是否确定无关）→ 该行**零信息量**，不能拿它当"确定性"证据。
       已于 2026-10-01 修 `clear_rr_cache()` 同步清内存 + 清 `_dirty`。
       **凡 2026-10-01 之前用 `--cold` / `--both` 得到的数字，冷档口径存疑，需重取。**
       可靠真冷算二选一：① 修后用 `--cold`；② 跨进程跑（每轮新 python + 磁盘已清）。

    ⚠️ **2026-10-01 #194 补充：34 条样本 Hit@1 批次间波动 ±1 条（±2.9pt）。**
       同配置跨进程两次真冷算实测 79.4% vs 82.4%。**差 1~2 条的对比一律不足以定论。**

⚠️ --cold 只有在你**跑全量**时才有意义：它会清掉**整份**缓存，若只跑 --limit N，
   会把原来 34 条的缓存一起清掉（下次全量又要 9 分钟重排）。要做「快速体检」请**不要**加 --cold。

⚠️ 条数别写死在文档里（本行曾长期写「25 条」而实际 34 条，属勘误 #32）。
   要查真实条数：py -3.12 -X utf8 kb_eval.py --count
"""
import os
import sys
import json
import time
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

EVAL_PATH = os.path.join(KB, 'qbank', 'eval_set.jsonl')


def _norm(s):
    return (s or '').replace('\\', '/').replace(os.sep, '/').lower()


def load_eval(path=EVAL_PATH):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _resolve_scope(item):
    sc = item.get('scope') or 'auto'
    if sc == 'auto':
        import kb_ask as ka
        return ka.auto_scope(item['q']), True
    return sc, False


def rr_cache_path():
    """返回重排缓存文件路径（与 kb_search_v4 用的同一个）。"""
    return os.path.join(k3.CACHE_DIR, 'rerank_cache_v4.json')


def clear_rr_cache(log=print):
    """清空重排缓存 → 下次全部走真重排 = 「冷口径」。

    2026-09-27 新增。动机见文件头 docstring：kb_eval 的分数依赖重排缓存命中，
    清与不清差约 3pt。要做**可复现、可跨版本对比**的评测，必须先清缓存。
    ⚠️ 只清重排缓存，**不动改写缓存**（rewrite_v4 的 key 不含盐，跨盐复用是设计内的）。

    🔴 2026-10-01 修复（总线 #194）——「假冷」bug：
        `kb_search_v4._RR_CACHE` 是**模块级** JsonCache，其 `__init__` 在 **import 时**
        就把磁盘 json 读进内存字典 `self.d`（kb_search_v3.JsonCache L86-90）。
        旧实现只 `os.replace` 磁盘文件，**内存字典纹丝不动** →
        同进程内 `--cold` / `--both` 的冷档照样命中导入时的旧值，「冷口径」形同虚设。
        症状：冷档出现 0.1~0.4s 的"秒算"条目；`--both` 冷热差恒为 +0.0pt（非发现）。
        影响面：**跨进程不受影响**（新进程 import 时磁盘已清 → self.d={}），
        所以跨进程是唯一可靠的真冷算法；同进程必须靠下面这段内存清空。
    """
    p = rr_cache_path()
    n = 0
    if os.path.exists(p):
        try:
            with open(p, encoding='utf-8') as f:
                n = len(json.load(f))
        except Exception:
            n = -1
        # 沙箱纪律：不删，移到 .cold_bak（同目录内，可人工恢复）
        bak = p + '.cold_bak'
        try:
            if os.path.exists(bak):
                os.remove(bak)
            os.replace(p, bak)
        except Exception:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        log('  [cold] 已清空磁盘重排缓存（原 %s 条）→ %s' % (n, os.path.basename(bak)))
    else:
        log('  [cold] 磁盘重排缓存不存在（本就冷），跳过')
    # —— 内存单例同步清空（2026-10-01 修复，见 docstring）——
    try:
        _mem_n = len(k4._RR_CACHE.d)
        k4._RR_CACHE.d = {}
        k4._RR_CACHE._dirty = 0
        log('  [cold] 已清空内存重排缓存（原 %d 条）→ 本进程后续全部真算' % _mem_n)
    except Exception as _e:
        log('  [cold] ⚠️ 内存缓存清空失败（%s）→ 本次仍是「假冷」，请改用跨进程跑' % _e)
    return n


def run(items, top=5, vstore=k4.DEFAULT_VSTORE, use_rewrite=True, use_rerank=True,
        use_en=True, use_pool_en=True, scope_override=None, log=print):
    meta, vecs = k4.load_store_v(vstore)
    bm25 = k2.build_bm25(meta, vstore)
    rows = []
    for i, it in enumerate(items, 1):
        sc, is_auto = _resolve_scope(it)
        if scope_override:
            sc, is_auto = scope_override, False
        kw = dict(use_rewrite=use_rewrite, use_rerank=use_rerank, scope=sc,
                  use_en=use_en, use_pool_en=use_pool_en,
                  rewrite_model='qwen2.5:1.5b')
        t0 = time.time()
        try:
            out = k4.retrieve(it['q'], meta=meta, vecs=vecs, bm25=bm25, top=top,
                              vstore=vstore,
                              log=lambda *a, **k: None, **kw)
            res = out['results']
        except Exception as e:
            log('  [%d/%d] %s 出错（重试1次）：%s' % (i, len(items), it['id'], str(e)[:80]))
            try:
                out = k4.retrieve(it['q'], meta=meta, vecs=vecs, bm25=bm25, top=top,
                                  vstore=vstore,
                                  log=lambda *a, **k: None, **kw)
                res = out['results']
            except Exception as e2:
                log('  [%d/%d] %s 重试仍出错：%s' % (i, len(items), it['id'], str(e2)[:80]))
                rows.append({'id': it['id'], 'ok': False, 'err': str(e2)[:200]})
                continue
        dt = time.time() - t0
        paths = [_norm(r.get('file_path')) for r in res]
        exps = [_norm(e) for e in it.get('expect') or []]
        rank = None
        for n, p in enumerate(paths, 1):
            if any(e in p for e in exps):
                rank = n
                break
        # 越域检查（只看显式指定域的条目）
        viol = 0
        if not is_auto:
            for r in res:
                if not k4.in_scope(r, sc):
                    viol += 1
        scope_ok = None
        if is_auto and it.get('expect_scope'):
            scope_ok = (sc == it['expect_scope'])
        rows.append({'id': it['id'], 'q': it['q'], 'scope': sc, 'is_auto': is_auto,
                     'rank': rank, 'seconds': round(dt, 1), 'viol': viol,
                     'n': len(res), 'scope_ok': scope_ok,
                     'top1': os.path.basename(res[0]['file_path']) if res else None})
        flag = '✓' if rank else '✗'
        log('  [%2d/%2d] %s %-30s rank=%-4s %.1fs%s'
            % (i, len(items), flag, it['id'], rank if rank else '-', dt,
               '' if scope_ok is None else ('  域%s' % ('对' if scope_ok else '错'))))
    return rows


def summarize(rows, top=5):
    ok = [r for r in rows if r.get('ok', True) and 'rank' in r]
    n = len(ok) or 1
    h1 = sum(1 for r in ok if r['rank'] == 1) / n
    h3 = sum(1 for r in ok if r['rank'] and r['rank'] <= 3) / n
    h5 = sum(1 for r in ok if r['rank'] and r['rank'] <= 5) / n
    mrr = sum(1.0 / r['rank'] for r in ok if r['rank']) / n
    viol = sum(r['viol'] for r in ok)
    nres = sum(r['n'] for r in ok) or 1
    auto = [r for r in ok if r.get('scope_ok') is not None]
    routing = (sum(1 for r in auto if r['scope_ok']) / len(auto)) if auto else None
    n_err = len(rows) - len(ok)
    return {'n': len(ok), 'n_err': n_err, 'hit@1': round(h1, 3), 'hit@3': round(h3, 3),
            'hit@5': round(h5, 3), 'mrr': round(mrr, 3),
            'violation_rate': round(viol / nres, 3), 'violation_n': viol,
            'auto_routing_acc': round(routing, 3) if routing is not None else None,
            'avg_seconds': round(sum(r['seconds'] for r in ok) / n, 1)}


def main():
    ap = argparse.ArgumentParser(description='本地知识库检索评测')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--vstore', default=k4.DEFAULT_VSTORE)
    ap.add_argument('--no-rerank', action='store_true')
    ap.add_argument('--no-rewrite', action='store_true')
    ap.add_argument('--no-en', action='store_true', help='关英文分池通道')
    ap.add_argument('--scope', default=None, help='把所有条目的域强制改成这个')
    ap.add_argument('--json', default=None, help='把逐条结果写到这个 json')
    ap.add_argument('--tag', default='', help='给这次跑的法起个名字')
    ap.add_argument('--cold', action='store_true',
                    help='跑前清空重排缓存（冷口径，跨版本可比；见文件头说明）')
    ap.add_argument('--both', action='store_true',
                    help='连跑冷/热两档并排打印（先冷后热），用于口径对照')
    ap.add_argument('--count', action='store_true',
                    help='只打印评测集条数与域分布，不跑检索（用于核对文档是否陈旧）')
    ap.add_argument('--path', default=None,
                    help='评测集路径（默认 qbank/eval_set.jsonl；院校B专项用 qbank/eval_set_fudan.jsonl）')
    args = ap.parse_args()

    items = load_eval(args.path or EVAL_PATH)
    if args.count:
        import collections
        c = collections.Counter((it.get('scope') or 'all') for it in items)
        print('评测集 %s ｜ %d 条' % (args.path or EVAL_PATH, len(items)))
        for k, v in sorted(c.items()):
            print('  %-9s %d' % (k, v))
        return
    if args.limit:
        items = items[:args.limit]
    print('评测集 %d 条 ｜ 库 %s ｜ 改写=%s 重排=%s 英文池=%s%s'
          % (len(items), os.path.basename(args.vstore), not args.no_rewrite,
             not args.no_rerank, not args.no_en, (' ｜ 强制域=' + args.scope) if args.scope else ''))
    print()
    t0 = time.time()
    import kb_ask as ka
    if not ka.ensure_ollama():
        sys.exit(2)

    def _one(tag, cold):
        if cold:
            clear_rr_cache()
        rr = run(items, top=args.top, vstore=args.vstore,
                 use_rewrite=not args.no_rewrite, use_rerank=not args.no_rerank,
                 use_en=not args.no_en, use_pool_en=not args.no_en,
                 scope_override=args.scope)
        st = summarize(rr, args.top)
        if args.json and not args.both:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump({'tag': args.tag or ('cold' if cold else 'warm'),
                           'summary': st, 'rows': rr}, f, ensure_ascii=False, indent=1)
        return rr, st

    if args.both:
        rr_c, s_c = _one('cold', cold=True)
        rr_w, s_w = _one('warm', cold=False)

        def _line(name, st):
            return ('  %-6s 样本 %d ｜ Hit@1 %5.1f%% ｜ Hit@3 %5.1f%% ｜ Hit@5 %5.1f%% ｜ MRR %.3f ｜ 出错 %d'
                    % (name, st['n'], st['hit@1'] * 100, st['hit@3'] * 100,
                       st['hit@5'] * 100, st['mrr'], st['n_err']))

        print()
        print('=' * 58)
        print(_line('冷', s_c))
        print(_line('热', s_w))
        if s_c['n_err'] or s_w['n_err']:
            print('  [警告] 本轮有出错条目：冷 %d 条 / 热 %d 条（已重试1次）' % (s_c['n_err'], s_w['n_err']))
        d = (s_w['hit@1'] - s_c['hit@1']) * 100
        print('  冷↔热 Hit@1 差 %+.1fpt（正数 = 热缓存红利，非真实检索差异）' % d)
        print('=' * 58)
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump({'tag': args.tag or 'both', 'cold': s_c, 'warm': s_w,
                           'rows_cold': rr_c, 'rows_warm': rr_w}, f,
                          ensure_ascii=False, indent=1)
            print('逐条结果（冷/热）→', args.json)
        print('总耗时 %.0fs' % (time.time() - t0))
        return

    rows, s = _one(args.tag or ('cold' if args.cold else 'warm'), cold=args.cold)
    print()
    print('=' * 58)
    print('  口径 %s ｜ 样本 %d ｜ Hit@1 %.1f%% ｜ Hit@3 %.1f%% ｜ Hit@5 %.1f%% ｜ MRR %.3f ｜ 出错 %d'
          % ('冷' if args.cold else '热', s['n'], s['hit@1'] * 100,
             s['hit@3'] * 100, s['hit@5'] * 100, s['mrr'], s['n_err']))
    if s['n_err']:
        print('  [警告] 本轮有 %d 条出错（已重试1次），指标可能低估' % s['n_err'])
    print('  越域片段 %d（%.1f%%）｜ auto 路由正确率 %s ｜ 平均 %.1fs/条 ｜ 总 %.0fs'
          % (s['violation_n'], s['violation_rate'] * 100,
             ('%.0f%%' % (s['auto_routing_acc'] * 100)) if s['auto_routing_acc'] is not None else '—',
             s['avg_seconds'], time.time() - t0))
    print('=' * 58)
    miss = [r for r in rows if not r.get('rank')]
    if miss:
        print('\n未命中（rank=None）：')
        for r in miss:
            print('  %-5s %-34s top1=%s' % (r['id'], (r.get('q') or '')[:32], (r.get('top1') or '')[:40]))
    if args.json:
        print('\n逐条结果 →', args.json)


if __name__ == '__main__':
    main()
