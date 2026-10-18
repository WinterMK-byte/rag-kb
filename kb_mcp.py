# -*- coding: utf-8 -*-
"""本地知识库 · MCP server（stdio）

作用：把本地知识库变成任何支持 MCP 的 agent 都能直接调用的工具集。
    Claude Desktop / WorkBuddy / Cursor / 各类自建 agent —— 只要它能配 MCP server。

暴露的工具：
    kb_search(query, scope, top)          检索片段（快，约 1~8 秒）
    kb_ask(query, scope, top, backend)    带 [编号] 引用的答案（慢，约 10~40 秒）
    kb_list_questions(kp, qtype, top)     考插数学列题（秒级，不走向量检索）
    kb_stats()                            库存与四域分布
    kb_scopes()                           可用检索域及含义

⚠️ 协议纪律：stdio 模式下 **stdout 只能有 JSON-RPC**。
   所有日志、print、Ollama 启动提示一律重定向到 stderr，否则客户端会解析失败。
"""
import os
import sys
import json

KB = os.path.dirname(os.path.abspath(__file__))
if KB not in sys.path:
    sys.path.insert(0, KB)
os.chdir(KB)

# MCP 客户端（如 WorkBuddy）不会给子进程注入用户的代理变量；
# 但若继承到了 HTTP_PROXY，打 localhost:11434 会被代理拦成 502 —— 显式清掉。
for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(_k, None)

from mcp.server.fastmcp import FastMCP           # noqa: E402

# ⚠️ 这里**绝不能**用 contextlib.redirect_stdout：
# MCP 的 stdio 传输自己就在 sys.stdout 上收发 JSON-RPC，
# 一旦把 sys.stdout 换成 stderr，响应就写进了 stderr → 客户端永远等不到 → 死锁。
# 实测症状：list_tools 正常，第一个 call_tool 直接卡死。
# 正确做法：把各 kb 模块全局命名空间里的 print 换成写 stderr 的版本（外科手术式，不动 sys.stdout）。
import builtins                                  # noqa: E402
import logging                                   # noqa: E402

_LOG = logging.getLogger('kb-mcp')


def _print_stderr(*a, **kw):
    kw['file'] = sys.stderr
    try:
        return builtins.print(*a, **kw)
    except Exception:
        return None


def _log(*a):
    print('[kb-mcp]', *a, file=sys.stderr, flush=True)


def _install_quiet():
    """① jieba 等日志压到 stderr 且降到 WARNING；② 各 kb 模块的 print → stderr。"""
    for name in ('jieba', 'jieba.posseg'):
        lg = logging.getLogger(name)
        lg.handlers = [logging.StreamHandler(sys.stderr)]
        lg.setLevel(logging.WARNING)
        lg.propagate = False
    mods = []
    for n in ('kb_search_v2', 'kb_search_v3', 'kb_search_v4', 'kb_ask', 'kb_answer'):
        try:
            __import__(n)
            mods.append(sys.modules[n])
        except Exception as e:
            _log('模块 %s 导入失败：%s' % (n, e))
    for m in mods:
        m.__dict__['print'] = _print_stderr
    _log('已隔离输出：%d 个模块的 print → stderr，jieba 日志 → WARNING' % len(mods))


mcp = FastMCP('local-kb')
_install_quiet()


# ------------------------------------------------------------------ 工具
@mcp.tool()
def kb_search(query: str, scope: str = 'auto', top: int = 5) -> str:
    """在老板的本地资料库里检索，返回最相关的片段（含文件路径、页码、正文）。

    Args:
        query: 要查的问题或关键词。带上专业术语效果最好。
        scope: 检索域。auto=自动判断(默认)｜math=考插数学题目库｜
               mathdoc=考插数学+数学讲解（高数/线代/概率讲义·课件，问概念/定理/公式用）｜
               en=考插英语｜kaocha=考插数学+英语｜course=华理课业｜tech=工具与技术库｜all=全库。
               查考插资料建议显式用 kaocha/math/en；查课业用 course。
        top: 返回条数，默认 5。
    """
    import kb_ask as ka
    ka.ensure_ollama()
    import kb_search_v2 as k2
    import kb_search_v4 as k4
    meta, vecs = k4.load_store_v(k4.DEFAULT_VSTORE)
    bm25 = k2.build_bm25(meta, k4.DEFAULT_VSTORE)
    sc = ka.auto_scope(query) if scope in (None, '', 'auto') else scope
    kw = dict(use_rewrite=True, use_rerank=True, scope=sc)
    if sc in k4.EN_FREE_SCOPES:
        kw.update(use_en=False, use_pool_en=False)
    out = k4.retrieve(query, meta=meta, vecs=vecs, bm25=bm25, top=top,
                      log=lambda *a, **k: None, **kw)
    rows = out['results']
    lines = ['检索域 %s ｜ 命中 %d 条' % (sc, len(rows))]
    for i, r in enumerate(rows, 1):
        lines.append('[%d] %s（第 %s 段，相关度 %.2f）'
                     % (i, r['rel_path'], r.get('chunk_index'), r.get('rank_score') or 0))
        txt = (r.get('text') or '').strip().replace('\n', ' ')
        lines.append('    ' + (txt[:400] if txt else '（该条目无可用正文）'))
    return '\n'.join(lines)


@mcp.tool()
def kb_ask(query: str, scope: str = 'auto', top: int = 6, backend: str = 'cloud') -> str:
    """基于本地资料库生成一段带 [编号] 出处的答案；资料里没有会明确说没有（不编造）。

    Args:
        query: 要问的问题。
        scope: 检索域，取值同 kb_search。
        top: 进入答案的片段数，默认 6。
        backend: cloud=云端 glm-5.2（默认，更准）｜local=本地 qwen2.5:7b（离线）。
    """
    import kb_answer
    import kb_ask as ka
    ka.ensure_ollama()
    out = kb_answer.ask(query, scope=scope, top=top, backend=backend, log=lambda *a, **k: None)
    if not out.get('answer'):
        head = '（未能生成答案：%s）' % (out.get('note') or '证据不足')
    else:
        head = out['answer']
    used = []
    for i, r in enumerate(out['results'][:top], 1):
        used.append('[%d] %s（第 %s 段）' % (i, r['rel_path'], r.get('chunk_index')))
    tail = ''
    if out.get('rounds') and len(out['rounds']) > 1:
        tail = '\n（Agentic 检索跑了 %d 轮：%s）' % (
            len(out['rounds']), ' → '.join('%s/%s' % (x['profile'].split('(')[0], x['scope'])
                                           for x in out['rounds']))
    return '%s\n\n出处：\n%s%s' % (head, '\n'.join(used), tail)


@mcp.tool()
def kb_list_questions(kp: str = '', qtype: str = '', top: int = 20) -> str:
    """列出考插数学题目库（1832 题 / 49 份资料），按知识点、题型筛选。秒级返回。

    Args:
        kp: 知识点，**子串匹配**。常用：极限 / 导数与微分 / 积分 / 定积分应用 /
            中值定理与泰勒 / 微分方程 / 无穷级数 / 重积分 / 曲线曲面积分 /
            多元函数微分 / 空间解析几何 / 线性代数 / 概率论与数理统计 / 其他。
            留空 = 不限。
        qtype: 题型：选择题 / 填空题 / 计算题 / 证明题 / 应用题。留空 = 不限。
        top: 最多返回条数，默认 20。
    """
    import kb_search_v2 as _k2  # noqa: F401
    import kb_search_v4 as k4
    items = k4.qbank_filter(kp=kp or None, qtype=qtype or None)
    hits = k4.qbank_search('', kp=kp or None, qtype=qtype or None, top=top, items=items)
    meta = k4.qbank_meta() or {}
    lines = ['考插数学题目库：共 %s 题 / %s 份资料 ｜ 筛选 知识点=%s 题型=%s → 命中 %d 条'
             % (meta.get('count'), meta.get('files'), kp or '不限', qtype or '不限', len(hits))]
    for n, q in enumerate(hits, 1):
        lines.append('[%d] %s ｜ 第 %s 页 ｜ %s'
                     % (n, os.path.basename(q.get('file_path') or ''), q.get('start_page'),
                        q.get('qtype') or '未标题型'))
        txt = (q.get('text') or '').strip().replace('\n', ' ')
        lines.append('    ' + txt[:200])
    return '\n'.join(lines)


@mcp.tool()
def kb_stats() -> str:
    """知识库库存：向量库版本、片段总数、四个检索域各装了多少。"""
    import kb_search_v4 as k4
    meta, vecs = k4.load_store_v(k4.DEFAULT_VSTORE)
    from collections import Counter
    c = Counter(k4.scope_of(r) for r in meta)
    qm = k4.qbank_meta() or {}
    files = len({r['file_path'] for r in meta})
    label = {'math': '考插数学（题目）', 'mathdoc': '数学讲解（高数/线代/概率讲义·课件）',
             'en': '考插英语', 'course': '华理课业',
             'tech': '工具与技术库', 'other': '未归属（散件）'}
    lines = ['库：%s ｜ 片段 %d ｜ 文件 %d ｜ 向量维度 %d'
             % (os.path.basename(k4.DEFAULT_VSTORE), len(meta), files, vecs.shape[1])]
    for k, n in c.most_common():
        lines.append('  %-4s %-10s %5d 片段（%.1f%%）' % (k, label.get(k, k), n, 100.0 * n / len(meta)))
    lines.append('考插数学题目库：%s 题 / %s 份资料（知识点 %d 个）'
                 % (qm.get('count'), qm.get('files'), len(qm.get('kp_set') or [])))
    return '\n'.join(lines)


@mcp.tool()
def kb_scopes() -> str:
    """列出可用检索域及各自含义，便于决定 kb_search / kb_ask 该传哪个 scope。"""
    import kb_search_v4 as k4
    rules = {k: list(v) for k, v in k4.SCOPE_RULES.items()}
    groups = {k: list(v) for k, v in k4.SCOPE_GROUPS.items()}
    groups['mathdoc'] = list(k4.SCOPE_MATHDOC)
    return json.dumps({
        'auto': '按问题自动（默认 → core）：含「考插/插班生」→ kaocha（再分 kmath/ken）；'
                '含其他学科词（物理/马原/数据结构…）→ gother；工具技术词 → tech；'
                '含数学词 → math（考插+华理）；含英语词 → en；其余 → core',
        'core': '**默认域**：考插 + 华理数学英语，不含其他学科与技术库（其他学科不干扰检索）',
        'kaocha': '考插全部（数学 + 英语）',
        'gpa': '绩点全部（华理数学 + 英语 + 其他学科）',
        'kmath': '考插数学（题目库 + 数学真题）—— 标记 %s' % rules.get('kmath'),
        'ken': '考插英语（题目库 + 英语真题）—— 标记 %s' % rules.get('ken'),
        'gmath': '华理数学（数学学院：讲义/课件/作业/真题）—— 标记 %s' % rules.get('gmath'),
        'gen': '华理英语（外国语学院）—— 标记 %s' % rules.get('gen'),
        'gother': '其他学科（信息/物理/马原/机动/学工部 + 保研）—— 标记 %s' % rules.get('gother'),
        'math': '数学全量（考插数学 + 华理数学）',
        'en': '英语全量（考插英语 + 华理英语）',
        'mathdoc': '考插数学 + 数学讲解（华理数学学院公共课「讲义/课件」白名单，'
                   '不含作业实验与 .swf）—— 问概念/定理/公式/怎么用时用这个',
        'tech': '工具与技术库 —— 标记 %s' % rules.get('tech'),
        'all': '全库，不过滤',
        '组别名': groups,
    }, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    _log('启动中：库=%s' % KB)
    mcp.run()
