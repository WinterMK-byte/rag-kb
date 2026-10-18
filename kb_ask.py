# -*- coding: utf-8 -*-
"""本地知识库 一键检索入口（kb_search_v4 检索链路 + 题目库的易用封装）

最简单的用法（不用记参数）：
    双击同目录下的「查资料.bat」，或双击桌面上的「本地知识库查询」快捷方式
    —— 然后在窗口里直接输入问题、回车即可。

命令行用法：
    py -3.12 -X utf8 kb_ask.py                     # 交互模式（可连续提问）
    py -3.12 -X utf8 kb_ask.py "要查的问题"          # 单次查询
    py -3.12 -X utf8 kb_ask.py "问题" --mode deep    # 指定档位
    py -3.12 -X utf8 kb_ask.py "问题" --json         # 机器可读输出（供其他程序调用）
    py -3.12 -X utf8 kb_ask.py "问题" --scope course  # 只搜课程资料（默认 auto：按问题自动判断域）
    py -3.12 -X utf8 kb_ask.py "极限的定义" --scope kaocha   # 只搜考插数学+英语资料
    py -3.12 -X utf8 kb_ask.py "极限的定义" --scope math     # 只搜考插数学资料
    py -3.12 -X utf8 kb_ask.py "四级长难句" --scope en       # 只搜考插英语资料

列题（数学题目库 · 知识点 → 题型，2026-09-19 新增）：
    py -3.12 -X utf8 kb_ask.py --list --kp 极限                 # 列「极限」全部题目
    py -3.12 -X utf8 kb_ask.py --list --kp 极限 --type 证明题    # 列「极限-证明题」
    py -3.12 -X utf8 kb_ask.py --list --kp 积分 --qtop 50       # 最多列 50 条
    py -3.12 -X utf8 kb_ask.py --list --kp 极限 --json          # 机器可读

交互模式内置命令：
    q 或直接回车      退出
    1..9             用系统默认程序打开第 N 条结果
    t 知识点 [题型]    列出该知识点（可选题型）下的题目
    m fast/deep/turbo  切换档位
    h                帮助

前置条件（2026-09-20 补记）：Ollama 必须常驻，脚本会自检并尝试自动拉起；
没有它向量化/重排都不可用。列题（--list）不依赖它。
"""
import os
import sys
import json
import time
import argparse
import subprocess

KB = os.path.dirname(os.path.abspath(__file__))
if KB not in sys.path:
    sys.path.insert(0, KB)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import kb_search_v2 as k2      # noqa: E402
import kb_search_v3 as k3      # noqa: E402
import kb_search_v4 as k4      # noqa: E402  （2026-09-17 起入口改用 v4 检索链路；向量库由 k4.DEFAULT_VSTORE 决定）

# 入口统一使用 v4 检索链路 + k4.DEFAULT_VSTORE 指定的向量库
# （2026-09-20 22:40 起 = vector_store_v8，嵌入模型换成 qwen3-embedding:0.6b / 1024 维；v7 可回退）
VSTORE = k4.DEFAULT_VSTORE

MODES = {
    'fast':  '本地检索：BGE 交叉编码器混合重排（推荐，约 1~2 秒/次）',
    'deep':  '云端模型：改写 + 重排（更准但更慢，约 1 分钟/次，需联网）',
    'turbo': '不调用大模型：只做向量 + 关键词召回（约 1~2 秒/次，精度较低）',
}
# 可用检索域（入口参数与交互菜单共用；含义见 kb_search_v4.SCOPE_RULES / SCOPE_GROUPS）
# 2026-09-21：改为从 kb_search_v4 的**单一真源**派生，不再本地各写一份（勘误 #31）
SCOPE_CHOICES = k4.SCOPE_CHOICES

HELP = """
可用操作：
  直接输入问题 + 回车    检索知识库（返回最相关的 5 条）
  1 / 2 / ... / 5       用系统默认程序打开第 N 条结果
  t 极限                列出「极限」知识点下的题目（前 10 条）
  t 极限 证明题          列出「极限」下「证明题」型的题目
  m fast|deep|turbo     切换检索档位
  s auto|core|kaocha|gpa|kmath|ken|gmath|gen|gother|math|en|mathdoc|tech|all  切换检索域
                        kaocha=考插（数学+英语）｜gpa=绩点（华理数学+英语+其他学科）｜core=考插+华理数学英语
                        kmath/ken=考插数学/英语｜gmath/gen=华理数学/英语｜gother=其他学科｜tech=技术库｜all=全库
  h                     显示本帮助
  q 或回车              退出

档位说明：
  fast   %s
  deep   %s
  turbo  %s
""" % (MODES['fast'], MODES['deep'], MODES['turbo'])


# 域分池（2026-09-20 22:40 重构）：默认域 = core（考插 + 华理数学英语）；其他学科（gother）
# 与技术库（tech）不在默认域内 —— 老板要求「除数学英语外的学科不干扰检索」。域表见 kb_search_v4。
# 只收「明确指向工具/技术文档」的词。故意不收 程序/函数/库/模型 —— 这些在课程语境里
# 很常见（C 程序设计、系统函数、电路模型），收了会把课程查询误送进全库。
TECH_HINTS = ('python', 'sympy', 'numpy', 'pandas', 'matlab', 'c++', 'java', 'pip', 'conda', 'git',
              'docker', 'api', 'latex', 'markdown', 'excel', 'word', 'ppt', 'ocr', 'pdf',
              '代码', '脚本', '报错', '安装', '配置', '框架', '技能', 'skill',
              '公式转', '命令行', '终端', '调试', 'debug', '检索', '向量')


# 考插班生是老板第一优先级（2026-09-20）：问题里出现这些词就直接送考插域。
KAOCHA_HINTS = ('考插', '插班生', '数学真题', '英语真题', '考试真题')
# 在「考插」基础上再细分科目（2026-09-20 加）：能判出科目就别用合并域，
# 否则「考插数学极限」会在 kaocha 里被同名的英语文档抢位。
MATH_HINTS = ('数学', '高数', '微积分', '极限', '导数', '微分', '积分', '级数', '中值定理',
              '线性代数', '行列式', '概率', '矩阵', '向量代数', '曲线积分', '重积分')
EN_HINTS = ('英语', '词汇', '单词', '背词', '翻译', '作文', '写作', '完形', '完型', '阅读',
            '语法', '长难句', '四级', '六级', 'cloze', 'translation')


# 其他学科线索（2026-09-20 22:40 加 / 23:05 大幅扩充）：默认域 core 只含「考插 + 华理数学英语」，
# 其他学科片段被排除在外。问题里出现这些学科词时，才主动路由到 gother。
#
# ⚠️ 覆盖不足的后果（2026-09-20 实测踩到）：问「PN结的形成原理」如果判不进 gother，
# 就会在 core 里搜数学讲义（返回泰勒公式），比原来更差；「电子的能带结构」则直接查不到。
# 所以这张表要按**学科术语**铺，不是按课程名铺。刻意仍不收：
# 「函数 / 模型 / 库 / 程序 / 计算 / 公式 / 实验」—— 这些在数学语境里太常见，收了会误伤。
OTHER_HINTS = (
    # 物理 / 大物
    '物理', '大物', '力学', '热学', '热力学', '光学', '波动', '振动', '声学',
    '电磁', '量子', '相对论', '统计物理', '原子', '分子', '刚体', '动量', '冲量',
    # 电子 / 信息（'电子' 单字词必要：「电子的能带结构」「电力电子器件」都靠它兜住）
    '电子', '半导体', 'pn结', 'PN结', '二极管', '三极管', '晶体管', '能带', '器件',
    '放大器', '运算放大器', '集成电路', '电路', '模电', '数电', '模拟电子', '数字电子',
    '信号与系统', '通信', '天线', '单片机', '嵌入式', '微机', 'fpga', 'FPGA', '数字信号',
    # 计算机
    '数据结构', '计算机', '操作系统', '计算机网络', '编译原理', '离散数学',
    'c语言', 'C语言', '程序设计', '算法', '数据库', '软件工程',
    '机器学习', '人工智能', '深度学习', '神经网络',
    # 化学 / 材料 / 机械
    '化学', '有机', '无机', '材料', '工程制图', '制图', '金工', '流体', '机械', '化工',
    # 人文社科 / 其他
    '马克思主义', '马原', '思政', '毛概', '毛中特', '近代史', '思修', '军事理论',
    '哲学', '经济学', '管理学', '法律', '保研', '体育',
)


def auto_scope(q):
    """按问题自动选域。

    2026-09-20 22:40 重构：默认域由 course（全部课业）改为 **core（考插 + 华理数学英语）**，
    其他学科（物理/信息/马原/机动）不再参与默认检索 —— 「其他学科不干扰检索」的落点。
    要搜其他学科：问题里带学科词（走 OTHER_HINTS），或显式用 --scope gother。
    域表见 kb_search_v4.SCOPE_RULES / SCOPE_GROUPS。
    """
    low = (q or '').lower()
    m = any(h in low for h in MATH_HINTS)
    e = any(h in low for h in EN_HINTS)
    k = any(h in low for h in KAOCHA_HINTS)
    o = any(h in low for h in OTHER_HINTS)
    t = any(h in low for h in TECH_HINTS)
    if k:
        # 考插是老板第一优先级；能判出科目就别用合并域，否则同名的英语文档会抢位
        if m and not e:
            return 'kmath'
        if e and not m:
            return 'ken'
        return 'kaocha'
    if o:
        return 'gother'
    if t and not m and not e:
        return 'tech'
    if m and not e:
        return 'math'    # 数学全量：考插数学 + 华理数学
    if e and not m:
        return 'en'      # 英语全量：考插英语 + 华理英语
    return 'core'


def mode_kwargs(mode, scope='all', rr_parallel=False):
    """把「档位 + 域」翻译成 kb_search_v4.retrieve 的参数（默认库 = VSTORE）。

    rr_parallel：重排是否并行（2026-09-22 新增，默认 False）。
      A/B 实测（34 条评测集，本机）：串行 23.8s/条 → 并行 8.5s/条（2.8×），
      且 Hit@1 64.7%→67.6%、Hit@3 88.2%→91.2%、MRR 0.781→0.798（**更好**），
      但 Hit@5 100%→97.1%，且**不可复现**（3 轮中 7 条排序漂移 —— Ollama 批合并所致）。
      → 故默认仍为串行（保评测基线可比）；要速度时显式开 `--rr-parallel`。
    """
    if mode == 'deep':
        return dict(use_rewrite=True, use_rerank=True, model=k3.DEEP_MODEL, backend='cloud',
                    cands_n=30, workers=1, preview_chars=220, two_stage=False, w_fuse=1.0,
                    scope=scope, rr_parallel=rr_parallel)
    if mode == 'turbo':
        return dict(use_rewrite=False, use_rerank=False, scope=scope)
    # fast：v4 全通道 + BGE 交叉编码器重排（2026-10-15 起默认引擎）。
    # 改写用 qwen2.5:1.5b（改写任务简单，1.5b 足够；与 7b 相比常驻显存从 4.8GB 降到 986MB，
    # 腾出的空间留给 BGE-Reranker 交叉编码器重排，单次检索 9.2s → 约 2~4s）。
    # 重排引擎：kb_search_v4.retrieve 默认 rerank_engine='bge'（如需回退 LLM 两阶段重排，
    # 显式传 rerank_engine='llm' 即可，旧 kb_search_v4 行为保留）。
    # scope=course 时额外关掉英文分池与英文保底通道 —— 实测这两路会把英语考辅文档
    # （如「英语四六级考试段落翻译题.doc」）顶到中文专业课查询的 top1。
    kw = dict(use_rewrite=True, use_rerank=True, use_en=True, use_pool_en=True, scope=scope,
              rr_parallel=rr_parallel, rewrite_model='qwen2.5:1.5b')
    if scope in k4.EN_FREE_SCOPES:
        # 只关英文分池，保底通道 use_guard 要留着 —— 消融实测：关掉它「能带结构」那类
        # 查询会退回不到半导体器件/量子物理课件（配置 E 才是最优组合）。
        # 中文语料域（kmath/gmath/gother/math/mathdoc）同理：中文问题不该被英语考辅文档拉走。
        # 含英语语料的域（core/kaocha/gpa/ken/gen/en）不关，否则英语题搜不到。
        kw.update(use_en=False, use_pool_en=False)
    return kw


def ensure_ollama(wait=30):
    """RAG 的硬依赖：Ollama 必须常驻（本地嵌入 + 重排都打 11434）。

    没起就自动拉起（分离进程，不弹窗、不随本窗口退出而结束）；拉不起来就明确报错，
    而不是抛一堆 requests 栈让人看不懂。
    """
    import requests

    url = 'http://127.0.0.1:11434/api/tags'
    no_proxy = {'http': None, 'https': None}

    def _up():
        try:
            return requests.get(url, timeout=3, proxies=no_proxy).status_code == 200
        except Exception:
            return False

    if _up():
        return True

    exe = os.path.expandvars(r'%LOCALAPPDATA%\Programs\Ollama\ollama app.exe')
    if not os.path.exists(exe):
        exe = os.path.expandvars(r'%LOCALAPPDATA%\Programs\Ollama\ollama.exe')
    if os.path.exists(exe):
        print('[启动] Ollama 没在跑，正在后台拉起 ...', flush=True)
        try:
            env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
            subprocess.Popen([exe, 'serve'] if exe.endswith('ollama.exe') else [exe],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                             creationflags=0x00000008 | 0x00000200)
        except Exception:
            try:
                subprocess.Popen(['start', '', exe], shell=True)
            except Exception:
                pass
        t0 = time.time()
        while time.time() - t0 < wait:
            if _up():
                print('[启动] Ollama 就绪（%.0f 秒）。' % (time.time() - t0), flush=True)
                return True
            time.sleep(1)

    print('')
    print('!' * 62)
    print('检索跑不起来：Ollama 不在运行。')
    print('  原因：向量化（qwen3-embedding:0.6b）和重排（qwen2.5:7b）都在本地 Ollama 上，')
    print('        这是本地知识库唯一的硬依赖，没它就完全没有结果。')
    print('  解决：双击  %s' % os.path.join(KB, '启动Ollama.bat'))
    print('        或命令行  ollama serve')
    print('  注意：列题模式（--list）不需要 Ollama，随时可用。')
    print('!' * 62)
    return False


def load_store(verbose=True):
    if verbose:
        print('正在加载索引与模型（首次约 10 秒）...', flush=True)
    t = time.time()
    meta, vecs = k4.load_store_v(VSTORE)
    bm25 = k2.build_bm25(meta, VSTORE)
    if verbose:
        print('索引就绪：%d 个片段（库=%s），用时 %.1f 秒'
              % (len(meta), os.path.basename(VSTORE), time.time() - t), flush=True)
    return meta, vecs, bm25


def search(query, mode, top, meta, vecs, bm25, kp=None, qtype=None, scope=None,
           rr_parallel=True):
    return k4.retrieve(query, meta, vecs, top=top, bm25=bm25, vstore=VSTORE,
                       log=lambda *a, **k: None, kp=kp, qtype=qtype,
                       **mode_kwargs(mode, scope or 'all', rr_parallel=rr_parallel))


# ------------------------------ 列题（题目库 · 知识点 → 题型，2026-09-19 新增）
def show_questions(kp=None, qtype=None, top=20, as_json=False, kw=None):
    """列出题目库中命中「知识点 / 题型」的题目（不走向量检索，秒级返回）。"""
    items = k4.qbank_search(kw or '', kp=kp, qtype=qtype, top=top)
    if as_json:
        print(json.dumps({'kp': kp, 'type': qtype, 'returned': len(items),
                          'qbank_total': k4.qbank_meta().get('count'),
                          'items': [{k: q.get(k) for k in
                                     ('qid', 'file_name', 'file_path', 'start_page', 'end_page',
                                      'kp', 'qtype', 'kp_src', 'q_score', 'text')} for q in items]},
                         ensure_ascii=False))
        return items
    qm = k4.qbank_meta()
    print('\n【题目清单】知识点=%s ｜ 题型=%s ｜ 题库共 %s 题 → 命中 %d 条'
          % (kp or '不限', qtype or '不限', qm.get('count'), len(items)))
    if not items:
        print('  （没有命中的题目）可用知识点：%s' % '、'.join(qm.get('kp_set') or []))
        return items
    for n, q in enumerate(items, 1):
        loc = ('第 %s 页' % q.get('start_page')) if str(q.get('start_page')) == str(q.get('end_page')) \
            else ('第 %s-%s 页' % (q.get('start_page'), q.get('end_page')))
        print('\n  [%d] %s ｜ %s ｜ %s%s' % (n, q.get('file_name'), loc, q.get('qtype'),
                                            (' ｜ 相关度 %s' % q['q_score']) if q.get('q_score') else ''))
        snip = ' '.join((q.get('text') or '').split())[:130]
        if snip:
            print('      %s' % snip)
        print('      %s' % (q.get('file_path') or ''))
    print('\n  提示：加 --json 可输出机器可读结果；--qtop N 调整条数。')
    return items


def show(info, top):
    res = info['results']
    st = info['stages']
    if not res:
        print('  （没有找到相关内容，可换一种说法再试）')
        return
    print('  用时 %.1f 秒（改写 %.1f / 召回 %.1f / 重排 %.1f）'
          % (st['total_seconds'], st['rewrite']['seconds'], st['recall']['seconds'], st['rerank']['seconds']))
    for i, r in enumerate(res, 1):
        snip = ' '.join((r.get('text') or '').split())[:150]
        print('\n  [%d] 相关度 %-5s %s' % (i, r['rank_score'], os.path.basename(r['file_path'])))
        print('      位置：%s  (第 %s 段)' % (r['rel_path'], r['chunk_index']))
        if snip:
            print('      片段：%s' % snip)


def to_json(info):
    return {'query': info['query'], 'vstore': os.path.basename(VSTORE), 'stages': info['stages'],
            'results': [{'rank': i + 1, 'rel_score': r['rank_score'], 'file_name': os.path.basename(r['file_path']),
                         'file_path': r['file_path'], 'rel_path': r['rel_path'], 'chunk_index': r['chunk_index'],
                         'snippet': ' '.join((r.get('text') or '').split())[:200]}
                        for i, r in enumerate(info['results'])]}


def interactive(mode, top, meta, vecs, bm25, scope='auto', rr_parallel=True):
    last = []
    print('\n本地知识库检索已就绪 —— 档位 %s（%s）｜检索域 %s' % (mode, MODES[mode], scope))
    if scope in ('math', 'en', 'kaocha'):
        try:
            _qm = k4.qbank_meta()
            if _qm.get('count'):
                print('考插题目库：%s 题 / %s 份资料' % (_qm.get('count'), _qm.get('files')))
                print('知识点：%s' % '、'.join(_qm.get('kp_set') or []))
                print('数学请优先列题：t 极限   或   t 极限 证明题')
        except Exception:
            pass
    print('直接输入问题回车即可；输入 h 看帮助，q 退出。\n')
    while True:
        try:
            q = input('问题> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n已退出。')
            return
        if not q or q.lower() in ('q', 'quit', 'exit', '退出'):
            print('已退出。')
            return
        if q.lower() == 'h':
            print(HELP)
            continue
        if q.lower().startswith('m '):
            m = q[2:].strip().lower()
            if m in MODES:
                mode = m
                print('已切换到 %s 档：%s' % (mode, MODES[mode]))
            else:
                print('档位只能选 fast / deep / turbo')
            continue
        if q.lower().startswith('s ') or q.lower().startswith('域'):
            _v = (q.split() + [''])[1].strip().lower() if q.startswith('s ') else q[1:].strip().lower()
            if _v in SCOPE_CHOICES:
                scope = _v
                print('已切换检索域 → %s' % scope)
            else:
                print('检索域只能选 %s' % ' / '.join(SCOPE_CHOICES))
            continue
        if q.lower().startswith('t ') or q.lower().startswith('列'):
            parts = q.split()[1:]
            show_questions(parts[0] if parts else None, parts[1] if len(parts) > 1 else None, 10)
            print()
            continue
        if q.isdigit() and last:
            i = int(q)
            if 1 <= i <= len(last):
                path = last[i - 1]['file_path']
                print('正在打开：%s' % path)
                try:
                    os.startfile(path)          # noqa: 仅打开，不做任何修改
                except Exception as e:
                    print('打开失败：%s' % e)
            else:
                print('编号超出范围（当前 1~%d）' % len(last))
            continue
        _sc = auto_scope(q) if scope == 'auto' else scope
        print('检索中（%s 档 · %s 域）...' % (mode, _sc), flush=True)
        try:
            info = search(q, mode, top, meta, vecs, bm25, scope=_sc,
                          rr_parallel=rr_parallel)
        except Exception as e:
            print('检索出错：%s' % e)
            continue
        last = info['results']
        show(info, top)
        print('\n输入编号可打开对应文件；或继续输入下一个问题，q 退出。\n')


def main():
    ap = argparse.ArgumentParser(description='本地知识库一键检索（易用入口）')
    ap.add_argument('query', nargs='?', default=None, help='要检索的问题；不填则进入交互模式')
    ap.add_argument('--mode', default='fast', choices=['fast', 'deep', 'turbo'], help='检索档位')
    ap.add_argument('--scope', default='auto',
                    choices=list(SCOPE_CHOICES),
                    help='检索域：auto=按问题自动（默认 → core）｜'
                         'core=考插+华理数学英语（不含其他学科）｜'
                         'kaocha=考插（数学+英语）｜gpa=绩点（华理数学+英语+其他学科）｜'
                         'kmath/ken=考插数学/英语｜gmath/gen=华理数学/英语｜'
                         'gother=其他学科（物理/信息/马原/机动）｜'
                         'math/en=该科目全量（考插+华理）｜'
                         'mathdoc=数学讲解（高数/线代/概率讲义·课件）｜'
                         'tech=技能/技术库｜all=全库')
    ap.add_argument('--top', type=int, default=5, help='返回条数（默认 5）')
    ap.add_argument('--json', action='store_true', help='以 JSON 输出结果（供脚本调用）')
    ap.add_argument('--kp', default=None, help='知识点过滤（极限/导数与微分/积分/微分方程/无穷级数/重积分/中值定理与泰勒/定积分应用/多元函数微分/曲线曲面积分/空间解析几何/线性代数/概率论与数理统计/其他）')
    ap.add_argument('--type', dest='qtype', default=None, help='题型过滤（选择题/填空题/计算题/证明题/应用题）')
    ap.add_argument('--list', dest='list_mode', action='store_true',
                    help='列题模式：按 --kp / --type 列出题目（不走向量检索）')
    ap.add_argument('--qtop', type=int, default=20, help='列题模式返回条数（默认 20）')
    ap.add_argument('--rr-parallel', dest='rr_parallel', action='store_true',
                    help='重排并行（已默认开启，此参数保留兼容，无需再传）')
    ap.add_argument('--serial-rerank', dest='serial_rerank', action='store_true',
                    help='重排改回串行（旧默认：~23.8s/条，但排序完全可复现、评测基线可比；追求速度不要加）')
    args = ap.parse_args()

    # 列题模式：完全不加载索引与模型（秒级返回）
    if args.list_mode or (args.query is None and (args.kp or args.qtype)):
        show_questions(args.kp, args.qtype, args.qtop, args.json, kw=args.query)
        return

    if not ensure_ollama():
        sys.exit(2)

    # [78施工单] 显存协调层前置检查：acquire ollama 非 0 → 打印提示并停止（绝不降级跑 CPU）
    import subprocess as _sp
    _vp = os.path.join(KB, '..', '30_协作', '_state', 'vram_lease.py')
    _p = _sp.run([sys.executable, os.path.abspath(_vp), 'acquire', '--group', 'ollama'],
                 capture_output=True, text=True, timeout=60)
    if _p.returncode != 0:
        # [老板授权 2026-10-16] 自主释放显存：非 0 先尝试自主释放（冷驻留/僵尸/残留），
        # 成功并重试 acquire 通过 → 继续；仍不足才中止，绝不降级跑 CPU。
        _ar = os.path.join(KB, '_scratch', 'vram_autorelease.py')
        if os.path.exists(_ar):
            try:
                _ap = _sp.run([sys.executable, _ar], capture_output=True, text=True, timeout=180)
                if _ap.returncode == 0:
                    print('[显存自主释放] 释放成功，重试 acquire 通过。', flush=True)
                else:
                    if _ap.stdout:
                        print(_ap.stdout[-400:], flush=True)
                    print('[显存自主释放] 释放后仍不足/冲突（rc=%d）：中止，绝不降级跑 CPU。' % _ap.returncode, flush=True)
                    sys.exit(6)
            except Exception as _e:
                print('[显存自主释放] 自主释放调用异常：%s' % str(_e)[:120], flush=True)
                sys.exit(6)
        print('', flush=True)
        print('=' * 62, flush=True)
        print('  [显存协调层] 显存不足/冲突：acquire ollama 返回 %d —— 中止，绝不降级跑 CPU。' % _p.returncode, flush=True)
        print('  请先释放显存（vram_status.py 可查占用方），再重试。', flush=True)
        print('=' * 62, flush=True)
        sys.exit(6)

    meta, vecs, bm25 = load_store(verbose=not args.json)
    if args.mode == 'fast':
        k3.llm_local('预热', num_predict=8)      # 预热本地模型，避免第一条查询多等十几秒
    _scope = args.scope
    if args.query:
        if _scope == 'auto':
            _scope = auto_scope(args.query)
        info = search(args.query, args.mode, args.top, meta, vecs, bm25,
                      kp=args.kp, qtype=args.qtype, scope=_scope,
                      rr_parallel=args.rr_parallel or (not args.serial_rerank))
        if args.json:
            print(json.dumps(to_json(info), ensure_ascii=False))
        else:
            print('\n【问题】%s%s' % (args.query,
                                    ('（限知识点 %s%s）' % (args.kp, ('/' + args.qtype) if args.qtype else ''))
                                    if (args.kp or args.qtype) else ''))
            print('  [检索域] %s' % _scope)
            _qb = (info.get('stages') or {}).get('qbank')
            if _qb:
                print('  [题目库收窄] 命中文件 %d 份（候选 %d → %d）'
                      % (_qb.get('allow_files') or 0, _qb.get('fused_before'), _qb.get('fused_after')))
            show(info, args.top)
    else:
        interactive(args.mode, args.top, meta, vecs, bm25, _scope,
                    rr_parallel=args.rr_parallel or (not args.serial_rerank))


if __name__ == '__main__':
    main()
