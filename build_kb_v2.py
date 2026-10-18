# -*- coding: utf-8 -*-
"""本地知识库重建/增量脚本 v2（Resource 版 + 质量修复）

相对 v1 的改动：
 1) 失效源文件条目清理：磁盘上已不存在的 file_path 记录自动移除（--keep-missing 可关闭）
 2) mtime+size 增量比对：每条记录写入 file_mtime/file_size；无此字段的旧条目视为待迁移重嵌
 3) MAX_CHUNKS_PER_FILE 15 → 60（--max-chunks 可调）；旧的 15 块记录（疑似截断）自动重嵌
    * 2026-09-15 修复：重嵌判定改为比对「记录内 max_chunks 标记」与「本次请求上限」，
      重嵌后把上限写回记录，因此同一批旧记录只重嵌一次，不再每轮重复处理；旧记录无该字段按 15 兜底。
 4) 文本优先取提取缓存 text_cache/<sha1>.txt（含 OCR / COM 结果），命中即不重复解析
 5) 断点续跑：每 --save-every 个文件原子落盘（.tmp + os.replace），中断重跑按 mtime/size 自动跳过已完成
 6) --rebuild 不再删除旧库，改为改名备份 .old_<时间戳>

用法:
    py -3.12 build_kb.py --root "C:\\Users\\<you>\\Desktop\\Resource" --out vector_store
              [--limit N] [--batch 32] [--text-cache <dir>] [--max-chunks 60]
              [--save-every 20] [--keep-missing] [--rebuild] [--report <json>]
输出: <out>/metadata.json + vectors.npy + config.json（与 kb_search.py 兼容）
"""
import os, sys, json, time, argparse, re, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 2026-09-28 P1-5（#139）：正文清洗钩子（Adobe Symbol PUA→真符号 + 控制符剥离），同目录 text_clean.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_clean
_PUA_RE = re.compile('[\ue000-\uf8ff]')

import numpy as np
import requests

OLLAMA = 'http://localhost:11434/api/embed'
# ⚠️ 2026-09-27 默认值由 nomic-embed-text（768 维，已淘汰）改为现役 qwen3-embedding:0.6b（1024 维）。
#    这只影响「既没传 --model、--out 又不是已有库」的全新库；沿用已有库时以库的 config.model 为准。
MODEL = 'qwen3-embedding:0.6b'
# 实际生效的嵌入模型：若 --out 指向已有库，自动沿用其 config.model（防维度冲突）
EMB_MODEL = MODEL
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
OLD_MAX_CHUNKS = 15          # v1 的截断上限；仅作旧记录兜底值（记录内 max_chunks 缺失时按此判定）
DEFAULT_MAX_CHUNKS = 60

SUPPORTED_TEXT_EXT = {'.txt', '.md', '.m', '.c', '.h', '.cpp', '.hpp', '.py', '.json', '.csv', '.log',
                      '.tex', '.yml', '.yaml', '.xml', '.html', '.htm', '.sh', '.cmd', '.bat', '.ps1', '.sql'}
SUPPORTED_DOC_EXT = {'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls'}
# 图片类：OCR 文本缓存的消费白名单（仅在缓存存在且有正文时生效）
OCR_TEXT_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tif', '.tiff'}
# 二进制/非文本扩展名：**不解析正文，只记元数据**（2026-09-22 扩充，改进 #6）
# 依据：v8 里 154 个正文为空的片段，扩展名分布揭露了一批被白跑解析器的类型。
#   新增：.opj/.opju(Origin 工程) .nb(Mathematica) .circ(Logisim) .mmap(MindManager)
#         .cap(抓包) .ms14 .mp4 .avi .mkv .mov .iso .apk .jar .a .so .lib .obj
#   收益：入库时少跑无意义的解析 + 减少「空壳片段」占位干扰重排信噪比。
META_ONLY_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.swf', '.zip', '.rar', '.7z',
                 '.url', '.exe', '.dll', '.opj', '.opju', '.nb', '.circ', '.mmap', '.cap', '.ms14',
                 '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.iso', '.apk', '.jar', '.a',
                 '.so', '.lib', '.obj', '.bin', '.dat', '.db', '.pkl', '.pt', '.onnx', '.h5'}


def walk_files(roots):
    """多源根目录：返回所有 root 下文件的并集（去重）"""
    if isinstance(roots, str):
        roots = [roots]
    files = set()
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                files.add(os.path.join(dirpath, fn))
    return sorted(files)


def rel_of(fp, roots):
    """按就近匹配的 root 计算相对路径（保证同一根目录内 rel_path 稳定）"""
    best = None
    for r in roots:
        try:
            rp = os.path.relpath(fp, r)
        except Exception:
            continue
        if not rp.startswith('..'):
            if best is None or len(rp) < len(best):
                best = rp
    return best if best is not None else os.path.basename(fp)


def cache_path(fp, cache_dir):
    return os.path.join(cache_dir, hashlib.sha1(fp.encode('utf-8')).hexdigest() + '.txt')


# ---- PDF 文本提取：按坐标排序 + Symbol 字体映射（2026-09-22 新增，改进 #5）
# 动机：原先 `pg.get_text()` 取的是 PDF 内部文本流的**物理顺序**，数理资料里
#      公式的分子/分母/上下标会被交错读出，例如
#          「例如, 2 0 lim 3 x x x」        ← lim、x→0、x²/3x 全被拆散
#      实测同一文件加 sort=True 后变成
#          「例如, 当x → 0时, x , x ,sin x, x sin 都是无穷小.」
#      关键记号（x → 0）被正确保留，lim 与分式的位置关系可辨。
#      10 个 PDF 抽样：公式记号数 10 → 178（17.8 倍）、数学词命中 14 → 46（3.3 倍）。
# 零新增依赖：PyMuPDF 已在用，只是之前没传 sort=True。
_PDF_SYM_MAP = {
    # ---- 希腊字母
    '\uf061': 'α', '\uf062': 'β', '\uf063': 'γ', '\uf064': 'δ', '\uf065': 'ε',
    '\uf066': 'φ', '\uf067': 'γ', '\uf068': 'η', '\uf069': 'ι', '\uf06a': 'φ',
    '\uf06b': 'κ', '\uf06c': 'λ', '\uf06d': 'μ', '\uf06e': 'ν', '\uf06f': 'ο',
    '\uf070': 'π', '\uf071': 'θ', '\uf072': 'ρ', '\uf073': 'σ', '\uf074': 'τ',
    '\uf075': 'υ', '\uf076': 'ϖ', '\uf077': 'ω', '\uf078': 'ξ', '\uf079': 'ψ',
    '\uf07a': 'ζ',
    # ---- 运算符与括号（最高频！实测 = F03D×98、− F02D×81、+ F02B×74）
    '\uf03d': '=', '\uf02b': '+', '\uf02d': '−', '\uf02a': '*', '\uf02f': '/',
    '\uf03c': '<', '\uf03e': '>', '\uf028': '(', '\uf029': ')', '\uf05b': '[',
    '\uf05d': ']', '\uf07b': '{', '\uf07d': '}', '\uf02c': ',', '\uf02e': '.',
    '\uf03a': ':', '\uf03b': ';', '\uf04c': '⋯', '\uf0bc': '≅',
    # ---- 关系符
    '\uf0b1': '±', '\uf0b2': '≤', '\uf0a3': '≤', '\uf0b3': '≥', '\uf0a3': '≤',
    '\uf0b4': '×', '\uf0b8': '÷', '\uf0b9': '≠', '\uf0ba': '≡', '\uf0bb': '≈',
    '\uf0bd': '∝', '\uf0bf': '∞', '\uf0a5': '∞',
    # ---- 大运算符与微积分
    '\uf0e5': '∑', '\uf0e6': '∏', '\uf0f2': '∫', '\uf0f3': '∫', '\uf0b6': '∂',
    '\uf0d1': '∇', '\uf0d6': '√', '\uf0d7': '·', '\uf0d8': '¬',
    # ---- 箭头
    '\uf0ae': '→', '\uf0ac': '←', '\uf0ad': '↑', '\uf0af': '↓', '\uf0db': '⇔',
    '\uf0dc': '⇒', '\uf0de': '⇓', '\uf0dd': '⇑',
    # ---- 集合与逻辑
    '\uf0ce': '∈', '\uf0cf': '∉', '\uf0c6': '∪', '\uf0c8': '⊂', '\uf0cd': '⊆',
    '\uf0c7': '∩', '\uf0d9': '∧',     '\uf0da': '∨', '\uf0d0': '∠', '\uf0e0': '∠',
    '\uf0b0': '°',
    # ---- 零散残留（2026-09-22 按实测频次补：F0A2×16、F05C×8、F0EF×6、F051×5、F0EC/ED）
    '\uf0a2': '∙', '\uf05c': '\\', '\uf0ef': '⌠', '\uf051': '∇', '\uf0ec': '⌡',
    '\uf0ed': '⎯', '\uf02a': '*', '\uf022': '·', '\uf0f7': '÷',
}



def _pdf_page_text(pg):
    """单页 PDF → 文本：按坐标排序读，并把 Symbol 字体私用区字符还原为真符号。"""
    try:
        t = pg.get_text(sort=True)
    except Exception:
        t = pg.get_text()
    for k, v in _PDF_SYM_MAP.items():
        if k in t:
            t = t.replace(k, v)
    return t


def extract_text(path, cache_dir=None):
    """返回正文；优先命中 text_cache（含 OCR/COM 结果）"""
    if cache_dir:
        cp = cache_path(path, cache_dir)
        try:
            if os.path.exists(cp) and os.path.getsize(cp) > 0:
                t = open(cp, encoding='utf-8', errors='ignore').read()
                if t.strip():
                    return t
        except Exception:
            pass
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_TEXT_EXT:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    if ext == '.pdf':
        try:
            import fitz
            doc = fitz.open(path)
            t = '\n'.join(_pdf_page_text(pg) for pg in doc)
            doc.close()
            return t
        except Exception:
            return None
    if ext == '.docx':
        try:
            import docx
            d = docx.Document(path)
            return '\n'.join(p.text for p in d.paragraphs)
        except Exception:
            return None
    if ext == '.pptx':
        try:
            from pptx import Presentation
            prs = Presentation(path)
            parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        parts.append(shape.text_frame.text)
            return '\n'.join(parts)
        except Exception:
            return None
    if ext == '.xlsx':
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            parts = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    cells = [str(c) for c in row if c is not None]
                    if cells:
                        parts.append(' | '.join(cells))
            wb.close()
            return '\n'.join(parts)
        except Exception:
            return None
    if ext == '.xls':
        try:
            import xlrd
            book = xlrd.open_workbook(path)
            parts = []
            for sh in book.sheets():
                for r in range(sh.nrows):
                    cells = [str(c.value).strip() for c in sh.row(r)]
                    cells = [c for c in cells if c]
                    if cells:
                        parts.append(' | '.join(cells))
            return '\n'.join(parts)
        except Exception:
            return None
    return None


def chunk_text(text, max_chunks):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n and len(chunks) < max_chunks:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            cut = text.rfind('。', start + CHUNK_SIZE // 2, end)
            if cut == -1:
                cut = text.rfind('. ', start + CHUNK_SIZE // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def embed_texts(texts, batch=32, retries=4):
    """批量嵌入，单批失败自动重试（指数退避），重试耗尽则抛出异常"""
    vecs = []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        last = None
        for attempt in range(retries):
            try:
                r = requests.post(OLLAMA, json={'model': EMB_MODEL, 'input': part}, timeout=600)
                r.raise_for_status()
                vecs.extend(np.array(v, dtype=np.float32) for v in r.json()['embeddings'])
                last = None
                break
            except Exception as e:
                last = e
                time.sleep(min(2 ** attempt * 2, 30))
        if last is not None:
            raise last
    return vecs


def save_store(out, metadata, vectors):
    meta_path = os.path.join(out, 'metadata.json')
    vec_path = os.path.join(out, 'vectors.npy')
    tmp_m = meta_path + '.tmp'
    tmp_v = vec_path + '.tmp.npy'
    arr = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, 768), dtype=np.float32)
    with open(tmp_m, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
    np.save(tmp_v, arr)
    os.replace(tmp_m, meta_path)
    os.replace(tmp_v, vec_path)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, action='append',
                    help='源根目录，可重复传入以纳入多个源目录（共用同一向量库）')
    ap.add_argument('--roots-file', default='',
                    help='源根目录清单文件（每行一个，# 开头为注释），与 --root 合并')
    ap.add_argument('--out', default='vector_store')
    ap.add_argument('--rebuild', action='store_true', help='旧库改名备份后重建（不删除）')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--text-cache', default='', help='提取缓存目录（text_cache）')
    ap.add_argument('--max-chunks', type=int, default=DEFAULT_MAX_CHUNKS)
    ap.add_argument('--save-every', type=int, default=20)
    ap.add_argument('--keep-missing', action='store_true', help='保留失效源文件条目（默认清理）')
    ap.add_argument('--drop-shells', action='store_true',
                    help='清除纯二进制空壳占位片段（META_ONLY_EXT 且 text_len==0），2026-09-26 新增')
    ap.add_argument('--force-list', default='',
                    help='强制重嵌的文件路径清单（JSON 数组或每行一个路径），用于 OCR 文本回填后的覆盖式重嵌')
    ap.add_argument('--report', default='')
    ap.add_argument('--model', default='',
                    help='嵌入模型名；留空则自动沿用 --out 已有库的 config.model，都没有才用 qwen3-embedding:0.6b')
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    roots = [os.path.abspath(r) for r in args.root]
    if args.roots_file and os.path.exists(args.roots_file):
        for line in open(args.roots_file, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#'):
                roots.append(os.path.abspath(line))
    roots = list(dict.fromkeys(roots))
    for r in roots:
        print(f'源根目录: {r}  (存在={os.path.isdir(r)})')
    out = args.out if os.path.isabs(args.out) else os.path.join(base, args.out)
    os.makedirs(out, exist_ok=True)
    # --- 嵌入模型解析：显式 --model > 已有库 config.model > 默认 ---
    global EMB_MODEL
    _cfg_p = os.path.join(out, 'config.json')
    if os.path.exists(_cfg_p):
        try:
            import json as _json
            _m = _json.load(open(_cfg_p, encoding='utf-8')).get('model')
            if _m:
                EMB_MODEL = _m
                print(f'沿用已有库嵌入模型: {_m}')
        except Exception as _e:
            print(f'[警告] 读已有库 config 失败，用默认 {MODEL}: {_e}')
    if args.model:
        EMB_MODEL = args.model
        print(f'使用 --model 指定模型: {EMB_MODEL}')
    print(f'>>> 本次嵌入模型 = {EMB_MODEL}')
    meta_path = os.path.join(out, 'metadata.json')
    vec_path = os.path.join(out, 'vectors.npy')
    cache_dir = os.path.abspath(args.text_cache) if args.text_cache else None

    metadata, vectors = [], []
    if os.path.exists(meta_path) and not args.rebuild:
        metadata = json.load(open(meta_path, encoding='utf-8'))
        vectors = list(np.load(vec_path))
        print(f'载入已有向量库: {len(metadata)} 条')
    elif args.rebuild:
        ts = time.strftime('%Y%m%d_%H%M%S')
        for p in (meta_path, vec_path, os.path.join(out, 'config.json')):
            if os.path.exists(p):
                os.replace(p, p + f'.old_{ts}')
        print(f'旧库已改名备份为 *.old_{ts}，开始重建')

    files = walk_files(roots)
    if args.limit:
        files = files[:args.limit]

    idx = {}
    for i, m in enumerate(metadata):
        idx.setdefault(m.get('file_path', ''), []).append(i)

    # ---- 0) 强制重嵌清单（OCR 文本回填：源文件未变但缓存文本已更新）----
    forced = set()
    if args.force_list and os.path.exists(args.force_list):
        _raw = open(args.force_list, encoding='utf-8').read().strip()
        try:
            _arr = json.loads(_raw)
        except Exception:
            _arr = [ln.strip() for ln in _raw.splitlines() if ln.strip()]
        forced = {os.path.normcase(os.path.abspath(x)) for x in _arr}
        print(f'强制重嵌清单: {len(forced)} 个文件')

    # ---- 1) 失效源文件条目清理 ----
    missing = [fp for fp in idx if not os.path.exists(fp)]
    removed_missing = 0
    if missing and not args.keep_missing:
        invalid = set()
        for fp in missing:
            invalid.update(idx[fp])
        metadata = [m for i, m in enumerate(metadata) if i not in invalid]
        vectors = [v for i, v in enumerate(vectors) if i not in invalid]
        removed_missing = len(invalid)
        idx = {}
        for i, m in enumerate(metadata):
            idx.setdefault(m.get('file_path', ''), []).append(i)
        print(f'失效源文件条目清理: {len(missing)} 个文件 / {removed_missing} 条')

    # ---- 2) 判定待处理文件 ----
    todo, kept = [], 0
    reasons = {'new': 0, 'migrate': 0, 'changed': 0, 'truncated-fix': 0, 'new-text': 0, 'forced': 0}
    for fp in files:
        try:
            st = os.stat(fp)
        except Exception:
            continue
        sig = (int(st.st_mtime), int(st.st_size))
        if os.path.normcase(fp) in forced:
            todo.append((fp, sig, 'forced')); reasons['forced'] += 1; continue
        recs = idx.get(fp)
        if not recs:
            todo.append((fp, sig, 'new')); reasons['new'] += 1; continue
        m0 = metadata[recs[0]]
        if 'file_mtime' not in m0 or 'file_size' not in m0:
            todo.append((fp, sig, 'migrate')); reasons['migrate'] += 1; continue
        if (int(m0.get('file_mtime', -1)), int(m0.get('file_size', -1))) != sig:
            todo.append((fp, sig, 'changed')); reasons['changed'] += 1; continue
        # 2026-09-15 修复：只在「该文件上次写入时记录的上限 max_chunks < 本次请求上限」时才重嵌，
        # 避免同一批旧 15 块记录每轮被反复重处理（原逻辑只与固定旧常量 OLD_MAX_CHUNKS 比较 → 永不收敛）。
        # 2026-09-27 修正（P0-3）：判定由 len(recs)>=OLD_MAX_CHUNKS(15) 改为 >=args.max_chunks——
        # 旧写法把「实际块数 ≥15」的文件都当作疑似截断重嵌，每轮约 110 个文件被重复解析（无重复条目，仅费时）；
        # 正确语义是「实际块数已触及本次请求上限」才可能是截断，需按本次上限重嵌。
        # 旧记录无 max_chunks 字段时按 OLD_MAX_CHUNKS 兜底，一次性迁移重嵌后写回上限即收敛。
        if len(recs) >= args.max_chunks and args.max_chunks > int(m0.get('max_chunks', OLD_MAX_CHUNKS)):
            todo.append((fp, sig, 'truncated-fix')); reasons['truncated-fix'] += 1; continue
        if not m0.get('has_text') and cache_dir and os.path.exists(cache_path(fp, cache_dir)) \
                and os.path.getsize(cache_path(fp, cache_dir)) > 0:
            todo.append((fp, sig, 'new-text')); reasons['new-text'] += 1; continue
        kept += 1

    print(f'磁盘文件 {len(files)} | 待处理 {len(todo)} | 已是最新 {kept} | 明细 {reasons}')
    print(f'注意: 旧记录无 mtime 字段的将以 migrate 方式重嵌一次，之后即为真增量')

    # ---- 3) 处理 ----
    t0 = time.time()
    invalid = set()
    n = 0
    stat = {'text': 0, 'meta-only': 0, 'failed': 0}
    clean_stat = {'pua_after': 0, 'ctrl_after': 0, 'low_readable_after': 0}  # 2026-09-28 P1-5
    failed_files = []
    net_streak = 0
    aborted = False

    def compact_store():
        """应用 invalid 删除集并重建索引，保证 metadata 与 vectors 严格一一对应"""
        nonlocal metadata, vectors, idx, invalid
        if invalid:
            keep = [i for i in range(len(metadata)) if i not in invalid]
            metadata = [metadata[i] for i in keep]
            vectors = [vectors[i] for i in keep]
            invalid = set()
        idx = {}
        for i, m in enumerate(metadata):
            idx.setdefault(m.get('file_path', ''), []).append(i)

    for fp, sig, reason in todo:
        rel = rel_of(fp, roots)
        ext = os.path.splitext(fp)[1].lower()
        new_records, new_vecs = [], []
        try:
            # 2026-09-26 修复：META_ONLY_EXT 此前在增量路径**完全没被校验**（只有全量分支 L183 查了），
            #   导致 .swf/.opj/.circ/.nb/.jar/.exe 等 113 个纯二进制文件被白跑解析、
            #   还留下 has_text=False 的「空壳占位片段」→ 占索引位、干扰重排信噪比。
            #   现在：黑名单扩展名一律直接只记元数据，不进解析器。
            #   ⚠️ 但 .jpg/.png 等若已被 OCR（text_cache 有正文）仍要正常提取 → 所以先看缓存。
            _cached = cache_dir and os.path.exists(cache_path(fp, cache_dir)) \
                and os.path.getsize(cache_path(fp, cache_dir)) > 0
            if ext in META_ONLY_EXT and not _cached:
                text = None
            else:
                text = extract_text(fp, cache_dir) if (ext in SUPPORTED_TEXT_EXT | SUPPORTED_DOC_EXT
                                                        | OCR_TEXT_EXT) else None
            # 2026-09-28 P1-5（#139）：两个正文出口（缓存命中/现场解析）统一过正文清洗
            if text:
                text = text_clean.clean_text(text)
            # 2026-09-28 P0-4：现场解析结果写回 text_cache（补齐覆盖率；仅写非空正文）
            if text and text.strip() and cache_dir:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cache_path(fp, cache_dir), 'w', encoding='utf-8') as _cf:
                        _cf.write(text)
                except Exception:
                    pass
            if text and text.strip():
                chunks = chunk_text(text, args.max_chunks)
                if chunks:
                    # 2026-09-28 P1-5：清洗门禁统计（clean 后 PUA/控制符/低可读片段数）
                    _pua = _ctrl = _lows = 0
                    for _c in chunks:
                        _pua += len(_PUA_RE.findall(_c))
                        _ctrl += len(text_clean.CTRL_RE.findall(_c))
                        if text_clean.readable_ratio(_c) < 0.5:
                            _lows += 1
                    clean_stat['pua_after'] += _pua
                    clean_stat['ctrl_after'] += _ctrl
                    clean_stat['low_readable_after'] += _lows
                    new_vecs = embed_texts(chunks, args.batch)
                    for ci, ch in enumerate(chunks):
                        new_records.append({'file_path': fp, 'rel_path': rel, 'chunk_index': ci,
                                            'has_text': True, 'text_len': len(ch), 'text': ch,
                                            'file_mtime': sig[0], 'file_size': sig[1],
                                            'chunk_total': len(chunks), 'max_chunks': args.max_chunks})
                    stat['text'] += 1
                else:
                    text = None
            if not text or not text.strip():
                label = os.path.splitext(os.path.basename(fp))[0]
                new_vecs = embed_texts([label], args.batch)
                new_records.append({'file_path': fp, 'rel_path': rel, 'chunk_index': 0,
                                    'has_text': False, 'text_len': 0, 'text': '',
                                    'file_mtime': sig[0], 'file_size': sig[1], 'chunk_total': 1,
                                    'max_chunks': args.max_chunks})
                stat['meta-only'] += 1
        except Exception as e:
            net = isinstance(e, requests.exceptions.RequestException) or 'Connection' in str(e) \
                or 'Max retries' in str(e) or 'timed out' in str(e).lower()
            stat['failed'] += 1
            failed_files.append({'file_path': fp, 'rel_path': rel, 'reason': reason,
                                 'err': f'{type(e).__name__}: {str(e)[:100]}'})
            print(f'  失败[{reason}] {rel[:58]}: {type(e).__name__} {str(e)[:70]}', flush=True)
            if net:
                net_streak += 1
                if net_streak >= 5:
                    print('嵌入服务连续失败 5 次，本轮中止（旧记录完整保留，重跑即可续做）', flush=True)
                    aborted = True
            else:
                net_streak = 0
            n += 1
            if aborted:
                break
            continue
        net_streak = 0
        # 仅在嵌入成功后才移除旧记录，避免失败导致文件条目整体丢失
        for i in idx.get(fp, []):
            invalid.add(i)
        metadata.extend(new_records)
        vectors.extend(new_vecs)
        n += 1
        if n % args.save_every == 0:
            compact_store()
            arr = save_store(out, metadata, vectors)
            print(f'  进度 {n}/{len(todo)} | {len(metadata)} 条 | {(time.time()-t0)/60:.1f}min | 最近: {rel[:50]}',
                  flush=True)
    compact_store()

    # 2026-09-26 新增：--drop-shells 清掉「纯二进制占位」的旧空壳片段
    #   判据：扩展名在 META_ONLY_EXT 且 text_len==0（= 没 OCR 到正文的二进制文件）
    #   .jpg/.png 若有 OCR 正文（text_len>0）不受影响。
    if getattr(args, 'drop_shells', False):
        def _is_shell(m):
            e = os.path.splitext(m.get('file_path') or '')[1].lower()
            return e in META_ONLY_EXT and not (m.get('text_len') or 0)
        shells = [i for i, m in enumerate(metadata) if _is_shell(m)]
        if shells:
            drop = set(shells)
            keep = [i for i in range(len(metadata)) if i not in drop]
            metadata = [metadata[i] for i in keep]
            vectors = [vectors[i] for i in keep]
            idx = {}
            for i, m in enumerate(metadata):
                idx.setdefault(m.get('file_path', ''), []).append(i)
            print(f'[--drop-shells] 清除空壳占位片段 {len(shells)} 条 → 现存 {len(metadata)} 条')
        else:
            print('[--drop-shells] 无空壳片段')

    arr = save_store(out, metadata, vectors)
    cfg = {'model': EMB_MODEL, 'dim': int(arr.shape[1]), 'count': len(metadata),
           'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'max_chunks_per_file': args.max_chunks, 'text_cache': cache_dir or '',
           'roots': roots}
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    files_now = {m.get('file_path') for m in metadata}
    no_text = sum(1 for m in metadata if not m.get('has_text'))
    report = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'roots': roots,
        'vectors': len(metadata),
        'unique_files': len(files_now),
        'chunks_no_text': no_text,
        'missing_files': len(missing),
        'missing_records_removed': removed_missing,
        'todo_reasons': reasons,
        'processed': len(todo),
        'processed_stat': stat,
        'clean_stat': clean_stat,
        'kept': kept,
        'max_chunks': args.max_chunks,
        'elapsed_min': round((time.time() - t0) / 60, 1),
        'aborted': aborted,
        'failed_count': len(failed_files),
        'failed_files': failed_files[:80],
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        json.dump(report, open(args.report, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
