# -*- coding: utf-8 -*-
"""本地知识库重建/增量脚本（Resource 新结构版）
用法:
    python build_kb.py --root "C:\\Users\\<you>\\Desktop\\Resource" [--out vector_store] [--rebuild] [--limit N] [--batch 32]
依赖: numpy, requests, Ollama qwen3-embedding:0.6b 已加载; 文档解析库尽量齐全(pypdf/pdfplumber/python-docx/python-pptx/openpyxl)

⚠️ 本脚本是**早期单库版**，已被 build_kb_v2.py / build_kb_v3.py 取代（支持多库、--model、增量）。
   保留仅作参考；现行建库请用 build_kb_v3.py。
输出: <out>/metadata.json + vectors.npy + config.json（与 kb_search.py 格式兼容）
"""
import os, sys, json, time, argparse, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import requests

OLLAMA = 'http://localhost:11434/api/embed'
# ⚠️ 2026-09-27 由 nomic-embed-text（768 维）改为现役 qwen3-embedding:0.6b（1024 维）。
#    注意：本脚本无 --model 参数，改这里只影响「用它新建的库」；已有库仍按各自 config.json 绑定。
MODEL = 'qwen3-embedding:0.6b'
CHUNK_SIZE = 2000          # 每块字符数
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_FILE = 15   # 超大文件只取前 15 块，防向量爆炸

SUPPORTED_TEXT_EXT = {'.txt', '.md', '.m', '.c', '.h', '.py', '.json', '.csv', '.log', '.tex', '.yml', '.yaml'}
SUPPORTED_DOC_EXT = {'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls'}
# 二进制/非文本扩展名：不解析正文，只记元数据（2026-09-22 扩充，改进 #6；与 build_kb_v2 保持一致）
META_ONLY_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.swf', '.zip', '.rar', '.7z',
                 '.url', '.exe', '.dll', '.opj', '.opju', '.nb', '.circ', '.mmap', '.cap', '.ms14',
                 '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.iso', '.apk', '.jar', '.a',
                 '.so', '.lib', '.obj', '.bin', '.dat', '.db', '.pkl', '.pt', '.onnx', '.h5'}


def _bkv2_pdf_page(pg):
    """单页 PDF → 文本（sort=True + Symbol 映射）。单一真源在 build_kb_v2._pdf_page_text。"""
    try:
        import build_kb_v2 as _b2
        return _b2._pdf_page_text(pg)
    except Exception:
        try:
            return pg.get_text(sort=True)
        except Exception:
            return pg.get_text()


def walk_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            files.append(os.path.join(dirpath, fn))
    return sorted(files)


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_TEXT_EXT:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    if ext == '.pdf':
        # 2026-09-22 改进 #5：优先用 PyMuPDF(sort=True)+Symbol 映射。
        # 原先 pypdf 排在最前，而它不排序、不留公式结构（实测公式记号数只有 PyMuPDF 的 1/17）。
        try:
            import fitz
            doc = fitz.open(path)
            t = '\n'.join(_bkv2_pdf_page(pg) for pg in doc)
            doc.close()
            if t and t.strip():
                return t
        except Exception:
            pass
        for mod in ('pypdf', 'PyPDF2'):
            try:
                m = __import__(mod)
                reader = m.PdfReader(path)
                return '\n'.join((pg.extract_text() or '') for pg in reader.pages)
            except Exception:
                continue
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
    # .doc/.ppt/.xls 老格式与图片等：无法本地解析 → 仅元数据
    return None


def chunk_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n and len(chunks) < MAX_CHUNKS_PER_FILE:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            # 尽量在句子边界断开
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


def embed_batch(texts):
    """返回 list[np.ndarray]"""
    r = requests.post(OLLAMA, json={'model': MODEL, 'input': texts}, timeout=300)
    r.raise_for_status()
    return [np.array(v, dtype=np.float32) for v in r.json()['embeddings']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Resource 根目录')
    ap.add_argument('--out', default='vector_store')
    ap.add_argument('--rebuild', action='store_true', help='清空 out 后全量重建')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个文件(测试用)')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(args.root)
    out = args.out if os.path.isabs(args.out) else os.path.join(base, args.out)
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, 'metadata.json')
    vec_path = os.path.join(out, 'vectors.npy')

    metadata, vectors = [], []
    if os.path.exists(meta_path) and not args.rebuild:
        metadata = json.load(open(meta_path, 'r', encoding='utf-8'))
        vectors = list(np.load(vec_path))
        print(f'加载已有向量库: {len(metadata)} 条，继续增量')
    elif args.rebuild:
        for p in (meta_path, vec_path, os.path.join(out, 'config.json')):
            if os.path.exists(p):
                os.remove(p)
        print('已清空旧库，全量重建')

    done_paths = {m['file_path'] for m in metadata}
    files = walk_files(root)
    if args.limit:
        files = files[:args.limit]
    todo = [f for f in files if f not in done_paths]
    print(f'扫描到 {len(files)} 个文件, 待处理 {len(todo)} 个')

    t0 = time.time()
    n_done = 0
    for fp in todo:
        rel = os.path.relpath(fp, root)
        ext = os.path.splitext(fp)[1].lower()
        if ext in META_ONLY_EXT or ext not in SUPPORTED_TEXT_EXT | SUPPORTED_DOC_EXT:
            # 不可解析类型 → 仅元数据（用文件名做语义向量）
            label = os.path.splitext(os.path.basename(fp))[0]
            metadata.append({'file_path': fp, 'rel_path': rel, 'chunk_index': 0,
                             'has_text': False, 'text_len': 0, 'text': ''})
            vectors.append(embed_batch([label])[0])
            n_done += 1
            continue
        text = extract_text(fp)
        if not text or not text.strip():
            label = os.path.splitext(os.path.basename(fp))[0]
            metadata.append({'file_path': fp, 'rel_path': rel, 'chunk_index': 0,
                             'has_text': False, 'text_len': 0, 'text': ''})
            vectors.append(embed_batch([label])[0])
            n_done += 1
            continue
        chunks = chunk_text(text)
        for ci, ch in enumerate(chunks):
            metadata.append({'file_path': fp, 'rel_path': rel, 'chunk_index': ci,
                             'has_text': True, 'text_len': len(ch), 'text': ch})
        batch_vecs = embed_batch([c for c in chunks])
        vectors.extend(batch_vecs)
        n_done += 1
        if n_done % 20 == 0:
            print(f'  进度 {n_done}/{len(todo)} | 当前 {len(metadata)} 条 | '
                  f'{(time.time()-t0)/60:.1f}min | 最近: {rel[:60]}', flush=True)
            json.dump(metadata, open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False)
            np.save(vec_path, np.stack(vectors).astype(np.float32))

    arr = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, 768), dtype=np.float32)
    json.dump(metadata, open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False)
    np.save(vec_path, arr)
    cfg = {'model': MODEL, 'dim': int(arr.shape[1]), 'count': len(metadata),
           'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'完成: {len(metadata)} 条向量, 维度 {arr.shape[1]}, 耗时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()
