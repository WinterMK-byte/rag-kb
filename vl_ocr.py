# -*- coding: utf-8 -*-
"""【OCR 步 - VL 引擎】用 Ollama qwen2.5-vl 视觉模型批出图片正文（中文/表格/手写准度远高于 RapidOCR）。

接口与 ocr_new.py 一致：输出同格式 text_cache_<批次>/<sha1>.txt，正文带标题头：
    【资料】<相对路径>
    【文件名】<文件名>
    <OCR 正文>

用法：
    py -3.12 vl_ocr.py <目标目录> [批次名] [--engine vl|rapid] [--model qwen2.5-vl:7b] [--dry N]
示例：
    py -3.12 vl_ocr.py "C:\\Users\\<you>\\Desktop\\Resource\\新资料" xinziliao --dry 3
    py -3.12 vl_ocr.py "C:\\Users\\<you>\\Desktop\\Resource\\真题" zhenti --engine rapid

规则（2026-10-15 用户定调）：
- 默认走 Ollama VL（GPU，不跑 CPU）；RapidOCR 兜底仅 --engine rapid 显式指定。
- VL 调用失败/超时不偷偷降级，计入失败清单，末尾报告可重跑。
"""
import os, io, sys, time, hashlib, argparse, base64, json, urllib.request

KB = r'C:\Users\<you>\Desktop\Marvis产出\本地知识库'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
OLLAMA = 'http://localhost:11434'
PROMPT = (
    '请完整识别这张图片中的全部文字内容：标题、正文、公式符号、表格数字、手写批注都要尽量保留。'
    '按原始阅读顺序输出为纯文本，公式可保留为 LaTeX 形式。'
    '只输出识别到的文字内容本身，不要解释、不要客套、不要添加额外说明。'
    '如果图片没有任何文字，只输出【无文字】。'
)


def sha1_of(fp):
    return hashlib.sha1(os.path.abspath(fp).encode('utf-8')).hexdigest()


def vl_ocr_one(fp, model, timeout=240, max_side=1568, num_ctx=8192):
    """调用本地 Ollama 视觉模型识别单张图片，返回正文文本。
    图片先等比缩放到长边 <= max_side（qwen2.5vl 视觉 token 与分辨率强相关，缩小后大幅省显存/context）。
    """
    img_bytes = _prepare_image(fp, max_side)
    b64 = base64.b64encode(img_bytes).decode()
    payload = {
        'model': model,
        'prompt': PROMPT,
        'images': [b64],
        'stream': False,
        'options': {'num_ctx': num_ctx, 'temperature': 0},
    }
    req = urllib.request.Request(
        OLLAMA + '/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode('utf-8'))
    text = (data.get('response') or '').strip()
    if text == '【无文字】':
        return ''
    return text


def _prepare_image(fp, max_side=1568):
    """读取图片并等比压缩（长边 <= max_side，PNG 转 JPEG 省体积）。返回 JPEG bytes。"""
    try:
        from PIL import Image
        im = Image.open(fp)
        im = im.convert('RGB')
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=92)
        return buf.getvalue()
    except Exception:
        # PIL 不可用或读取失败：回退原始文件字节（仅当确实无法缩放时才走原图）
        with open(fp, 'rb') as f:
            return f.read()


def rapid_ocr_one(fp):
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    res, _ = engine(fp)
    return '\n'.join(r[1] for r in res) if res else ''


def main():
    ap = argparse.ArgumentParser(description='VL/Rapid OCR 批量识别图片正文')
    ap.add_argument('target', help='目标目录（递归扫描图片）')
    ap.add_argument('batch', nargs='?', default=None, help='批次名，默认取目录名')
    ap.add_argument('--engine', default='vl', choices=['vl', 'rapid'])
    ap.add_argument('--model', default='qwen2.5vl:7b')
    ap.add_argument('--dry', type=int, default=0, help='仅试跑前 N 张（0=全量）')
    args = ap.parse_args()

    batch = args.batch or os.path.basename(args.target.rstrip('\\/')) or 'batch'
    outdir = os.path.join(KB, 'text_cache_%s' % batch)
    os.makedirs(outdir, exist_ok=True)

    files = []
    for dp, dn, fn in os.walk(args.target):
        for f in fn:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                files.append(os.path.join(dp, f))
    files.sort()
    print('目标目录: %s' % args.target)
    print('引擎: %s（%s）｜ 图片 %d 张 → %s' % (args.engine, args.model, len(files), outdir))

    todo = [p for p in files if not os.path.exists(os.path.join(outdir, sha1_of(p) + '.txt'))]
    print('待处理 %d 张（已有缓存 %d 张，自动跳过）' % (len(todo), len(files) - len(todo)))
    if not todo:
        print('无新增，结束。')
        return
    if args.dry:
        todo = todo[:args.dry]
        print('--dry 试跑 %d 张' % len(todo))

    t0 = time.time()
    ok = fail = 0
    empty = []
    for i, fp in enumerate(todo, 1):
        try:
            if args.engine == 'vl':
                text = vl_ocr_one(fp, args.model)
            else:
                text = rapid_ocr_one(fp)
            mark = ''
            if not text.strip():
                empty.append(fp)
                mark = '  [空!]'
            rel = os.path.relpath(fp, os.path.dirname(args.target))
            body = '【资料】%s\n【文件名】%s\n%s' % (rel.replace('\\', '/'),
                                                  os.path.basename(fp), text)
            with io.open(os.path.join(outdir, sha1_of(fp) + '.txt'), 'w', encoding='utf-8') as f:
                f.write(body)
            ok += 1
            print('  [%d/%d] %s  %d 字%s' % (i, len(todo), os.path.basename(fp), len(text), mark))
        except Exception as e:
            fail += 1
            print('  [%d/%d] 失败 %s: %s' % (i, len(todo), os.path.basename(fp), e))
    dt = time.time() - t0
    print()
    print('完成: 成功 %d / 失败 %d / 空正文 %d ｜ 用时 %.1f 分钟' % (ok, fail, len(empty), dt / 60))
    if fail:
        print('失败 %d 张未降级（规则：不偷偷跑低准度引擎），重跑本命令可续。' % fail)
    if empty:
        print('空正文（公式图/纯图，检索贡献低，可接受）:')
        for p in empty[:10]:
            print('   ', os.path.basename(p))
    print()
    print('下一步: 生成 force-list 后跑 build_kb_v3.py 派生新库（与 ocr_new.py 相同衔接）')


if __name__ == '__main__':
    main()
