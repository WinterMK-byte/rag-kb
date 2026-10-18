# -*- coding: utf-8 -*-
"""【OCR 步】把新资料里的图片/扫描件批出正文，写进 text_cache_<批次名>\\。
每张一个 txt，key = sha1(绝对路径)。正文格式（必带标题头，否则路径词命中退化）：

    【资料】<相对路径>
    【文件名】<文件名>
    <OCR 正文>

用法：
    py -3.12 ocr_new.py <目标目录> [批次名]
    py -3.12 ocr_new.py "C:\\Users\\<you>\\Desktop\\Resource\\新资料" xinziliao

内存纪律：单 worker 串行；可用内存 < 3GB 直接停。实测 51 张 5.9 分钟。
前置：先停 Ollama 腾内存（OCR 与嵌入不并行）。
"""
import os, io, sys, time, hashlib, glob

KB = r'C:\Users\<you>\Desktop\Marvis产出\本地知识库'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def sha1_of(fp):
    return hashlib.sha1(os.path.abspath(fp).encode('utf-8')).hexdigest()


def mem_ok():
    try:
        import psutil
        return psutil.virtual_memory().available >= 1.5 * 1024 ** 3
    except Exception:
        return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = sys.argv[1]
    batch = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(target.rstrip('\\/')) or 'batch'
    outdir = os.path.join(KB, 'text_cache_%s' % batch)
    os.makedirs(outdir, exist_ok=True)

    files = []
    for dp, dn, fn in os.walk(target):
        for f in fn:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                files.append(os.path.join(dp, f))
    files.sort()
    print('目标目录: %s' % target)
    print('图片 %d 张 → %s' % (len(files), outdir))

    todo = [p for p in files if not os.path.exists(os.path.join(outdir, sha1_of(p) + '.txt'))]
    print('待处理 %d 张（已有缓存 %d 张，自动跳过）' % (len(todo), len(files) - len(todo)))
    if not todo:
        print('无新增，结束。')
        return
    if not mem_ok():
        print('!! 可用内存不足，按内存纪律中止（exit=2，重跑可续）。')
        sys.exit(2)

    t0 = time.time()
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    ok = fail = 0
    empty = []
    for i, fp in enumerate(todo, 1):
        if i % 5 == 0 and not mem_ok():
            print('!! 内存不足，已在第 %d 张中止（已完成的都落盘了，重跑可续）' % i)
            break
        try:
            res, _ = engine(fp)
            text = '\n'.join(r[1] for r in res) if res else ''
            mark = ''
            if not text.strip():
                empty.append(fp)
                mark = '  [空!]'
            rel = os.path.relpath(fp, os.path.dirname(target))
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
    if empty:
        print('空正文（公式图/纯图，检索贡献低，可接受）:')
        for p in empty[:10]:
            print('   ', os.path.basename(p))
    print()
    print('下一步: 生成 force-list 后跑 build_kb_v3.py 派生新库（见 skill kb-ocr-backfill 第 2 步）')


if __name__ == '__main__':
    main()
