# -*- coding: utf-8 -*-
"""【只读侦察】新资料落地后第一步：定位新目录 + 判定「能不能进库 / 能不能被搜到」。

用法（win 上建议用 py -3.12，本脚本无 numpy 依赖，3.13 也能跑）：
    py -3.12 scan_new.py                       # 全盘对照，列出「该入库但不在库里」的目录
    py -3.12 scan_new.py "D:\\某新资料目录"      # 深查该目录
    py -3.12 scan_new.py 关键词                  # 按路径关键词筛

★ 不写、不删、不起 Ollama，秒级出结果。跑完再决定是否进回填流程。

判定三件事：
  1. 在不在库里（file_path 集合差集）
  2. 域归属（不在 SCOPE_RULES 路径词里 → 落 other → 默认 core 搜不到 = 域外不可达）
  3. 需不需要 OCR（图片类扩展名 = 默认只有元数据）
"""
import os, io, json, sys, collections

KB = r'C:\Users\<you>\Desktop\Marvis产出\本地知识库'
V9 = os.path.join(KB, 'vector_store_v9')
RES = r'C:\Users\<you>\Desktop\Resource'

# ---- 扫描白名单：只这些根目录「本就该入库」
SCAN_ROOTS = [
    RES,                                      # 课程资料主库（只读）
    os.path.join(KB, 'ECUST课程资料'),          # 知识库自带副本
    os.path.join(KB, '考插数学习题资料'),
    os.path.join(KB, '考插英语习题资料'),
    os.path.join(KB, '真题'),
    r'C:\Users\<you>\Desktop\Marvis产出\RAG技术知识库',
]
# ---- 递归时跳过的目录名（自家向量库/备份/缓存/元数据，不该入库）
SKIP_DIR = ('vector_store', '_backup', '_待删', '_cleanup_stage', '__pycache__',
            'cache', 'text_cache', '.git', '_archive', '.obsidian')

# ---- SCOPE_RULES 里出现过的路径词（域内判据）
SCOPE_WORDS = [
    '考插数学习题资料', '考插英语习题资料', 'RAG技术知识库', 'ECUST课程资料',
    'robomaster', 'transformer', 'skills', '保研', '真题',
    '院校B真题', '上交真题集', '院校A真题', '单词本', '语音克隆', 'GLM Chat',
    '数学学院', '物理学院', '信息科学与工程学院', '马克思主义学院', '外语学院',
]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
META_ONLY_EXT = {'.swf', '.opj', '.nb', '.circ', '.exe', '.dll', '.zip', '.rar',
                 '.7z', '.iso', '.mp4', '.avi', '.mkv', '.mp3', '.wav'}


def short(p):
    return p.replace(RES, 'Resource\\').replace(KB, '本地知识库\\')


def load_meta():
    return json.load(io.open(os.path.join(V9, 'metadata.json'), encoding='utf-8'))


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def scope_of_dir(d):
    """判定目录是否会落 other 域"""
    up = d
    return [w for w in SCOPE_WORDS if w in up]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    meta = load_meta()
    indexed = set(norm(m.get('file_path') or '') for m in meta)
    print('现役库 vector_store_v9：%d 片段 / %d 个文件' % (len(meta), len(indexed)))
    print()

    # ---------- 收集磁盘文件 ----------
    if args and os.path.isdir(args[0]):
        roots = [args[0]]
        kws = args[1:]
    else:
        roots = SCAN_ROOTS
        kws = args

    disk = []
    for r in roots:
        if not os.path.isdir(r):
            print('!! 根目录不存在:', r)
            continue
        for dp, dn, fn in os.walk(r):
            dn[:] = [x for x in dn if not x.startswith(SKIP_DIR)]
            for f in fn:
                disk.append(os.path.join(dp, f))
    print('扫描到磁盘文件 %d 个（根: %s）' % (len(disk), ' | '.join(short(x) for x in roots)))
    print()

    # ---------- 库外文件 ----------
    outside = [p for p in disk if norm(p) not in indexed]
    print('=' * 72)
    print('① 库外文件：%d 个' % len(outside))
    print('=' * 72)
    agg = collections.defaultdict(list)
    for p in outside:
        agg[os.path.dirname(p)].append(p)
    if not agg:
        print('  无 —— 全部已在索引里')
    for d, fs in sorted(agg.items(), key=lambda x: -len(x[1]))[:25]:
        words = scope_of_dir(d)
        flag = 'OK' if words else '❗域外'
        print('  %4d 文件  [%s]  %s' % (len(fs), flag, short(d)))
    print()

    # ---------- 关键词深查 ----------
    for kw in kws:
        print('=' * 72)
        print('② 深查「%s」' % kw)
        print('=' * 72)
        hits = [p for p in (disk if os.path.isdir(kw) else outside) if kw in p]
        if not hits:
            print('  无匹配')
            continue
        ext = collections.Counter(os.path.splitext(p)[1].lower() for p in hits)
        imgs = [p for p in hits if os.path.splitext(p)[1].lower() in IMG_EXT]
        metaonly = [p for p in hits if os.path.splitext(p)[1].lower() in META_ONLY_EXT]
        size = sum(os.path.getsize(p) for p in hits if os.path.exists(p))
        print('  文件 %d ｜ %.1f MB ｜ 图片(需OCR) %d ｜ 二进制(只记元数据) %d'
              % (len(hits), size / 1048576.0, len(imgs), len(metaonly)))
        print('  类型: %s' % dict(ext.most_common(8)))
        print()
        for p in sorted(hits)[:12]:
            print('    %s' % short(p))
        d0 = os.path.commonpath([os.path.dirname(p) for p in hits]) if len(hits) > 1 else os.path.dirname(hits[0])
        words = scope_of_dir(d0)
        print()
        if words:
            print('  ✅ 域判定：命中 %s → 可被默认域检索' % words)
        else:
            print('  ❗ 域判定：未命中任何 SCOPE_RULES 路径词 → 会落 other 域')
            print('     → 默认 core/kaocha 搜不到（只有 --scope all 可达）')
            print('     → 必须在 kb_search_v4.SCOPE_RULES 对应基础域加目录名 + 升 RR_SALT/ANS_SALT')
    print()
    print('=' * 72)
    print('下一步（若确认要入库）：')
    print('  1) 图片类 → 先 OCR 到 本地知识库\\text_cache_<新名>\\（key=sha1(绝对路径).txt，必带标题头）')
    print('  2) 派生新库 v10（复制 v9 → build_kb_v3.py --out vector_store_v10 --force-list ...）')
    print('  3) 改 SCOPE_RULES（若域外）+ 升两个盐 → 切 DEFAULT_VSTORE 到 v10')
    print('  4) python kb_eval.py 对比基线 Hit@1 67.6% 验证不回退')
    print('  详见 skill: kb-ocr-backfill')


if __name__ == '__main__':
    main()
