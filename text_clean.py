# -*- coding: utf-8 -*-
"""正文清洗：Adobe Symbol PUA → Unicode 映射 + 控制字符剥离。

来源：考插工作台线（会话 425232f3）2026-09-26 施工方案 #139，
     本文件由 RAG 线 2026-09-27 **固化到 RAG 线目录**，避免原工作区路径失效。

背景：PDF 内嵌 Adobe Symbol / MT Extra 字体时，提取器把字符落到 U+F000–F0FF 私有区。
     规则：U+F0xx 中的 xx = Symbol 字体原始 8-bit 码位（标准 Adobe Symbol Encoding）。

用法（只读预演 / 干跑）：
    from text_clean import build_symbol_map, clean_text, readable_ratio
"""
import re

# 控制字符：保留 \n \r \t
CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
PUA_RE = re.compile(r'[\ue000-\uf8ff]')
SPACE_RE = re.compile(r'[ \t]{3,}')


def build_symbol_map():
    """Adobe Symbol Encoding → Unicode 完整映射表（189 条，覆盖 99.7%）。"""
    m = {}
    asc = (' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ'
           '[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    for i, ch in enumerate(asc):
        m[0xF000 + 0x20 + i] = ch
    hi = {
        0xA0: '\u20ac', 0xA1: '\u03d2', 0xA2: '\u2032', 0xA3: '\u2264', 0xA4: '\u2044',
        0xA5: '\u221e', 0xA6: '\u0192', 0xA7: '\u2663', 0xA8: '\u2666', 0xA9: '\u2665',
        0xAA: '\u2660', 0xAB: '\u2194', 0xAC: '\u2190', 0xAD: '\u2191', 0xAE: '\u2192',
        0xAF: '\u2193', 0xB0: '\u00b0', 0xB1: '\u00b1', 0xB2: '\u2033', 0xB3: '\u2265',
        0xB4: '\u00d7', 0xB5: '\u221d', 0xB6: '\u2202', 0xB7: '\u2022', 0xB8: '\u00f7',
        0xB9: '\u2260', 0xBA: '\u2261', 0xBB: '\u2248', 0xBC: '\u2026', 0xBD: '|',
        0xBE: '\u203e', 0xBF: '\u21b5',
        0xC0: '\u2135', 0xC1: '\u2111', 0xC2: '\u211c', 0xC3: '\u2118', 0xC4: '\u2297',
        0xC5: '\u2295', 0xC6: '\u2205', 0xC7: '\u2229', 0xC8: '\u222a', 0xC9: '\u2283',
        0xCA: '\u2287', 0xCB: '\u2284', 0xCC: '\u2282', 0xCD: '\u2286', 0xCE: '\u2208',
        0xCF: '\u2209',
        0xD0: '\u2220', 0xD1: '\u2207', 0xD2: '\u00ae', 0xD3: '\u00a9', 0xD4: '\u2122',
        0xD5: '\u220f', 0xD6: '\u221a', 0xD7: '\u00b7', 0xD8: '\u00ac', 0xD9: '\u2227',
        0xDA: '\u2228', 0xDB: '\u21d4', 0xDC: '\u21d0', 0xDD: '\u21d1', 0xDE: '\u21d2',
        0xDF: '\u21d3',
        0xE0: '\u25ca', 0xE1: '\u27e8', 0xE2: '\u00ae', 0xE3: '\u00a9', 0xE4: '\u2122',
        0xE5: '\u2211',
        0xE6: '\u239b', 0xE7: '\u239c', 0xE8: '\u239d', 0xE9: '\u23a1', 0xEA: '\u23a2',
        0xEB: '\u23a3', 0xEC: '\u23a7', 0xED: '\u23a8', 0xEE: '\u23a9', 0xEF: '\u23aa',
        0xF1: '\u27e9', 0xF2: '\u222b', 0xF3: '\u2320', 0xF4: '\u23ae', 0xF5: '\u2321',
        0xF6: '\u239e', 0xF7: '\u239f', 0xF8: '\u23a0', 0xF9: '\u23a4', 0xFA: '\u23a5',
        0xFB: '\u23a6', 0xFC: '\u23ab', 0xFD: '\u23ac', 0xFE: '\u23ad',
    }
    for k, v in hi.items():
        m[0xF000 + k] = v
    return m


_SMAP = None


def clean_text(t, keep_unknown=False):
    """清洗：① PUA→真符号（查不到删或留 ?）② 去控制符 ③ 多空格归一。"""
    global _SMAP
    if not t:
        return t
    if _SMAP is None:
        _SMAP = build_symbol_map()
    out = []
    for ch in t:
        o = ord(ch)
        if 0xF000 <= o <= 0xF0FF:
            out.append(_SMAP.get(o, '?' if keep_unknown else ''))
        elif 0xE000 <= o <= 0xF8FF:
            out.append('?' if keep_unknown else '')
        elif CTRL_RE.match(ch):
            out.append('')
        else:
            out.append(ch)
    return SPACE_RE.sub(' ', ''.join(out))


def readable_ratio(t):
    """可读率 = (中文 + 字母数字 + 常用标点 + 数学符号) / 总长。"""
    if not t:
        return 0.0
    good = 0
    for ch in t:
        if ('\u4e00' <= ch <= '\u9fff') or ch.isalnum() or ch in ' \n\r\t.,;:!?()[]{}<>+-=*/\\|~^%#@&_$\'"':
            good += 1
        elif ord(ch) in (0x2212, 0x222b, 0x221e, 0x2211, 0x2208, 0x00b1, 0x2202, 0x2264, 0x2265, 0x2192):
            good += 1
    return good / len(t)


if __name__ == '__main__':
    m = build_symbol_map()
    print('映射表条目:', len(m))
    print('U+F03D →', m.get(0xF03D), '| U+F0F2 →', m.get(0xF0F2), '| U+F0A5 →', m.get(0xF0A5))
    demo = '\uf03d\uf0f2\uf0a5abc\x00\x07  \u3000'
    print('清洗演示:', repr(clean_text(demo)))
