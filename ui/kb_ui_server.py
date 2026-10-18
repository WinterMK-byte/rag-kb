# -*- coding: utf-8 -*-
"""
工兵台 · 本地双模式控制台后端
  - 模式一：知识库搜索（调 kb_ask / kb_search_v4 本地 RAG 检索）
  - 模式二：应用台（白名单应用一键启动）
用法：python kb_ui_server.py [--port 8765]
"""
import os
import sys
import json
import time
import threading
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KB_ROOT = r'C:\Users\<you>\Desktop\Marvis产出\本地知识库'
UI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KB_ROOT)

import kb_search_v4 as k4  # noqa: E402
import kb_ask as ka        # noqa: E402

PORT = 8765
MODES = ('fast', 'deep', 'turbo')

SCOPE_LABELS = {
    'auto': '自动判断', 'core': '默认域（考插+华理）', 'kaocha': '考插全部（数学+英语）',
    'gpa': '绩点全部（华理全科）', 'course': '绩点（旧名）', 'kmath': '考插数学',
    'ken': '考插英语', 'gmath': '华理数学', 'gen': '华理英语', 'gother': '其他学科',
    'math': '数学全量', 'en': '英语全量', 'mathdoc': '数学讲解白名单',
    'tech': '技术库（skills/robomaster）', 'all': '全库',
}


def scope_catalog():
    return [{'id': s, 'label': SCOPE_LABELS.get(s, s)} for s in k4.SCOPE_CHOICES]


def load_apps():
    p = os.path.join(UI_DIR, 'apps.json')
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def do_search(q, mode='fast', scope='all', top=8):
    mode = mode if mode in MODES else 'fast'
    kw = ka.mode_kwargs(mode, scope=scope, rr_parallel=True)
    t0 = time.time()
    r = k4.retrieve(q, **kw)
    dt = time.time() - t0
    stages = r.get('stages', {})
    rerank = stages.get('rerank', {})
    results = []
    for it in (r.get('results') or [])[:top]:
        results.append({
            'file': it.get('rel_path') or it.get('file_path') or '',
            'path': it.get('file_path') or '',
            'text': (it.get('text') or '')[:600],
            'score': round(float(it.get('rank_score') or 0), 4),
            'idx': it.get('idx'),
        })
    return {
        'ok': True,
        'query': q,
        'mode': mode,
        'scope': scope,
        'seconds': round(dt, 2),
        'stages': {
            'rewrite': rerank.get('seconds') if False else stages.get('rewrite', {}).get('seconds', 0),
            'recall': stages.get('recall', {}).get('seconds', 0),
            'rerank': rerank.get('seconds', 0),
            'total': round(dt, 2),
        },
        'top_files': rerank.get('top_files', [])[:top],
        'results': results,
    }


def launch_app(app_id):
    apps = load_apps()
    for a in apps:
        if a.get('id') == app_id:
            cmd = a.get('cmd') or ''
            if not cmd:
                return {'ok': False, 'err': 'no command'}
            try:
                if cmd.startswith('start '):
                    subprocess.Popen(cmd, shell=True)
                else:
                    subprocess.Popen(cmd, shell=False)
                return {'ok': True, 'name': a.get('name')}
            except Exception as e:
                return {'ok': False, 'err': str(e)[:120]}
    return {'ok': False, 'err': 'unknown app'}


def open_file(path):
    if not path:
        return {'ok': False, 'err': 'empty path'}
    path = os.path.abspath(path)
    if not path.lower().startswith(KB_ROOT.lower()):
        return {'ok': False, 'err': 'path out of KB root'}
    if not os.path.exists(path):
        return {'ok': False, 'err': 'file not exists'}
    try:
        os.startfile(path)  # noqa: S606
        return {'ok': True}
    except Exception as e:
        return {'ok': False, 'err': str(e)[:120]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path, ctype):
        if not os.path.isfile(path):
            self._send(404, {'ok': False, 'err': 'not found'})
            return
        with open(path, 'rb') as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if p == '/':
            return self._serve_file(os.path.join(UI_DIR, 'index.html'), 'text/html; charset=utf-8')
        if p == '/styles.css':
            return self._serve_file(os.path.join(UI_DIR, 'styles.css'), 'text/css; charset=utf-8')
        if p == '/app.js':
            return self._serve_file(os.path.join(UI_DIR, 'app.js'), 'application/javascript; charset=utf-8')
        if p == '/apps.json':
            return self._serve_file(os.path.join(UI_DIR, 'apps.json'), 'application/json; charset=utf-8')

        if p == '/api/scopes':
            return self._send(200, {'ok': True, 'scopes': scope_catalog()})

        if p == '/api/search':
            q = (qs.get('q') or [''])[0].strip()
            if not q:
                return self._send(400, {'ok': False, 'err': 'empty query'})
            mode = (qs.get('mode') or ['fast'])[0]
            scope = (qs.get('scope') or ['all'])[0]
            try:
                return self._send(200, do_search(q, mode, scope))
            except Exception as e:
                return self._send(500, {'ok': False, 'err': str(e)[:200]})

        if p == '/api/apps':
            return self._send(200, {'ok': True, 'apps': load_apps()})

        if p == '/api/launch':
            app_id = (qs.get('app') or [''])[0]
            return self._send(200, launch_app(app_id))

        if p == '/api/open':
            path = (qs.get('path') or [''])[0]
            return self._send(200, open_file(path))

        return self._send(404, {'ok': False, 'err': 'not found'})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=PORT)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('工兵台已启动: http://127.0.0.1:%d  (Ctrl+C 退出)' % args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
