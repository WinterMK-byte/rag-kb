/* 工兵台 · 前端交互 */
(function () {
  'use strict';

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  const els = {
    srv: $('#srv-status'),
    modeBtns: $$('.mode-btn'),
    views: $$('.view'),
    form: $('#search-form'),
    q: $('#q'),
    mode: $('#mode'),
    scope: $('#scope'),
    history: $('#history'),
    empty: $('#empty'),
    skeleton: $('#skeleton'),
    error: $('#error'),
    results: $('#results'),
    metaLine: $('#meta-line'),
    metaStages: $('#meta-stages'),
    resultList: $('#result-list'),
    appsGroups: $('#apps-groups'),
    toast: $('#toast'),
    footTime: $('#foot-time'),
  };

  const HISTORY_KEY = 'gbdesk_history';
  let history = [];
  try { history = JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch (e) { history = []; }

  /* ---------- 模式切换 ---------- */
  els.modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      els.modeBtns.forEach((b) => {
        const on = b === btn;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      const view = btn.dataset.view;
      els.views.forEach((v) => v.classList.toggle('is-active', v.id === 'view-' + view));
      if (view === 'apps') loadApps();
      if (view === 'search') setTimeout(() => els.q.focus(), 60);
    });
  });

  /* ---------- 服务状态 ---------- */
  function setSrv(ok, txt) {
    els.srv.classList.remove('ok', 'err');
    els.srv.classList.add(ok ? 'ok' : 'err');
    els.srv.querySelector('.srv-txt').textContent = txt;
  }

  async function probe() {
    try {
      const r = await fetch('/api/apps');
      if (!r.ok) throw 0;
      const j = await r.json();
      setSrv(j.ok, '服务在线');
      return true;
    } catch (e) {
      setSrv(false, '服务离线');
      return false;
    }
  }

  /* ---------- 搜索 ---------- */
  function saveHistory(q) {
    history = [q, ...history.filter((x) => x !== q)].slice(0, 5);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch (e) {}
    renderHistory();
  }

  function renderHistory() {
    els.history.hidden = history.length === 0;
    els.history.innerHTML = '';
    history.forEach((q) => {
      const b = document.createElement('button');
      b.className = 'chip';
      b.textContent = q;
      b.addEventListener('click', () => { els.q.value = q; runSearch(); });
      els.history.appendChild(b);
    });
  }

  function setVisible(el, on) { el.hidden = !on; }

  function runSearch() {
    const q = els.q.value.trim();
    if (!q) return;
    saveHistory(q);
    setVisible(els.empty, false);
    setVisible(els.results, false);
    setVisible(els.error, false);
    setVisible(els.skeleton, true);

    const params = new URLSearchParams({ q, mode: els.mode.value, scope: els.scope.value });
    fetch('/api/search?' + params.toString())
      .then((r) => r.json())
      .then((j) => {
        setVisible(els.skeleton, false);
        if (!j.ok) throw new Error(j.err || '检索失败');
        renderResults(j);
      })
      .catch((e) => {
        setVisible(els.skeleton, false);
        setVisible(els.error, true);
        els.error.textContent = '检索出错：' + e.message;
      });
  }

  function fmtStage(v) {
    if (v === undefined || v === null || v === 0) return '—';
    return v < 1 ? Math.round(v * 1000) + 'ms' : v.toFixed(1) + 's';
  }

  function renderResults(j) {
    els.metaLine.textContent = `「${j.query}」 · ${j.mode} · 命中 ${j.results.length} 条 · 耗时 ${j.seconds}s`;
    const s = j.stages || {};
    els.metaStages.textContent = `改写 ${fmtStage(s.rewrite)} · 召回 ${fmtStage(s.recall)} · 重排 ${fmtStage(s.rerank)}`;

    els.resultList.innerHTML = '';
    j.results.forEach((it, i) => {
      const li = document.createElement('li');
      li.className = 'result-item';
      li.style.animationDelay = (i * 0.04) + 's';

      const head = document.createElement('div');
      head.className = 'ri-head';

      const num = document.createElement('span');
      num.className = 'ri-num';
      head.appendChild(num);

      const file = document.createElement('span');
      file.className = 'ri-file';
      file.textContent = it.file || '(未知来源)';
      file.title = it.path || it.file || '';
      file.addEventListener('click', () => openFile(it.path || it.file || ''));
      head.appendChild(file);

      const actions = document.createElement('div');
      actions.className = 'ri-actions';

      const openBtn = document.createElement('button');
      openBtn.className = 'icon-btn';
      openBtn.textContent = '打开';
      openBtn.addEventListener('click', () => openFile(it.path || it.file || ''));
      actions.appendChild(openBtn);

      const copyBtn = document.createElement('button');
      copyBtn.className = 'icon-btn';
      copyBtn.textContent = '复制';
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(it.text || '').then(() => toast('已复制片段')).catch(() => toast('复制失败', true));
      });
      actions.appendChild(copyBtn);

      head.appendChild(actions);

      const score = document.createElement('span');
      score.className = 'ri-score';
      score.innerHTML = 'score <b>' + it.score.toFixed(3) + '</b>';
      head.appendChild(score);

      const body = document.createElement('div');
      body.className = 'ri-body';
      const code = document.createElement('div');
      code.className = 'ri-code';
      code.textContent = it.text || '';
      body.appendChild(code);

      li.appendChild(head);
      li.appendChild(body);
      els.resultList.appendChild(li);
    });

    setVisible(els.results, true);
  }

  async function openFile(path) {
    if (!path) { toast('无文件路径', true); return; }
    try {
      const r = await fetch('/api/open?path=' + encodeURIComponent(path));
      const j = await r.json();
      if (j.ok) toast('已打开文件');
      else toast(j.err || '打开失败', true);
    } catch (e) {
      toast('打开失败', true);
    }
  }

  /* ---------- 应用台 ---------- */
  async function loadApps() {
    els.appsGroups.innerHTML = '';
    try {
      const r = await fetch('/api/apps');
      const j = await r.json();
      if (!j.ok) throw 0;
      const groups = {};
      j.apps.forEach((a) => {
        const g = a.tag || '其他';
        (groups[g] = groups[g] || []).push(a);
      });
      Object.entries(groups).forEach(([tag, list]) => {
        const block = document.createElement('div');
        block.className = 'app-group';
        const title = document.createElement('div');
        title.className = 'app-group-title';
        title.textContent = tag.toUpperCase();
        block.appendChild(title);
        list.forEach((a) => block.appendChild(makeAppRow(a)));
        els.appsGroups.appendChild(block);
      });
    } catch (e) {
      toast('应用台加载失败：请确认服务已启动', true);
    }
  }

  function makeAppRow(a) {
    const row = document.createElement('div');
    row.className = 'app-row';

    const glyph = document.createElement('div');
    glyph.className = 'app-glyph';
    glyph.textContent = (a.name || '?').slice(0, 1);
    row.appendChild(glyph);

    const info = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'app-name';
    name.textContent = a.name || '';
    const desc = document.createElement('div');
    desc.className = 'app-desc';
    desc.textContent = a.desc || '';
    info.appendChild(name);
    info.appendChild(desc);
    row.appendChild(info);

    const right = document.createElement('div');
    right.className = 'app-right';
    const tag = document.createElement('span');
    tag.className = 'app-tag';
    tag.textContent = a.tag || '';
    right.appendChild(tag);

    const btn = document.createElement('button');
    btn.className = 'launch-btn';
    btn.textContent = a.link ? '前往' : '启动';
    btn.addEventListener('click', () => {
      if (a.link) {
        els.modeBtns.find((b) => b.dataset.view === a.link.slice(1))?.click();
        return;
      }
      btn.disabled = true;
      btn.textContent = '启动中…';
      fetch('/api/launch?app=' + encodeURIComponent(a.id))
        .then((r) => r.json())
        .then((j) => {
          btn.disabled = false;
          btn.textContent = '启动';
          if (j.ok) toast('已启动：' + (j.name || a.name));
          else toast('启动失败：' + (j.err || ''), true);
        })
        .catch(() => { btn.disabled = false; btn.textContent = '启动'; toast('启动失败', true); });
    });
    right.appendChild(btn);

    row.appendChild(right);
    return row;
  }

  /* ---------- 工具 ---------- */
  let toastTimer = null;
  function toast(msg, isErr) {
    els.toast.hidden = false;
    els.toast.textContent = msg;
    els.toast.classList.toggle('err', !!isErr);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { els.toast.hidden = true; }, 2400);
  }

  function tickTime() {
    const d = new Date();
    els.footTime.textContent = d.toLocaleTimeString('zh-CN', { hour12: false });
  }

  /* ---------- 分类 ---------- */
  async function loadScopes() {
    try {
      const r = await fetch('/api/scopes');
      const j = await r.json();
      if (!j.ok || !j.scopes) return;
      const cur = els.scope.value;
      els.scope.innerHTML = '';
      j.scopes.forEach((s) => {
        const o = document.createElement('option');
        o.value = s.id;
        o.textContent = s.label;
        els.scope.appendChild(o);
      });
      if (j.scopes.some((s) => s.id === cur)) els.scope.value = cur;
    } catch (e) { /* 静默：保留默认选项 */ }
  }

  /* ---------- 启动 ---------- */
  renderHistory();
  loadScopes();
  tickTime();
  setInterval(tickTime, 1000);
  probe();
  setInterval(probe, 15000);

  els.form.addEventListener('submit', (e) => { e.preventDefault(); runSearch(); });
  els.q.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); runSearch(); }
  });

  const urlMode = new URLSearchParams(location.search).get('view');
  if (urlMode === 'apps') els.modeBtns.find((b) => b.dataset.view === 'apps')?.click();
})();
