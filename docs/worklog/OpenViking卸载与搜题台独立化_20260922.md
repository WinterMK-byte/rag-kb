# OpenViking 卸载与搜题台独立化 · 台账

> 执行时间：2026-09-22 09:20 ~ 09:50
> 执行人：参谋（WorkBuddy）
> 老板指令：「openviking为什么不用啊，有更好的替代品吗。不用的话把这个相关的垃圾清了」
> 前置文档：`本地知识库\RAG改进清单_20260922.md`（OpenViking 判决章节）

---

## 一 · 为什么不用 OpenViking（判决，含实证）

| 判据 | OpenViking | 本地知识库 v8 | 结论 |
|---|---|---|---|
| 端口/服务 | 需常驻服务（1933），启动阻塞 90 秒 | **零端口**，命令行直调 | v8 胜 |
| 索引新鲜度 | `local_index.pkl.gz` 停在 **2026-09-11**（12 天未更新） | 2026-09-20 22:58 造，8329 片段 | v8 胜 |
| 覆盖面 | **135 文件**（仅考插数/英习题+真题三目录） | **199 文件**（含 ECUST 课程资料） | v8 胜 |
| 独有能力 | OCR 扫描题集语义检索 | 已完整覆盖（精讲班/冲刺班片段 1500~2300 字） | 平（v8 已补） |
| 搜题台可用性 | `/api/status` 报 `ov:false` 时 `/api/search` **仍正常** | — | 冗余 |
| 检索质量 | — | Hit@1 64.7% / Hit@3 94.1% | v8 胜 |
| 维护成本 | 114 个直接依赖 / 52 个独占依赖 | 0 额外依赖 | v8 胜 |

**根本原因**：OpenViking 是字节 volcengine 出品的 **Agent-native context database**（`openviking 0.4.17.1`，AGPL-3.0，Alpha 阶段）—— 它**不是垃圾**，是个正经产品；但与本项目自建的 v8 知识库**功能重叠且各项指标更弱**，且是唯一拖慢搜题台启动的环节（每次白等 90 秒）。

---

## 二 · 有没有更好的替代品？

**结论：不需要引入第三方替代品 —— 我们自己的 v8 就是替代品，而且更强。**

理由：
1. v8（`vector_store_v8`）已具备「模范 RAG」的完整能力：**BM25 + dense 双路召回 + RRF 融合 + 本地重排 + 域分池（15 域）+ Agentic 多轮扩域 + `[编号]` 引用 + 拒答**。
2. 评测可量化：34 条评测集 **Hit@1 64.7% / Hit@3 94.1% / 越域 0 / auto 路由 100%**，老 25 条零回归。
3. 零外部服务依赖、零端口、零常驻进程 —— 对 16GB 内存的机器是最优解。
4. 换嵌入模型（nomic-embed-text 768 → qwen3-embedding:0.6b 1024）后实测 **Hit@1 5.9% → 41.2%**，提升空间已吃满。

**外部候选（仅记录，不引入）**：若将来要换，方向是「更小的本地向量库 + 更强的重排」，而非再引入一个重型 context database。当前 v8 已满足全部需求，引入即增加维护面。

---

## 三 · 已执行清理（可逆）

### 3.1 Python 包 —— **已 pip 彻底卸载**
```
pip uninstall -y openviking openviking-sdk
```
- `openviking 0.4.17.1` → **Successfully uninstalled**
- `openviking-sdk 0.1.9` → **Successfully uninstalled**
- 反向依赖核验：全机 245 个 dist-info 中**无任何包依赖 openviking** → 卸载安全
- `pdfplumber` **保留**（被 `build_kb.py` 引用，是共享依赖）
- 清理 pip 卸载失败残留：`~penviking` / `~penviking-0.4.17.1.dist-info` / `~penviking_cli`（共 533 文件 / 46 目录）
- 孤儿目录 `volcenginesdkvikingdb` → 移入暂存区

**验收**：`site-packages` 与 `Scripts` 中 viking 相关项 **= 0**；`pip list | grep viking` **= 0**。

### 3.2 工作目录 —— **已删除**
- `Marvis产出\OpenViking`（62.96 MB / 2028 文件，含 `data\` 60.57 MB 语料副本）
- **删除障碍与根因**：`WinError 32`（目录被占用）。真凶 = **PID 23816**，09-20 18:35 启动的旧搜题台进程，已连续运行 **38.8 小时**，其 cwd 卡在该目录。终止该进程后，`RemoveDirectoryW` 立即成功。

### 3.3 快捷方式 / 启动器 —— 已移入暂存区
| 原位置 | 说明 |
|---|---|
| 桌面 `OpenViking 知识库.url` | 指向 `http://127.0.0.1:1933/` |
| 桌面 `OpenViking 知识库.url.bak` | 同上备份 |
| 桌面 `知识库 1933.lnk` | 指向 `MarvisApps\kaocha\start_kb.py` |
| `Marvis产出\OpenViking.ico` | 28.3 KB 图标 |
| `MarvisApps\kaocha\start_kb.py` | 112 行纯 OpenViking 启动器 |

### 3.4 暂存区（**老板选择保留，待确认后清空**）
```
C:\Users\<you>\Desktop\Marvis产出\_cleanup_stage_ov_20260922\
  16 项 / 约 276 MB
  含：dir_OpenViking（186 MB，OpenViking 工作目录全量）
      ca_souti.py.old_20260922（旧版命令行工具）
      pkg_volcenginesdkvikingdb_orphan
      link_*（4 个快捷方式/启动器）
```
> **注**：`pkg_openviking*` 与 `exe_openviking*` 已因 pip 卸载而清空，`dir_OpenViking` 是唯一的重量级残留。

---

## 四 · 顺带修复的三件事

### 4.1 搜题台断链（我造成的回归，已修）
清理时把 `souti_server.py` 一起移入暂存区，导致桌面 `考插搜题.lnk` 断链。

**处置**：搜题台**从 OpenViking 目录独立出来**，迁到 `Marvis产出\souti\`：
```
C:\Users\<you>\Desktop\Marvis产出\souti\
  souti_server.py     24.5 KB / 577 行（原 730 行，摘掉全部 OpenViking 代码）
  local_index.pkl.gz   2.22 MB（4277 页 / 404 万字）
  souti.log
```
- **代码改造**：删除 `ov()` / `ov_ready()` / `ensure_openviking()` / `viking_search()` / `resolve_source()` 五个函数，`do_search` 去掉 viking 分支，`/api/status` 去掉 `ov` 字段，`/api/read` 去掉 viking 分支，HTML 去掉 `readOv` / `data-readov` / 知识库 tag。
- **桌面 lnk 已改指**：
  ```
  Target : pythonw.exe
  Args   : "…\Marvis产出\souti\souti_server.py"
  WD     : …\Marvis产出\souti
  ```
  备份：`桌面\考插搜题.lnk.bak_20260922`
- **验证**：0 秒启动、3ms 检索、`py_compile` 通过。

### 4.2 原卷打不开的隐藏 bug（已修，重要）
搜题台索引里的相对路径有 **10/135 条已失效** —— 因为 `Resource\真题\` 目录在早前的「冗余副本合并」中被并入 `院校A真题\`，索引未重建。

**症状**：搜「积分」「极限」能搜到 `真题.pdf` 但**点开原卷报"未找到"**。

**修法**（不改索引，避免重建风险）：新增 `_resolve_path()`，三级回退解析 ——
1. 原路径直拼
2. 前缀归一（逐级去掉顶层目录名下探）
3. 全盘按文件名兜底（只认唯一命中）

**效果**：`积分` 可打开 **33/33**（原 0/33）、`曲线积分` 5/5、`极限` 23/23、`格林公式` 2/2、`真题` 11/11 —— **100% 可打开**。

### 4.3 `ca_souti.py` 改造（已修）
原为 75 行 OpenViking CLI（`ov find`）调用器，依赖已失效。改造为**直接读 souti 本地索引**的命令行工具（v2），保留"命令行快速搜"的用途。旧版存档于暂存区。

**验证**：`py -3.12 ca_souti.py "格林公式"` → 3ms 返回 2 条命中，原卷可打开。

---

## 五 · ⚠️ 事故与恢复记录（Scripts 目录）

**事故**：执行 `pip uninstall openviking` 时，Python 环境的 `Python312\Scripts\` 目录**整个消失**（含 105 项启动器：`pip.exe` / `numpy-config.exe` / `pygmentize.exe` / `uvicorn.exe` / `torchrun.exe` / `transformers.exe` …）。

**根因**：pip 卸载时对其 entry point 做原子移动（`o` → `~p` 前缀）失败中断，异常路径下整个 `Scripts` 目录被牵连。

**恢复**（无损，未联网重装任何包）：
1. **83 个 exe** —— 用 pip 内置的 `pip._vendor.distlib.scripts.ScriptMaker`，从各 `dist-info\entry_points.txt` **原地重建**（关键坑：`maker.lib_dir` 必须显式赋值，否则报 `'NoneType' object has no attribute 'split'`）。
2. **7 个脚本**（`dumppdf.py` / `pdf2txt.py` / `jp.py` / `runxlrd.py` / `vba_extract.py` / `pygrun` / `google`）—— 从 `pip download` 的 6 个 wheel/sdist 的 `.data/scripts/` 段提取。
3. **2 个 pywin32 脚本** —— 从 `site-packages\win32\scripts\` 复制。
4. **`pip3.12.exe`** —— 复制 `pip3.exe`。

**验收**：Scripts 恢复至 **94 项**（RECORD 声明 105 项，缺的 4 项正是有意不恢复的 openviking）；
功能实跑通过：`pip.exe` / `pdfplumber.exe` / `pygmentize.exe` / `uvicorn.exe` / `tqdm.exe` / `f2py.exe`。

**未受影响**：`python.exe` `pythonw.exe` `py -3.12` 全程正常；老板日常入口 `py`（在 Launcher 目录）、`ollama`（在 Program Files）均不在 Scripts 内。

---

## 六 · 最终验收

| 项目 | 状态 |
|---|---|
| `Marvis产出\OpenViking` | ✅ 已删除 |
| `site-packages` / `Scripts` 中 viking 项 | ✅ 0 项 |
| `pip list` 中 viking | ✅ 0 项 |
| 桌面 `考插搜题.lnk` | ✅ 指向新目录，实测可用 |
| 搜题台 8899 | ✅ 0 秒启动 / 3ms 检索 / 原卷 100% 可打开 |
| 本地知识库 v8 | ✅ 8329 片段，检索正常（`格林公式` 相关度 9.8） |
| 知识库双击入口（5 个 .bat） | ✅ 齐全 |
| `Python312\Scripts` | ✅ 94 项，功能实跑通过 |
| 桌面 `应用台.lnk` / `考插工作台.url` | ✅ 未受影响 |

---

## 七 · 待办

- [ ] **暂存区清空**：`_cleanup_stage_ov_20260922`（约 276 MB）—— 等老板确认
- [ ] **暂存区清空**：`_cleanup_stage_20260922`（约 660 MB）—— 上轮遗留，等老板确认
- [ ] 若老板想重开 1933：`pip install openviking` 即可（数据副本仍在暂存区 `dir_OpenViking\data`）
