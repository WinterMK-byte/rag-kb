# -*- coding: utf-8 -*-
"""夜间真冷回归矩阵驱动（c07fix 2026-10-02）
用途：跨进程逐轮全量真冷评测，取多轮分布而非单点，回答：
  A) v11-fix 全量 Hit@5 是否 ≥ 0.794（原基线）且不劣化
  B) v11-fix 相对 v10 的增益是否落在波动带内（±2.9pt）还是真实提升
用法（后台）：
  py -3.12 -X utf8 night_matrix.py --rounds 14 --store vector_store_v11 --tag v11fix
  py -3.12 -X utf8 night_matrix.py --rounds 7  --store vector_store_v10 --tag v10
每轮：subprocess 跑 kb_eval.py --cold --json <round.json> --vstore <store> --top 5
     -> 解析 summary -> 追加 summary.jsonl -> 下一轮（天然跨进程 = 真冷）
断点续跑：round json 已存在且合法 -> 跳过（直接汇总）。
产物目录：_scratch/夜间矩阵_c07fix_20261002/<tag>/round_%02d.json + summary.jsonl + driver.log
"""
import argparse, json, os, subprocess, sys, time, datetime

KB = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe"
OUT_ROOT = os.path.join(KB, "_scratch", "夜间矩阵_c07fix_20261002")


def log(msg):
    line = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=14)
    ap.add_argument("--store", default="vector_store_v11")
    ap.add_argument("--tag", default="v11fix")
    ap.add_argument("--kb-eval", default=os.path.join(KB, "kb_eval.py"))
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    tag_dir = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(tag_dir, exist_ok=True)
    summary_path = os.path.join(tag_dir, "summary.jsonl")
    driver_log = os.path.join(tag_dir, "driver.log")
    sys.stdout = open(driver_log, "a", encoding="utf-8")
    log("启动夜间矩阵：tag=%s store=%s rounds=%d top=%d" % (
        args.tag, args.store, args.rounds, args.top))

    done, todo = [], []
    for r in range(1, args.rounds + 1):
        rp = os.path.join(tag_dir, "round_%02d.json" % r)
        if os.path.exists(rp):
            try:
                with open(rp, encoding="utf-8") as f:
                    json.load(f)
                done.append(r)
                continue
            except Exception:
                log("round %02d json 损坏，重跑" % r)
        todo.append((r, rp))
    if done:
        log("断点续跑：已完成 %d 轮 %s，待跑 %d 轮" % (len(done), done, len(todo)))
    if not todo:
        log("全部轮次已完成，退出")
        return

    t_start = time.time()
    for r, rp in todo:
        t0 = time.time()
        cmd = [PY, "-X", "utf8", args.kb_eval, "--cold", "--json", rp,
               "--vstore", args.store, "--top", str(args.top), "--tag", args.tag]
        log("轮次 %02d/%02d 开始" % (r, args.rounds))
        try:
            p = subprocess.run(cmd, cwd=KB, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=2400)
        except subprocess.TimeoutExpired:
            log("轮次 %02d 超时(40min)，本轮标记失败" % r)
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"round": r, "ok": False, "err": "timeout",
                                    "seconds": 2400}, ensure_ascii=False) + "\n")
            continue
        dt = time.time() - t0
        try:
            with open(rp, encoding="utf-8") as f:
                data = json.load(f)
            s = data["summary"]
            s["round"] = r
            s["ok"] = True
            s["seconds"] = round(dt, 1)
            s["tag"] = args.tag
            s["store"] = args.store
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            log("轮次 %02d 完成 %.0fs ｜ Hit@1=%.1f%% Hit@3=%.1f%% Hit@5=%.1f%% MRR=%.3f 出错=%d"
                % (r, dt, s["hit@1"] * 100, s["hit@3"] * 100, s["hit@5"] * 100,
                   s["mrr"], s["n_err"]))
        except Exception as e:
            log("轮次 %02d 结果解析失败：%s" % (r, str(e)[:200]))
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"round": r, "ok": False,
                                    "err": str(e)[:200], "seconds": round(dt, 1)},
                                   ensure_ascii=False) + "\n")
    log("全部完成，总耗时 %.0f 分钟" % ((time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
