"""运行实验：每道题 × 两种用户情境 × 五种评分指令 × 两种回答顺序。

    python -m syco.run --backend ollama --model qwen2.5:7b
    python -m syco.run --backend openai --model deepseek-chat --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY
    python -m syco.run --backend mock          # 只检查流程

结果逐条追加写入 results/raw_<评分者>.jsonl。中途中断后重新运行同一命令会从断点继续，之前失败的判断也会重试。
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data import build_items
from .prompts import CONDITIONS, build_prompt
from .judges import make_judge, ask_with_retry, MockJudge


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai", "mock"], default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama 地址")
    ap.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI 兼容接口地址")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="存放 API Key 的环境变量名")
    ap.add_argument("--n", type=int, default=200, help="抽取的题目数")
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help="逗号分隔的评分指令条件")
    ap.add_argument("--workers", type=int, default=2, help="并发请求数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    judge = make_judge(args)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    items = build_items(n=args.n, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"raw_{safe(judge.name)}.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("choice") is not None:  # 失败的判断（如连接中断）下次运行会自动重试
                    done.add((r["qid"], r["context"], r["condition"], r["sycophantic_first"]))

    jobs = [(it, c, sf) for it in items for c in conds for sf in (True, False)
            if (it["qid"], it["context"], c, sf) not in done]
    total = len(jobs) + len(done)
    print(f"评分者 {judge.name}｜{len(items) // 2} 道题 × 2 情境 × {len(conds)} 条件 × 2 顺序 = {total} 次判断，"
          f"已完成 {len(done)}，本次运行 {len(jobs)}")
    if not jobs:
        return

    lock = threading.Lock()
    t0, finished = time.time(), 0

    def work(job):
        it, c, sf = job
        prompt = build_prompt(it, c, sf)
        kw = {"condition": c, "context": it["context"], "sycophantic_first": sf} if isinstance(judge, MockJudge) else {}
        choice, raw = ask_with_retry(judge, prompt, **kw)
        picked_s = None if choice is None else ((choice == "A") == sf)
        return {"judge": judge.name, "qid": it["qid"], "context": it["context"], "condition": c,
                "sycophantic_first": sf, "choice": choice, "picked_sycophantic": picked_s, "raw": raw}

    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                finished += 1
                if finished % max(50, len(jobs) // 20) == 0 or finished == len(jobs):
                    rate = finished / max(time.time() - t0, 1e-9)
                    left = (len(jobs) - finished) / max(rate, 1e-9)
                    print(f"  {finished}/{len(jobs)}  约 {rate:.1f} 次/秒，剩余约 {left / 60:.0f} 分钟")
    print(f"完成，结果写入 {path}\n下一步：python -m syco.analyze {path}")


if __name__ == "__main__":
    main()
