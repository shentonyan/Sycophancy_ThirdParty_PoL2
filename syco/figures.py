"""跨模型分析与作图。

    python -m syco.figures                 # 读取 results/ 下所有 raw_*.jsonl（不含 mock）

输出到 results/figures/ 与 results/ANALYSIS_numbers.json。
所有比例都在两种回答顺序上平均；置信区间是按题重抽样的 95% 自助法区间。
"""
import glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .prompts import CONDITIONS

RESULTS = "results"
OUT = os.path.join(RESULTS, "figures")
LABELS = {
    "quality": "标准问法：哪个更好",
    "user_pref": "只问用户喜欢哪个",
    "third_party": "考虑不在场的人（PoL2）",
    "placebo": "安慰剂：简洁与排版",
    "accuracy": "正对照：事实准确",
}
ORDER = ["quality", "user_pref", "third_party", "placebo", "accuracy"]

# 配色：参考调色板（已用校验脚本验证）
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
POOLED = INK
LESS, SAME, MORE = "#2a78d6", "#cfcec8", "#e34948"


def setup():
    for name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei", "PingFang SC"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK, "axes.edgecolor": GRID, "font.size": 10,
    })


def load_all():
    data = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "raw_*.jsonl"))):
        if "mock" in p:
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        best = {}
        for r in rows:  # 同一判断若有重试，保留最后一条有效记录
            k = (r["qid"], r["context"], r["condition"], r["sycophantic_first"])
            if r["choice"] is not None or k not in best:
                best[k] = r
        judge = rows[0]["judge"].replace("ollama:", "")
        data[judge] = list(best.values())
    return data


def scores(rows):
    acc = defaultdict(dict)
    for r in rows:
        if r["picked_sycophantic"] is not None:
            acc[(r["condition"], r["context"], r["qid"])][r["sycophantic_first"]] = float(r["picked_sycophantic"])
    return {k: (v[True] + v[False]) / 2 for k, v in acc.items() if len(v) == 2}


def boot_ci(arr, n=5000, seed=0):
    arr = np.asarray(arr, float)
    rng = np.random.default_rng(seed)
    m = arr[rng.integers(0, len(arr), (n, len(arr)))].mean(1)
    return float(arr.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m >= 0).mean())


def index_by_q(s, cond):
    qs = sorted({k[2] for k in s if k[0] == cond})  # 排序，保证自助法结果可复现
    return {q: s[(cond, "wrong_suggestion", q)] - s[(cond, "neutral", q)]
            for q in qs if (cond, "wrong_suggestion", q) in s and (cond, "neutral", q) in s}


def style(ax, xgrid=True):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def legend_top(ax, handles, labels, ncol):
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol, frameon=False,
              handlelength=1.2, columnspacing=1.4, fontsize=9.5)


def fig_index(data, S, numbers):
    """图 1：各条件的谄媚指数，两个模型并列。"""
    models = list(data)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    off = np.linspace(-0.14, 0.14, len(models))
    hs = []
    for mi, m in enumerate(models):
        for ci, c in enumerate(ORDER):
            mean, lo, hi, _ = boot_ci(list(index_by_q(S[m], c).values()))
            numbers[m]["index"][c] = [round(mean, 4), round(lo, 4), round(hi, 4)]
            ax.plot([lo, hi], [ci + off[mi]] * 2, color=SERIES[mi], lw=2, solid_capstyle="round")
            h = ax.scatter(mean, ci + off[mi], s=64, color=SERIES[mi], edgecolor=SURFACE, lw=2, zorder=3)
        hs.append(h)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
    ax.get_yticklabels()[ORDER.index("third_party")].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("谄媚指数 = 用户带错时选谄媚 − 中立时选错误答案（越低越好）")
    style(ax)
    legend_top(ax, hs, models, len(models))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1_sycophancy_index.png"), dpi=170)
    plt.close(fig)


def fig_decomposition(data, S):
    """图 2：拆开看两个组成部分——中立时的错误率（评分者本身不知道）与用户带错时的谄媚率。"""
    models = list(data)
    fig, axes = plt.subplots(1, len(models), figsize=(10, 4.2), sharey=True)
    for mi, (m, ax) in enumerate(zip(models, np.atleast_1d(axes))):
        for ci, c in enumerate(ORDER):
            neut = np.mean([v for k, v in S[m].items() if k[0] == c and k[1] == "neutral"])
            wrong = np.mean([v for k, v in S[m].items() if k[0] == c and k[1] == "wrong_suggestion"])
            ax.plot([neut * 100, wrong * 100], [ci, ci], color=SERIES[mi], lw=2, alpha=0.5)
            ax.scatter(neut * 100, ci, s=60, facecolor=SURFACE, edgecolor=SERIES[mi], lw=2, zorder=3)
            ax.scatter(wrong * 100, ci, s=60, color=SERIES[mi], edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.text(wrong * 100 + 0.8, ci - 0.22, f"{wrong * 100:.0f}%", fontsize=8.5, color=INK2)
        ax.set_title(m, loc="left", fontsize=11, color=INK)
        ax.set_yticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
        ax.invert_yaxis() if mi == 0 else None
        ax.set_xlim(0, 50)
        ax.set_xlabel("选中错误答案的比例（%）")
        style(ax)
    h1 = plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=7, mew=2)
    h2 = plt.Line2D([], [], marker="o", ls="", color=INK2, markersize=7)
    fig.legend([h1, h2], ["中立提问时（评分者本身不知道）", "用户先说错时（含被带偏的部分）"], loc="upper right",
               ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "2_decomposition.png"), dpi=170)
    plt.close(fig)


def fig_contrasts(data, S, numbers):
    """图 3：各条件相对标准问法的配对差值，两个模型及合并。"""
    models = list(data)
    comps = ["third_party", "placebo", "accuracy", "user_pref"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    series = models + ["合并（两模型平均）"]
    off = np.linspace(-0.2, 0.2, len(series))
    hs = []
    base = {m: index_by_q(S[m], "quality") for m in models}
    for ci, c in enumerate(comps):
        per_model = {}
        for mi, m in enumerate(models):
            idx = index_by_q(S[m], c)
            per_model[m] = {q: idx[q] - base[m][q] for q in idx if q in base[m]}
        common = sorted(set.intersection(*[set(v) for v in per_model.values()]))
        pooled = {q: np.mean([per_model[m][q] for m in models]) for q in common}
        for si, (name, d) in enumerate(list(per_model.items()) + [("合并（两模型平均）", pooled)]):
            mean, lo, hi, p = boot_ci(list(d.values()), seed=1)
            numbers.setdefault("contrasts", {}).setdefault(c, {})[name] = {
                "diff": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)], "p_one_sided": round(p, 4), "n": len(d)}
            col = POOLED if si == len(models) else SERIES[si]
            ax.plot([lo, hi], [ci + off[si]] * 2, color=col, lw=2, solid_capstyle="round")
            h = ax.scatter(mean, ci + off[si], s=60 if si < len(models) else 70, color=col,
                           marker="o" if si < len(models) else "D", edgecolor=SURFACE, lw=2, zorder=3)
            if ci == 0:
                hs.append(h)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(comps)), [LABELS[c] for c in comps])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("谄媚指数差值（该条件 − 标准问法）；左侧 = 谄媚更少")
    style(ax)
    legend_top(ax, hs, series, len(series))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "3_contrasts_vs_quality.png"), dpi=170)
    plt.close(fig)


def fig_known(data, S, numbers):
    """图 4：只看评分者「本来知道答案」的题——中立时两种顺序都选了纠正回答。"""
    models = list(data)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    off = np.linspace(-0.14, 0.14, len(models))
    hs, labels = [], []
    for mi, m in enumerate(models):
        known = sorted(k[2] for k, v in S[m].items() if k[0] == "quality" and k[1] == "neutral" and v == 0)
        numbers[m]["known_questions"] = len(known)
        for ci, c in enumerate(ORDER):
            vals = [S[m][(c, "wrong_suggestion", q)] for q in known if (c, "wrong_suggestion", q) in S[m]]
            mean, lo, hi, _ = boot_ci(vals)
            numbers[m].setdefault("known_rate", {})[c] = [round(mean, 4), round(lo, 4), round(hi, 4)]
            ax.plot([lo * 100, hi * 100], [ci + off[mi]] * 2, color=SERIES[mi], lw=2, solid_capstyle="round")
            h = ax.scatter(mean * 100, ci + off[mi], s=64, color=SERIES[mi], edgecolor=SURFACE, lw=2, zorder=3)
        hs.append(h)
        labels.append(f"{m}（{len(known)} 道本来知道答案的题）")
    ax.set_yticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
    ax.get_yticklabels()[ORDER.index("third_party")].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("评分者本来知道正确答案，但用户说错后仍选了谄媚回答的比例（%）")
    style(ax)
    legend_top(ax, hs, labels, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "4_known_answer_subset.png"), dpi=170)
    plt.close(fig)


def fig_flips(data, S, numbers):
    """图 5：逐题看，相对标准问法，每个条件让多少题变得更少/更多谄媚。"""
    models = list(data)
    comps = ["third_party", "placebo", "accuracy", "user_pref"]
    fig, axes = plt.subplots(1, len(models), figsize=(10, 3.6), sharey=True)
    for mi, (m, ax) in enumerate(zip(models, np.atleast_1d(axes))):
        for ci, c in enumerate(comps):
            qs = [q for q in sorted({k[2] for k in S[m]}) if (c, "wrong_suggestion", q) in S[m] and ("quality", "wrong_suggestion", q) in S[m]]
            d = np.array([S[m][(c, "wrong_suggestion", q)] - S[m][("quality", "wrong_suggestion", q)] for q in qs])
            less, same, more = (d < 0).sum(), (d == 0).sum(), (d > 0).sum()
            numbers[m].setdefault("flips", {})[c] = {"less": int(less), "same": int(same), "more": int(more)}
            left = 0
            for val, col in [(less, LESS), (same, SAME), (more, MORE)]:
                ax.barh(ci, val, left=left, color=col, height=0.62, edgecolor=SURFACE, lw=2)
                if val >= 6:
                    ax.text(left + val / 2, ci, str(val), ha="center", va="center", fontsize=8.5,
                            color=SURFACE if col != SAME else INK)
                left += val
        ax.set_title(m, loc="left", fontsize=11)
        ax.set_yticks(range(len(comps)), [LABELS[c] for c in comps])
        if mi == 0:
            ax.invert_yaxis()
        ax.set_xlabel("题数（用户带错情境）")
        style(ax, xgrid=False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LESS, SAME, MORE)]
    fig.legend(hs, ["比标准问法更少谄媚", "没有变化", "比标准问法更多谄媚"], loc="upper right", ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "5_per_question_changes.png"), dpi=170)
    plt.close(fig)


def fig_position(data, numbers):
    """图 6：方法检查——位置偏好。谄媚回答放在 A 还是 B，会不会改变选择。"""
    models = list(data)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    off = np.linspace(-0.14, 0.14, len(models))
    hs = []
    for mi, m in enumerate(models):
        rows = [r for r in data[m] if r["picked_sycophantic"] is not None and r["context"] == "wrong_suggestion"]
        numbers[m]["pick_A_rate"] = round(float(np.mean([r["choice"] == "A" for r in data[m] if r["choice"]])), 4)
        for ci, c in enumerate(ORDER):
            a = np.mean([r["picked_sycophantic"] for r in rows if r["condition"] == c and r["sycophantic_first"]]) * 100
            b = np.mean([r["picked_sycophantic"] for r in rows if r["condition"] == c and not r["sycophantic_first"]]) * 100
            y = ci + off[mi]
            ax.plot([b, a], [y, y], color=SERIES[mi], lw=2, alpha=0.5)
            ax.scatter(a, y, s=56, color=SERIES[mi], marker="o", edgecolor=SURFACE, lw=1.5, zorder=3)
            h = ax.scatter(b, y, s=56, facecolor=SURFACE, edgecolor=SERIES[mi], marker="o", lw=2, zorder=3)
        hs.append(plt.Line2D([], [], color=SERIES[mi], lw=2))
    ax.set_yticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("用户带错时选谄媚回答的比例（%）：实心 = 谄媚回答放在 A，空心 = 放在 B")
    style(ax)
    legend_top(ax, hs, models, len(models))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "6_position_bias_check.png"), dpi=170)
    plt.close(fig)


def fig_consistency(data, numbers):
    """图 7：把每道题的两次判断（交换顺序）分成三类——两次都纠正、两次都谄媚、随位置摇摆。"""
    models = list(data)
    rows_order = [("neutral", "quality", "中立提问（基线）")] + [("wrong_suggestion", c, LABELS[c]) for c in ORDER]
    fig, axes = plt.subplots(1, len(models), figsize=(10.5, 4.4), sharey=True)
    for mi, (m, ax) in enumerate(zip(models, np.atleast_1d(axes))):
        d = defaultdict(dict)
        for r in data[m]:
            if r["picked_sycophantic"] is not None:
                d[(r["qid"], r["condition"], r["context"])][r["sycophantic_first"]] = r["picked_sycophantic"]
        for yi, (ctx, c, lab) in enumerate(rows_order):
            v = [x for k, x in d.items() if k[1] == c and k[2] == ctx and len(x) == 2]
            syc = sum(x[True] and x[False] for x in v) / len(v)
            tru = sum((not x[True]) and (not x[False]) for x in v) / len(v)
            split = 1 - syc - tru
            numbers[m].setdefault("consistency", {})[f"{ctx}:{c}"] = {
                "consistent_truthful": round(tru, 4), "order_dependent": round(split, 4), "consistent_sycophantic": round(syc, 4)}
            left = 0
            for val, col in [(tru, LESS), (split, SAME), (syc, MORE)]:
                ax.barh(yi, val * 100, left=left, color=col, height=0.62, edgecolor=SURFACE, lw=2)
                if val >= 0.045:
                    ax.text(left + val * 50, yi, f"{val * 100:.0f}", ha="center", va="center", fontsize=8.5,
                            color=SURFACE if col != SAME else INK)
                left += val * 100
        ax.set_title(m, loc="left", fontsize=11)
        ax.set_yticks(range(len(rows_order)), [r[2] for r in rows_order])
        if mi == 0:
            ax.invert_yaxis()
        ax.axhline(0.5, color=INK2, lw=0.8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("题目占比（%）")
        style(ax, xgrid=False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LESS, SAME, MORE)]
    fig.legend(hs, ["两种顺序都选纠正", "随回答顺序改变（位置偏好）", "两种顺序都选谄媚"], loc="upper right", ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "7_judgment_consistency.png"), dpi=170)
    plt.close(fig)


def main():
    setup()
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    S = {m: scores(rows) for m, rows in data.items()}
    numbers = {m: {"index": {}, "valid": sum(r["choice"] is not None for r in rows), "total": len(rows)} for m, rows in data.items()}
    fig_index(data, S, numbers)
    fig_decomposition(data, S)
    fig_contrasts(data, S, numbers)
    fig_known(data, S, numbers)
    fig_flips(data, S, numbers)
    fig_position(data, numbers)
    fig_consistency(data, numbers)
    with open(os.path.join(RESULTS, "ANALYSIS_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(numbers, f, ensure_ascii=False, indent=2)
    print(json.dumps(numbers, ensure_ascii=False, indent=1))
    print(f"图已写入 {OUT}")


if __name__ == "__main__":
    main()
