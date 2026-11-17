"""run_demo.py - one integrated demo: NN reads -> AC-3 conflict -> Search corrects -> CSP solves.
Usage: python run_demo.py [--seed 3] [--misreads 2] [--strategy ucs|astar|greedy]
Output: console log + results/demo.png"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sudoku.nn import MLP
from sudoku.csp import solve, VARIANTS, conflicts, ac3, initial_domains, Stats
from sudoku.generator import make_puzzle, build_mosaic
from sudoku.integration import solve_mosaic
from eval.experiments import pools

ROOT = os.path.dirname(os.path.abspath(__file__))


def draw_grid(ax, grid, title, hl=None, bad=None, ref=None):
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)
    for i in range(10):
        ax.plot([0, 9], [i, i], "k", lw=1.6 if i % 3 == 0 else 0.4)
        ax.plot([i, i], [0, 9], "k", lw=1.6 if i % 3 == 0 else 0.4)
    for i, d in enumerate(grid):
        r, c = divmod(i, 9)
        if hl and i in hl:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#ffd54f"))
        if bad and i in bad:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#ef9a9a"))
        if d:
            col = "tab:blue" if (ref is not None and ref[i] == 0) else "black"
            ax.text(c + .5, r + .55, d, ha="center", va="center", fontsize=11, color=col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--strategy", default="ucs")
    a = ap.parse_args()
    net = MLP.load(os.path.join(ROOT, "data", "mlp.npz"))
    pool = pools(net)
    puz, sol = make_puzzle(a.seed, 30)
    cells = [i for i in range(81) if puz[i]]
    inj = [int(c) for c in np.random.default_rng(a.seed).choice(cells, a.misreads, replace=False)]
    mosaic = build_mosaic(puz, sol, pool, np.random.default_rng(a.seed), inject=inj)

    print("=== INPUT: 252x252 handwritten mosaic, %d givens, %d injected hard digits at %s" %
          (len(cells), a.misreads, [(c // 9 + 1, c % 9 + 1) for c in inj]))
    out = solve_mosaic(mosaic, net, strategy=a.strategy, trace=True)
    rg = out["read_grid"]
    print("=== 1) NN reading: misread cells:", [(c // 9 + 1, c % 9 + 1, puz[c], rg[c]) for c in cells if rg[c] != puz[c]])
    print("=== 2) AC-3 on first reading: consistent =", out["ac3_ok"], "| duplicate pairs:",
          [((a_ // 9 + 1, a_ % 9 + 1), (b // 9 + 1, b % 9 + 1)) for a_, b in out["conflicts"]])
    sr = out["search"]
    print("=== 3) %s search trace (expanded, corrections, cost g):" % a.strategy.upper())
    for n, key, g in sr.trace[:15]:
        print("   #%-3d %-45s g=%.3f" % (n, [((c // 9 + 1, c % 9 + 1), d) for c, d in key], g))
    if len(sr.trace) > 15:
        print("   ... %d more" % (len(sr.trace) - 15))
    print("   expanded=%d generated=%d CSP calls=%d" % (sr.expanded, sr.generated, out["csp_calls"]))
    print("=== 4) RESULT:", out["message"])
    if sr.success:
        print("   correction log (row, col, read -> new, cost):")
        for c, old, new, cost in sr.corrections:
            print("     r%dc%d  %d -> %d  cost %.3f" % (c // 9 + 1, c % 9 + 1, old, new, cost))
        print("   exact solution recovered:", out["solution"] == sol)

    print("\n=== NN/search ON vs OFF (same puzzle) ===")
    off = solve(rg, fc=True, ac3_pre=True)
    print("   OFF (solve the first reading directly):", off.status, "-", off.message or "ok")
    print("   ON  (with correction search)          :", out["status"], "| exact =", out.get("solution") == sol)

    print("\n=== CSP pruning on the corrected puzzle: BT vs BT+FC vs BT+FC+AC3 (row-major order) ===")
    gfin = sr.grid if sr.success else rg
    print("   %-10s %12s %11s %10s %8s" % ("variant", "assignments", "backtracks", "revisions", "prunes"))
    rows = {}
    for name, cfg in VARIANTS.items():
        r = solve(gfin, mrv=False, max_assign=300000, **cfg)
        rows[name] = r.stats
        print("   %-10s %12d %11d %10d %8d" % (name, r.stats.assignments, r.stats.backtracks, r.stats.revisions, r.stats.prunes))

    # ---------- figure
    probs = out["probs"]
    lowc = sorted(cells, key=lambda i: probs[i].max())[:6]
    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_subplot(2, 4, 1); ax.imshow(mosaic, cmap="gray"); ax.axis("off"); ax.set_title("1. Input mosaic (handwritten)", fontsize=9)
    draw_grid(fig.add_subplot(2, 4, 2), rg, "2. NN reading (red = duplicate, yellow = least confident)",
              hl=set(lowc), bad={c for p in out["conflicts"] for c in p})
    ax = fig.add_subplot(2, 4, 3)
    labels = ["r%dc%d" % (i // 9 + 1, i % 9 + 1) for i in lowc]
    ax.barh(labels[::-1], [probs[i].max() for i in lowc][::-1], color="tab:orange")
    ax.set_xlim(0, 1); ax.set_title("NN confidence of 6 least sure cells", fontsize=9)
    ax = fig.add_subplot(2, 4, 4)
    if sr.trace:
        xs = [t[0] for t in sr.trace]; ys = [t[2] for t in sr.trace]
        ax.plot(xs, ys, "o-", color="tab:purple"); ax.set_xlabel("node expanded"); ax.set_ylabel("cost g")
    ax.set_title("3. Search trace (%s)" % a.strategy.upper(), fontsize=9)
    corr = {c for c, _, _, _ in sr.corrections}
    draw_grid(fig.add_subplot(2, 4, 5), gfin, "4. After correction (yellow = corrected)", hl=corr)
    draw_grid(fig.add_subplot(2, 4, 6), out.get("solution") or gfin, "5. CSP solution (blue = filled by CSP)", ref=gfin)
    ax = fig.add_subplot(2, 4, 7)
    names = list(rows); ax.bar(names, [rows[n].assignments for n in names], color=["#90a4ae", "#4db6ac", "#1976d2"])
    ax.set_yscale("log"); ax.set_title("CSP assignments (row-major order)", fontsize=9)
    ax = fig.add_subplot(2, 4, 8)
    ok_on = out.get("solution") == sol
    ax.bar(["NN+search OFF", "NN+search ON"], [1 if (off.solved and off.solutions[0] == sol) else 0, 1 if ok_on else 0],
           color=["#e57373", "#81c784"]); ax.set_ylim(0, 1.1); ax.set_title("Exact solution recovered?", fontsize=9)
    plt.tight_layout()
    p = os.path.join(ROOT, "results", "demo.png")
    plt.savefig(p, dpi=140)
    print("\nfigure saved:", p)


if __name__ == "__main__":
    main()
