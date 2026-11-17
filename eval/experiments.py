"""Run the three experiments (5 seeds each) and save json/csv + plots in results/."""
import csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sudoku.nn import MLP
from sudoku.csp import solve, VARIANTS
from sudoku.generator import make_puzzle, build_mosaic
from sudoku.integration import read_mosaic, GoalChecker
from sudoku.search import build_candidates, build_all_candidates, correction_search

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEEDS = [0, 1, 2, 3, 4]
PER_SEED = 20           # 5 seeds x 20 puzzles = 100 puzzles per level
GIVENS = 30


def pools(net):
    d = np.load(os.path.join(ROOT, "data", "mnist19.npz"))
    xt, yt = d["x_test"], d["y_test"]
    p = net.predict_proba(xt)
    pred = p.argmax(1) + 1
    second = np.argsort(-p, 1)[:, 1] + 1
    return {"clean": {k: xt[(yt == k) & (pred == k)] for k in range(1, 10)},
            "hard": {k: xt[(yt == k) & (pred != k) & (second == k)] for k in range(1, 10)}}


def ms(vals):
    a = np.array(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def instance(net, pool, seed, j, level, givens=GIVENS):
    puz, sol = make_puzzle(1000 * seed + j, givens)
    cells = [i for i in range(81) if puz[i]]
    r = np.random.default_rng([seed, j, level])
    inj = list(r.choice(cells, level, replace=False)) if level else []
    mo = build_mosaic(puz, sol, pool, np.random.default_rng([seed, j, 99]), inject=inj)
    return puz, sol, mo, inj


def run_search(net, mo, sol, strategy, goal="solvable", all_cand=False, max_nodes=None, rng=None):
    grid, probs, gm = read_mosaic(mo, net)
    cand = build_all_candidates(probs, gm) if all_cand else build_candidates(probs, gm, K=12, n_alt=2)
    gc = GoalChecker(goal=goal)
    t = time.time()
    sr = correction_search(grid, cand, gc, strategy=strategy, max_corr=3, max_nodes=max_nodes, rng=rng)
    return dict(solved=sr.success, exact=bool(sr.success and sr.solution == sol), expanded=sr.expanded,
                generated=sr.generated, cost=sr.cost if sr.success else float("nan"),
                time=time.time() - t, corrections=sr.corrections, read=grid)


def exp1(net, pool):
    algos = [("UCS", "ucs", "solvable"), ("Greedy", "greedy", "solvable"), ("A*", "astar", "solvable"),
             ("UCS (unique goal)", "ucs", "unique")]
    out, log_rows = {}, []
    for level in range(5):
        for name, strat, goal in algos:
            per_seed = []
            for s in SEEDS:
                rows = []
                for j in range(PER_SEED):
                    puz, sol, mo, inj = instance(net, pool, s, j, level)
                    r = run_search(net, mo, sol, strat, goal)
                    rows.append(r)
                    if name == "UCS":
                        for c, old, new, cost in r["corrections"]:
                            log_rows.append([level, s, j, c // 9 + 1, c % 9 + 1, old, new, round(cost, 4),
                                             "yes" if c in inj else "no"])
                per_seed.append(dict(
                    expanded=np.mean([r["expanded"] for r in rows]),
                    generated=np.mean([r["generated"] for r in rows]),
                    cost=np.nanmean([r["cost"] for r in rows]) if any(r["solved"] for r in rows) else float("nan"),
                    solved=np.mean([r["solved"] for r in rows]), exact=np.mean([r["exact"] for r in rows]),
                    time=np.mean([r["time"] for r in rows])))
            out[f"{level}|{name}"] = {k: ms([p[k] for p in per_seed]) for k in per_seed[0]}
            print("exp1", level, name, {k: round(v[0], 3) for k, v in out[f"{level}|{name}"].items()}, flush=True)
    with open(os.path.join(RES, "correction_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["misreads_injected", "seed", "puzzle", "row", "col", "read_digit", "new_digit", "cost", "was_injected"])
        w.writerows(log_rows)
    return out


def exp2(cap=200000):
    cfgs = [("BT", dict(fc=False, ac3_pre=False)), ("BT+FC", dict(fc=True, ac3_pre=False)),
            ("BT+FC+AC3", dict(fc=True, ac3_pre=True))]
    out = {}
    for order in ("static", "mrv"):
        for g in (25, 30, 35):
            puzzles = [[make_puzzle(1000 * s + j + 500, g)[0] for j in range(PER_SEED)] for s in SEEDS]
            for name, cfg in cfgs:
                per_seed = []
                for pz in puzzles:
                    rr = []
                    for p in pz:
                        mrv = True if order == "mrv" else False
                        t = time.time()
                        r = solve(p, mrv=mrv, max_assign=cap, **cfg)
                        rr.append((r.stats.assignments, r.stats.backtracks, r.stats.revisions, time.time() - t, r.status == "solved"))
                    a = np.array(rr, float)
                    per_seed.append(dict(assignments=a[:, 0].mean(), backtracks=a[:, 1].mean(), revisions=a[:, 2].mean(),
                                         time=a[:, 3].mean(), solved=a[:, 4].mean()))
                out[f"{order}|{g}|{name}"] = {k: ms([p[k] for p in per_seed]) for k in per_seed[0]}
                print("exp2", order, g, name, {k: round(v[0], 2) for k, v in out[f"{order}|{g}|{name}"].items()}, flush=True)
    return out


def exp3(net, pool):
    out = {}
    for level in (1, 2, 3):
        for name in ("No correction", "Random correction", "NN-guided (UCS)", "NN-guided (UCS, unique goal)"):
            per_seed = []
            for s in SEEDS:
                ex, so = [], []
                for j in range(PER_SEED):
                    puz, sol, mo, inj = instance(net, pool, s, j, level)
                    if name == "No correction":
                        grid, _, _ = read_mosaic(mo, net)
                        ok, sl = GoalChecker()(grid)
                        ex.append(bool(ok and sl == sol)); so.append(bool(ok))
                    elif name == "Random correction":
                        r = run_search(net, mo, sol, "random", all_cand=True, max_nodes=300, rng=np.random.default_rng([s, j, level]))
                        ex.append(r["exact"]); so.append(r["solved"])
                    elif name == "NN-guided (UCS)":
                        r = run_search(net, mo, sol, "ucs")
                        ex.append(r["exact"]); so.append(r["solved"])
                    else:
                        r = run_search(net, mo, sol, "ucs", goal="unique")
                        ex.append(r["exact"]); so.append(r["solved"])
                per_seed.append(dict(exact=np.mean(ex), solvable=np.mean(so)))
            out[f"{level}|{name}"] = {k: ms([p[k] for p in per_seed]) for k in per_seed[0]}
            print("exp3", level, name, {k: round(v[0], 3) for k, v in out[f"{level}|{name}"].items()}, flush=True)
    return out


def plots(e1, e2, e3):
    algos = ["UCS", "Greedy", "A*", "UCS (unique goal)"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a in algos:
        ax[0].errorbar(range(5), [e1[f"{l}|{a}"]["expanded"][0] for l in range(5)],
                       [e1[f"{l}|{a}"]["expanded"][1] for l in range(5)], marker="o", label=a, capsize=2)
        ax[1].errorbar(range(5), [e1[f"{l}|{a}"]["exact"][0] for l in range(5)],
                       [e1[f"{l}|{a}"]["exact"][1] for l in range(5)], marker="o", label=a, capsize=2)
    ax[0].set_yscale("log"); ax[0].set_title("Nodes expanded (mean +- sd, 5 seeds)")
    ax[1].set_title("Exact-solution recovery rate")
    for a in ax[:2]:
        a.set_xlabel("injected misreads"); a.set_xticks(range(5))
    ax[1].legend(fontsize=7)
    for a in algos[:3]:
        ax[2].errorbar(range(5), [e1[f"{l}|{a}"]["cost"][0] for l in range(5)],
                       [e1[f"{l}|{a}"]["cost"][1] for l in range(5)], marker="o", label=a, capsize=2)
    ax[2].set_title("Total correction cost (solved cases)"); ax[2].set_xlabel("injected misreads"); ax[2].set_xticks(range(5))
    plt.tight_layout(); plt.savefig(os.path.join(RES, "exp1.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["BT", "BT+FC", "BT+FC+AC3"]
    for k, order in enumerate(("static", "mrv")):
        w = 0.25
        for i, n in enumerate(names):
            m = [e2[f"{order}|{g}|{n}"]["assignments"][0] for g in (25, 30, 35)]
            sd = [e2[f"{order}|{g}|{n}"]["assignments"][1] for g in (25, 30, 35)]
            ax[k].bar(np.arange(3) + i * w, m, w, yerr=sd, label=n, capsize=2)
        ax[k].set_xticks(np.arange(3) + w); ax[k].set_xticklabels(["25 givens", "30 givens", "35 givens"])
        ax[k].set_yscale("log"); ax[k].set_ylabel("assignments (log)")
        ax[k].set_title("Variable order: " + ("row-major (static)" if order == "static" else "MRV for all three"))
    ax[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "exp2.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = ["No correction", "Random correction", "NN-guided (UCS)", "NN-guided (UCS, unique goal)"]
    w = 0.2
    for i, n in enumerate(names):
        ax.bar(np.arange(3) + i * w, [e3[f"{l}|{n}"]["exact"][0] for l in (1, 2, 3)], w,
               yerr=[e3[f"{l}|{n}"]["exact"][1] for l in (1, 2, 3)], label=n, capsize=2)
    ax.set_xticks(np.arange(3) + 1.5 * w); ax.set_xticklabels(["1 misread", "2 misreads", "3 misreads"])
    ax.set_ylabel("exact-solution recovery"); ax.set_title("NN-guided vs no vs random correction"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "exp3.png"), dpi=150); plt.close()


if __name__ == "__main__":
    net = MLP.load(os.path.join(ROOT, "data", "mlp.npz"))
    pool = pools(net)
    t0 = time.time()
    e1 = exp1(net, pool)
    e2 = exp2()
    e3 = exp3(net, pool)
    json.dump(dict(exp1=e1, exp2=e2, exp3=e3, pool_sizes={k: len(v) for k, v in pool["hard"].items()}),
              open(os.path.join(RES, "experiments.json"), "w"), indent=1)
    plots(e1, e2, e3)
    print("done in %.0fs" % (time.time() - t0))
