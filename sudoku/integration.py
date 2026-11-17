"""integration.py - the full pipeline: NN reads -> AC-3 finds conflict -> Search corrects -> CSP solves."""
import numpy as np
from .csp import solve, conflicts, ac3, initial_domains, Stats
from .search import build_candidates, correction_search
from .generator import split_mosaic

BLANK_THRESHOLD = 0.02


class GoalChecker:
    """Goal test = ask the CSP. Caches answers and sums the CSP counters."""

    def __init__(self, goal="solvable", max_assign=5000, fc=True, ac3_pre=True):
        self.goal, self.max_assign, self.fc, self.ac3_pre = goal, max_assign, fc, ac3_pre
        self.cache, self.stats, self.csp_calls = {}, Stats(), 0

    def __call__(self, grid):
        k = tuple(grid)
        if k in self.cache:
            return self.cache[k]
        if conflicts(grid):                       # cheap exit: equal givens in one unit
            out = (False, None)
        else:
            self.csp_calls += 1
            n = 1 if self.goal == "solvable" else 2
            r = solve(grid, fc=self.fc, ac3_pre=self.ac3_pre, max_assign=self.max_assign, max_solutions=n)
            self.stats.add(r.stats)
            if r.status != "solved":
                out = (False, None)
            elif self.goal == "unique" and len(r.solutions) != 1:
                out = (False, None)
            else:
                out = (True, r.solutions[0])
        self.cache[k] = out
        return out


def read_mosaic(mosaic, net):
    """NN reading step. Blank cells are found by brightness (not by the NN)."""
    tiles = split_mosaic(mosaic)
    blank = tiles.mean(1) < BLANK_THRESHOLD
    probs = np.zeros((81, 9), np.float32)
    if (~blank).any():
        probs[~blank] = net.predict_proba(tiles[~blank])
    grid = [0 if blank[i] else int(probs[i].argmax()) + 1 for i in range(81)]
    return grid, probs, ~blank


def solve_mosaic(mosaic, net, strategy="ucs", goal="solvable", max_corr=3, K=12, n_alt=2,
                 max_nodes=None, trace=False):
    """Complete system. Returns a dict with everything needed for logs/plots."""
    grid, probs, given_mask = read_mosaic(mosaic, net)
    out = dict(read_grid=grid, probs=probs, given_mask=given_mask, conflicts=conflicts(grid))
    dom = initial_domains(grid)
    st = Stats()
    out["ac3_ok"] = ac3(dom, st)
    out["ac3_stats"] = st
    cand = build_candidates(probs, given_mask, K=K, n_alt=n_alt)
    gc = GoalChecker(goal=goal)
    sr = correction_search(grid, cand, gc, strategy=strategy, max_corr=max_corr, max_nodes=max_nodes, trace=trace)
    out.update(search=sr, candidates=cand, csp_stats=gc.stats, csp_calls=gc.csp_calls)
    if sr.success:
        out["status"] = "solved"
        out["solution"] = sr.solution
        out["message"] = "solved with %d correction(s), total cost %.3f" % (len(sr.corrections), sr.cost)
    else:
        conf = [(probs[i].max(), i) for i in range(81) if given_mask[i]]
        least = sorted(conf)[:5]
        out["status"] = "unreadable"
        out["least_confident"] = [(i, int(probs[i].argmax()) + 1, float(c)) for c, i in least]
        out["message"] = ("unreadable: " + sr.message + ". Least confident cells (cell, read, confidence): "
                          + ", ".join("r%dc%d=%d (%.2f)" % (i // 9 + 1, i % 9 + 1, d, c) for i, d, c in out["least_confident"]))
    return out
