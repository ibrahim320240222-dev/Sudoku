"""search.py - search over CORRECTION SETS of the NN's first reading.

State  : frozenset of (cell, new_digit) corrections (at most `max_corr`, one per cell).
Action : change ONE not-yet-corrected uncertain cell to one of its top alternative digits.
Cost   : log p(read digit) - log p(new digit)  >= 0  (read digit is the arg-max, so it is never negative).
Goal   : goal_fn(grid) is True (puzzle solvable, or uniquely solvable).
Strategies:
  ucs    - priority g                      -> cheapest (= most likely) correction set, optimal
  greedy - priority = number of conflicts  -> fast, not optimal
  astar  - priority g + h                  -> optimal, h admissible (see `h_astar`)
  random - random priority (baseline used in experiment 3)
Counters: expanded, generated, goal_tests.
"""
import heapq, math, itertools
from dataclasses import dataclass, field
import numpy as np
from .csp import conflicts

EPS = 1e-12
INF = float("inf")


def build_candidates(probs, given_mask, K=12, n_alt=2):
    """probs: (81,9) array (rows of blank cells ignored). Returns dict cell -> [(digit, cost), ...]."""
    cells = [i for i in range(81) if given_mask[i]]
    conf = {i: probs[i].max() for i in cells}
    chosen = sorted(cells, key=lambda i: conf[i])[:K]
    cand = {}
    for i in chosen:
        p = probs[i]
        read = int(p.argmax())
        order = [k for k in np.argsort(-p) if k != read][:n_alt]
        cand[i] = [(int(k) + 1, max(0.0, math.log(p[read] + EPS) - math.log(p[k] + EPS))) for k in order]
    return cand


def build_all_candidates(probs, given_mask):
    """Every given cell, every other digit (used by the random baseline)."""
    cand = {}
    for i in range(81):
        if given_mask[i]:
            p = probs[i]
            read = int(p.argmax())
            cand[i] = [(k + 1, max(0.0, math.log(p[read] + EPS) - math.log(p[k] + EPS)))
                       for k in range(9) if k != read]
    return cand


def apply(read_grid, key):
    g = list(read_grid)
    for c, d in key:
        g[c] = d
    return g


def h_astar(grid, key, cand, max_corr):
    """Admissible heuristic: every duplicate pair needs >=1 of its two cells changed.
    Take disjoint pairs; each costs at least the cheapest alternative of its cheaper cell."""
    pairs = conflicts(grid)
    if not pairs:
        return 0.0
    done = {c for c, _ in key}

    def mc(cell):
        if cell in done or cell not in cand:
            return INF
        return min(cost for _, cost in cand[cell])

    scored = sorted(((min(mc(a), mc(b)), a, b) for a, b in pairs), key=lambda t: -t[0] if t[0] != INF else 1e18)
    used, h, n = set(), 0.0, 0
    for m, a, b in scored:
        if a in used or b in used:
            continue
        if m == INF:
            return INF            # this conflict can never be repaired inside the action space
        used.update((a, b))
        h += m
        n += 1
    if n > max_corr - len(key):
        return INF                # more disjoint conflicts than corrections left
    return h


@dataclass
class SearchResult:
    success: bool
    corrections: list = field(default_factory=list)   # (cell, read_digit, new_digit, cost)
    cost: float = 0.0
    grid: list = None
    solution: list = None
    expanded: int = 0
    generated: int = 0
    goal_tests: int = 0
    trace: list = field(default_factory=list)          # [(expanded_no, key, g)]
    message: str = ""


def correction_search(read_grid, cand, goal_fn, strategy="ucs", max_corr=3, max_nodes=None,
                      rng=None, trace=False):
    """Best-first search over correction sets. goal_fn(grid) -> (bool, solution_or_None)."""
    counter = itertools.count()
    root = frozenset()
    res = SearchResult(False)
    best_g = {root: 0.0}
    rng = rng or np.random.default_rng(0)

    def prio(key, g, grid):
        if strategy == "ucs":
            return (g, 0)
        if strategy == "greedy":
            return (len(conflicts(grid)), g)
        if strategy == "astar":
            h = h_astar(grid, key, cand, max_corr)
            return (g + h, g) if h != INF else None
        if strategy == "random":
            return (rng.random(), 0)
        raise ValueError(strategy)

    g0 = apply(read_grid, root)
    p0 = prio(root, 0.0, g0)
    heap = []
    if p0 is not None:
        heap.append((p0, next(counter), 0.0, root))
    res.generated = 1
    while heap:
        _, _, g, key = heapq.heappop(heap)
        if g > best_g.get(key, INF) + 1e-12:
            continue
        res.expanded += 1
        grid = apply(read_grid, key)
        if trace:
            res.trace.append((res.expanded, tuple(sorted(key)), g))
        res.goal_tests += 1
        ok, sol = goal_fn(grid)
        if ok:
            res.success, res.cost, res.grid, res.solution = True, g, grid, sol
            res.corrections = [(c, read_grid[c], d, _cost(cand, c, d)) for c, d in sorted(key)]
            return res
        if max_nodes is not None and res.expanded >= max_nodes:
            res.message = "node budget exhausted"
            return res
        if len(key) >= max_corr:
            continue
        done = {c for c, _ in key}
        for c, opts in cand.items():
            if c in done:
                continue
            for d, cost in opts:
                nk = key | {(c, d)}
                ng = g + cost
                if ng < best_g.get(nk, INF) - 1e-12:
                    best_g[nk] = ng
                    pr = prio(nk, ng, apply(read_grid, nk))
                    res.generated += 1
                    if pr is not None:
                        heapq.heappush(heap, (pr, next(counter), ng, nk))
    res.message = "search space exhausted: no correction set (<= %d corrections) makes the puzzle solvable" % max_corr
    return res


def _cost(cand, c, d):
    for dd, cost in cand[c]:
        if dd == d:
            return cost
    return 0.0
