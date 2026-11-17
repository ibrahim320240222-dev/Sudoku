import os, sys, itertools, math
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sudoku.csp import solve, VARIANTS, is_valid_solution, conflicts
from sudoku.generator import make_puzzle
from sudoku.search import correction_search, build_candidates
from sudoku.integration import GoalChecker

# ---------- NORMAL (5)
@pytest.mark.parametrize("seed,givens", [(1, 35), (2, 30), (3, 25), (4, 32), (5, 28)])
def test_normal_all_variants_solve_and_agree(seed, givens):
    puz, sol = make_puzzle(seed, givens)
    for name, cfg in VARIANTS.items():
        r = solve(puz, max_assign=500000, **cfg)
        assert r.solved, name
        assert is_valid_solution(r.solutions[0], puz)
        assert r.solutions[0] == sol          # unique puzzle -> same answer for every variant

# ---------- EDGE (3)
def test_edge_already_complete_grid():
    _, sol = make_puzzle(7, 30)
    r = solve(sol, fc=True, ac3_pre=True)
    assert r.solved and r.solutions[0] == sol and r.stats.assignments == 0

def test_edge_empty_grid_is_solvable():
    r = solve([0] * 81, fc=True, ac3_pre=True)
    assert r.solved and is_valid_solution(r.solutions[0])

def test_edge_search_zero_corrections_needed():
    puz, _ = make_puzzle(8, 30)
    probs = np.zeros((81, 9), np.float32)
    mask = [d != 0 for d in puz]
    for i, d in enumerate(puz):
        if d:
            probs[i] = 0.01; probs[i][d - 1] = 0.92
    cand = build_candidates(probs, mask)
    sr = correction_search(puz, cand, GoalChecker(), "ucs")
    assert sr.success and sr.corrections == [] and sr.expanded == 1

# ---------- IMPOSSIBLE (2) -> clear "no solution" message
def test_impossible_equal_givens_in_row():
    g = [0] * 81; g[0] = 5; g[1] = 5
    for name, cfg in VARIANTS.items():
        r = solve(g, **cfg)
        assert r.status == "no_solution" and "no solution" in r.message

def test_impossible_consistent_givens_but_unsolvable():
    g = [0] * 81
    for c, d in enumerate(range(1, 9)):
        g[c] = d            # row 1 has 1..8 -> cell (0,8) must be 9
    g[17] = 9               # 9 already in column 9 (row 2) -> cell (0,8) has no value
    assert not conflicts(g)
    for name, cfg in VARIANTS.items():
        r = solve(g, **cfg)
        assert r.status == "no_solution" and "no solution" in r.message

# ---------- search: small example with a KNOWN answer + brute force check
def _toy():
    puz, sol = make_puzzle(11, 30)
    cells = [i for i in range(81) if puz[i]]
    a, b = cells[0], cells[1]
    wrong = list(puz)
    # make the reading wrong at cell a: another digit that conflicts with a peer given
    from sudoku.csp import PEERS
    peer = next(p for p in PEERS[a] if puz[p])
    wrong[a] = puz[peer]
    probs = np.zeros((81, 9), np.float32)
    for i in cells:
        probs[i] = 0.002; probs[i][wrong[i] - 1] = 0.984
    probs[a] = 0.01; probs[a][wrong[a] - 1] = 0.52; probs[a][puz[a] - 1] = 0.43
    return puz, wrong, probs, a

@pytest.mark.parametrize("strategy", ["ucs", "astar", "greedy"])
def test_search_known_answer(strategy):
    puz, wrong, probs, a = _toy()
    cand = build_candidates(probs, [d != 0 for d in wrong])
    sr = correction_search(wrong, cand, GoalChecker(goal="unique"), strategy)
    assert sr.success and sr.solution is not None
    assert [(c, new) for c, _, new, _ in sr.corrections] == [(a, puz[a])]
    assert sr.cost == pytest.approx(math.log(0.52 / 0.43), rel=1e-3)

def test_ucs_and_astar_match_brute_force():
    puz, wrong, probs, a = _toy()
    cand = build_candidates(probs, [d != 0 for d in wrong], K=6)
    gc = GoalChecker()
    best = math.inf
    items = [(c, d, k) for c, o in cand.items() for d, k in o]
    for n in range(0, 4):
        for combo in itertools.combinations(items, n):
            if len({c for c, _, _ in combo}) < n:
                continue
            g = list(wrong)
            for c, d, _ in combo:
                g[c] = d
            if gc(g)[0]:
                best = min(best, sum(k for _, _, k in combo))
    for s in ("ucs", "astar"):
        sr = correction_search(wrong, cand, GoalChecker(), s)
        assert sr.cost == pytest.approx(best, abs=1e-9)

def test_counters_exist_and_fc_reduces_work():
    puz, _ = make_puzzle(3, 30)
    bt = solve(puz, fc=False, mrv=False)
    fc = solve(puz, fc=True, mrv=False)
    assert fc.stats.assignments <= bt.stats.assignments
    assert bt.stats.backtracks <= bt.stats.assignments

def test_search_unreadable_when_too_many_misreads():
    g = [0] * 81; g[0] = 5; g[1] = 5; g[2] = 5
    probs = np.zeros((81, 9), np.float32)
    for i in range(3):
        probs[i] = 0.01; probs[i][4] = 0.92
    cand = build_candidates(probs, [d != 0 for d in g])
    sr = correction_search(g, cand, GoalChecker(), "ucs", max_corr=1)
    assert not sr.success and "no correction set" in sr.message
