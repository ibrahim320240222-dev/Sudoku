"""generator.py - seeded Sudoku puzzle generator and mosaic builder."""
import numpy as np
from .csp import PEERS, FULL, solve


def full_grid(rng):
    """Complete valid grid by randomised backtracking (MRV)."""
    g = [0] * 81
    dom = [FULL] * 81

    def rec(g, dom):
        free = [i for i in range(81) if g[i] == 0]
        if not free:
            return True
        i = min(free, key=lambda x: (bin(dom[x]).count("1"), x))
        digits = [d for d in range(1, 10) if dom[i] & (1 << (d - 1))]
        rng.shuffle(digits)
        for d in digits:
            b = 1 << (d - 1)
            nd = list(dom)
            ok = True
            for p in PEERS[i]:
                if g[p] == 0 and nd[p] & b:
                    nd[p] &= ~b
                    if nd[p] == 0:
                        ok = False
                        break
            if not ok:
                continue
            g[i] = d
            if rec(g, nd):
                return True
            g[i] = 0
        return False

    rec(g, dom)
    return g


def count_solutions(grid, limit=2):
    r = solve(grid, fc=True, ac3_pre=True, max_solutions=limit)
    return len(r.solutions)


def make_puzzle(seed, givens):
    """Return (puzzle, solution) with a UNIQUE solution and exactly `givens` given cells."""
    rng = np.random.default_rng(seed)
    py = np.random.RandomState(seed)
    for _ in range(50):
        sol = full_grid(py)
        puz = list(sol)
        order = list(rng.permutation(81))
        n = 81
        for c in order:
            if n <= givens:
                break
            keep = puz[c]
            puz[c] = 0
            if count_solutions(puz, 2) == 1:
                n -= 1
            else:
                puz[c] = keep
        if n == givens:
            return puz, sol
    raise RuntimeError("could not reach the requested number of givens")


def build_mosaic(puzzle, solution, x_pool, rng_np, inject=None):
    """Make a 252x252 image from a puzzle.

    x_pool: dict with 'clean' and 'hard' -> {digit: array of images (n,784)}
    inject: list of cell indices that get a hard (misread-prone) digit image.
    Returns mosaic (252,252) float32, and the list of cell indices that are really misread-prone.
    """
    inject = set(inject or [])
    tiles = np.zeros((81, 784), np.float32)
    for i, d in enumerate(puzzle):
        if d == 0:
            tiles[i] = np.clip(rng_np.normal(0, 0.01, 784), 0, 1)
        else:
            pool = x_pool["hard"][d] if i in inject else x_pool["clean"][d]
            tiles[i] = pool[rng_np.integers(len(pool))]
    m = tiles.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)
    return m


def split_mosaic(m):
    return m.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3).reshape(81, 784)
