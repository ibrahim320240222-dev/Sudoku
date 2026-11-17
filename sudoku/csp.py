"""csp.py - Sudoku as a CSP with three SWITCHABLE parts.

Variables: 81 cells. Domains: digits 1..9 stored as bit masks (bit d-1). Givens: singleton domains.
Constraint: cells in the same row / column / 3x3 box differ (binary != constraints).

  * Backtracking (BT)         : assign a digit, check it against already assigned peers.
  * Forward Checking (FC)     : after each assignment remove the digit from unassigned peers; stop if a domain is empty.
  * Arc consistency (AC-3)    : run once BEFORE the search to shrink domains / detect conflicts (e.g. two equal
                                givens in one row empty a domain).

Counters (class Stats): assignments, backtracks (= assignments that were undone), ac3 revisions, ac3 prunes.
"""
from collections import deque
from dataclasses import dataclass

FULL = 0x1FF


def _build():
    peers = [set() for _ in range(81)]
    units = []
    for r in range(9):
        units.append([r * 9 + c for c in range(9)])
    for c in range(9):
        units.append([r * 9 + c for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(br * 3 + i) * 9 + bc * 3 + j for i in range(3) for j in range(3)])
    for u in units:
        for a in u:
            for b in u:
                if a != b:
                    peers[a].add(b)
    return [sorted(p) for p in peers], units


PEERS, UNITS = _build()
BOX = [(r // 3) * 3 + c // 3 for r in range(9) for c in range(9)]


@dataclass
class Stats:
    assignments: int = 0
    backtracks: int = 0
    revisions: int = 0
    prunes: int = 0

    def add(self, o):
        self.assignments += o.assignments
        self.backtracks += o.backtracks
        self.revisions += o.revisions
        self.prunes += o.prunes


class CapReached(Exception):
    pass


@dataclass
class Result:
    status: str            # "solved" | "no_solution" | "cap"
    solutions: list
    stats: Stats
    message: str = ""

    @property
    def solved(self):
        return self.status == "solved"


def conflicts(grid):
    """Pairs (a, b) of GIVEN cells with the same digit in a shared row/col/box."""
    out = []
    for u in UNITS:
        seen = {}
        for c in u:
            d = grid[c]
            if d:
                seen.setdefault(d, []).append(c)
        for cells in seen.values():
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    out.append((cells[i], cells[j]))
    return sorted(set(out))


def ac3(dom, stats):
    """AC-3 on the binary '!=' constraints. Returns False if some domain becomes empty."""
    q = deque((i, j) for i in range(81) for j in PEERS[i])
    while q:
        xi, xj = q.popleft()
        stats.revisions += 1
        dj = dom[xj]
        if dj & (dj - 1) == 0 and dom[xi] & dj:   # xj has a single value -> xi cannot use it
            dom[xi] &= ~dj
            stats.prunes += 1
            if dom[xi] == 0:
                return False
            for xk in PEERS[xi]:
                if xk != xj:
                    q.append((xk, xi))
    return True


def initial_domains(grid):
    return [(1 << (d - 1)) if d else FULL for d in grid]


def _solve_bt(grid, mrv, max_assign, max_solutions, stats):
    rowm, colm, boxm = [0] * 9, [0] * 9, [0] * 9
    g = list(grid)
    for i, d in enumerate(g):
        if d:
            b = 1 << (d - 1)
            r, c = divmod(i, 9)
            if (rowm[r] | colm[c] | boxm[BOX[i]]) & b:
                return []  # two equal givens in a unit
            rowm[r] |= b; colm[c] |= b; boxm[BOX[i]] |= b
    empties = [i for i in range(81) if g[i] == 0]
    sols = []

    def rec(k, left):
        if not left:
            sols.append(list(g))
            return len(sols) >= max_solutions
        if mrv:
            best, bm = None, 0
            bc = 10
            for i in left:
                r, c = divmod(i, 9)
                m = FULL & ~(rowm[r] | colm[c] | boxm[BOX[i]])
                n = bin(m).count("1")
                if n < bc:
                    best, bm, bc = i, m, n
                    if n <= 1:
                        break
            i, m = best, bm
            rest = [x for x in left if x != i]
        else:
            i = left[0]
            r, c = divmod(i, 9)
            m = FULL & ~(rowm[r] | colm[c] | boxm[BOX[i]])
            rest = left[1:]
        r, c = divmod(i, 9)
        for d in range(1, 10):
            b = 1 << (d - 1)
            if not m & b:
                continue
            stats.assignments += 1
            if max_assign is not None and stats.assignments > max_assign:
                raise CapReached
            g[i] = d
            rowm[r] |= b; colm[c] |= b; boxm[BOX[i]] |= b
            if rec(k + 1, rest):
                return True
            g[i] = 0
            rowm[r] &= ~b; colm[c] &= ~b; boxm[BOX[i]] &= ~b
            stats.backtracks += 1
        return False

    rec(0, empties)
    return sols


def _solve_fc(grid, use_ac3, mrv, max_assign, max_solutions, stats):
    dom = initial_domains(grid)
    assigned = [d != 0 for d in grid]
    # forward-check the givens (pre-processing, not counted as search)
    for i, d in enumerate(grid):
        if d:
            b = 1 << (d - 1)
            for p in PEERS[i]:
                if dom[p] & b and not assigned[p]:
                    dom[p] &= ~b
                elif assigned[p] and grid[p] == d:
                    return []   # equal givens
                if dom[p] == 0:
                    return []
    if use_ac3 and not ac3(dom, stats):
        return []
    sols = []

    def rec(dom, assigned):
        free = [i for i in range(81) if not assigned[i]]
        if not free:
            sols.append([d.bit_length() for d in dom])
            return len(sols) >= max_solutions
        if mrv:
            i = min(free, key=lambda x: (bin(dom[x]).count("1"), x))
        else:
            i = free[0]
        m = dom[i]
        for d in range(1, 10):
            b = 1 << (d - 1)
            if not m & b:
                continue
            stats.assignments += 1
            if max_assign is not None and stats.assignments > max_assign:
                raise CapReached
            nd = list(dom)
            nd[i] = b
            ok = True
            for p in PEERS[i]:
                if not assigned[p] and nd[p] & b:
                    nd[p] &= ~b
                    if nd[p] == 0:
                        ok = False
                        break
            if ok:
                na = list(assigned)
                na[i] = True
                if rec(nd, na):
                    return True
            stats.backtracks += 1
        return False

    rec(dom, assigned)
    return sols


def solve(grid, fc=False, ac3_pre=False, mrv=None, max_assign=None, max_solutions=1):
    """Solve a Sudoku. grid: list of 81 ints (0 = empty).

    fc      : use Forward Checking (otherwise plain Backtracking)
    ac3_pre : run AC-3 before the search (only meaningful together with fc)
    mrv     : variable order. None -> MRV when fc is on, row-major order when off.
    """
    stats = Stats()
    if mrv is None:
        mrv = fc
    try:
        if fc or ac3_pre:
            sols = _solve_fc(grid, ac3_pre, mrv, max_assign, max_solutions, stats)
        else:
            sols = _solve_bt(grid, mrv, max_assign, max_solutions, stats)
    except CapReached:
        return Result("cap", [], stats, "assignment budget exhausted")
    if sols:
        return Result("solved", sols, stats)
    return Result("no_solution", [], stats, "no solution: the givens are inconsistent or cannot be completed")


VARIANTS = {
    "BT": dict(fc=False, ac3_pre=False),
    "BT+FC": dict(fc=True, ac3_pre=False),
    "BT+FC+AC3": dict(fc=True, ac3_pre=True),
}


def is_valid_solution(sol, puzzle=None):
    if sol is None or len(sol) != 81:
        return False
    for u in UNITS:
        if sorted(sol[c] for c in u) != list(range(1, 10)):
            return False
    if puzzle is not None:
        return all(p == 0 or p == s for p, s in zip(puzzle, sol))
    return True
