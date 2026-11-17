"""nn.py - small numpy MLP digit classifier (784-256-128-9) with temperature scaling.

Written in plain numpy so there is no heavy dependency and training takes seconds.
Classes are digits 1..9 stored as indices 0..8.
"""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes=(784, 256, 128, 9), seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.T = 1.0  # temperature (fitted on validation set)

    # ---- forward / backward
    def logits(self, x):
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
        return h

    def predict_proba(self, x, T=None):
        T = self.T if T is None else T
        return softmax(self.logits(x) / T)

    def fit(self, x, y, xv, yv, epochs=15, lr=1e-3, bs=128, seed=0, wd=1e-5, verbose=True):
        rng = np.random.default_rng(seed)
        y0 = y - 1
        yv0 = yv - 1
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        t = 0
        best, best_state = -1, None
        for ep in range(epochs):
            idx = rng.permutation(len(x))
            cur_lr = lr * (0.85 ** ep)
            for s in range(0, len(x), bs):
                bi = idx[s:s + bs]
                xb, yb = x[bi], y0[bi]
                acts = [xb]
                h = xb
                for i, (W, b) in enumerate(zip(self.W, self.b)):
                    h = h @ W + b
                    if i < len(self.W) - 1:
                        h = np.maximum(h, 0)
                    acts.append(h)
                p = softmax(h)
                d = p
                d[np.arange(len(bi)), yb] -= 1
                d /= len(bi)
                grads = []
                for i in reversed(range(len(self.W))):
                    gW = acts[i].T @ d + wd * self.W[i]
                    gb = d.sum(0)
                    grads.append((gW, gb))
                    if i > 0:
                        d = (d @ self.W[i].T) * (acts[i] > 0)
                grads = grads[::-1]
                gl = [g[0] for g in grads] + [g[1] for g in grads]
                t += 1
                for k, (p_, g) in enumerate(zip(self.W + self.b, gl)):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    mh = m[k] / (1 - 0.9 ** t)
                    vh = v[k] / (1 - 0.999 ** t)
                    p_ -= cur_lr * mh / (np.sqrt(vh) + 1e-8)
            acc = (self.logits(xv).argmax(1) == yv0).mean()
            if verbose:
                print(f"epoch {ep + 1:2d}  val acc {acc:.4f}")
            if acc > best:
                best = acc
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
        self.W, self.b = best_state
        return best

    # ---- calibration
    def fit_temperature(self, xv, yv):
        z = self.logits(xv)
        y0 = yv - 1
        best_T, best_nll = 1.0, 1e9
        for T in np.linspace(0.5, 4.0, 141):
            p = softmax(z / T)
            nll = -np.log(p[np.arange(len(y0)), y0] + 1e-12).mean()
            if nll < best_nll:
                best_T, best_nll = float(T), nll
        self.T = best_T
        return best_T

    def save(self, path):
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, T=self.T, n=len(self.W), **d)

    @staticmethod
    def load(path):
        z = np.load(path)
        n = int(z["n"])
        m = MLP.__new__(MLP)
        m.W = [z[f"W{i}"] for i in range(n)]
        m.b = [z[f"b{i}"] for i in range(n)]
        m.sizes = tuple([m.W[0].shape[0]] + [w.shape[1] for w in m.W])
        m.T = float(z["T"])
        return m


def ece(probs, y, bins=15):
    """Expected calibration error. y in 1..9."""
    conf = probs.max(1)
    pred = probs.argmax(1) + 1
    corr = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.any():
            e += mk.mean() * abs(corr[mk].mean() - conf[mk].mean())
    return float(e)


def nll(probs, y):
    return float(-np.log(probs[np.arange(len(y)), y - 1] + 1e-12).mean())
