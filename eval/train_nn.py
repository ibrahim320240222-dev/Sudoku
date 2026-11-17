"""Train the NN + logistic-regression baseline, calibrate, save metrics and figures."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sudoku.nn import MLP, ece, nll

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "data", "mnist19.npz"))
xtr, ytr, xva, yva, xte, yte = (d[k] for k in
                                ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"])
t0 = time.time()
net = MLP(seed=0)
net.fit(xtr, ytr, xva, yva, epochs=15, seed=0)
net.save(os.path.join(ROOT, "data", "mlp_raw.npz"))
p_raw = net.predict_proba(xte, T=1.0)
T = net.fit_temperature(xva, yva)
p_cal = net.predict_proba(xte)
net.save(os.path.join(ROOT, "data", "mlp.npz"))
print("NN trained in %.1fs, T=%.2f" % (time.time() - t0, T))

t0 = time.time()
lr = LogisticRegression(max_iter=150)
lr.fit(xtr, ytr)
p_lr = lr.predict_proba(xte)
print("LR trained in %.1fs" % (time.time() - t0))

def metrics(p):
    return dict(accuracy=float((p.argmax(1) + 1 == yte).mean()), ece=ece(p, yte), nll=nll(p, yte))

res = {"MLP (temperature scaled)": metrics(p_cal), "MLP (raw)": metrics(p_raw),
       "Logistic regression": metrics(p_lr), "temperature": T,
       "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte))}
json.dump(res, open(os.path.join(ROOT, "results", "nn_metrics.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

cm = confusion_matrix(yte, p_cal.argmax(1) + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].imshow(cm, cmap="Blues")
ax[0].set_xticks(range(9)); ax[0].set_xticklabels(range(1, 10))
ax[0].set_yticks(range(9)); ax[0].set_yticklabels(range(1, 10))
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title("Confusion matrix (MLP, test)")
for i in range(9):
    for j in range(9):
        if cm[i, j]:
            ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
def rel(p, label, c):
    conf = p.max(1); corr = (p.argmax(1) + 1 == yte)
    edges = np.linspace(0, 1, 11); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.sum() > 20:
            xs.append(conf[mk].mean()); ys.append(corr[mk].mean())
    ax[1].plot(xs, ys, "o-", label=label, color=c)
rel(p_raw, "MLP raw", "tab:red"); rel(p_cal, "MLP + temperature", "tab:green")
rel(p_lr, "Logistic regression", "tab:blue")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[1].set_xlabel("mean confidence"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Reliability diagram")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "results", "nn_eval.png"), dpi=150)
