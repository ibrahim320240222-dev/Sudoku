"""make_data.py - download MNIST and preprocess it reproducibly (seed fixed).

Source: the original MNIST database (LeCun et al.). The yann.lecun.com page is not
always reachable, so we use the widely used mirror inside M. Nielsen's repository
(mnist.pkl.gz). It contains the SAME images, already split 50,000 train / 10,000
validation (last 10k of the original training set) / 10,000 test (original test set).

Preprocessing:  pixels scaled to 0-1, digit 0 dropped (Sudoku only has 1-9),
labels stored as 1..9.  Output: data/mnist19.npz
"""
import gzip, os, pickle, urllib.request
import numpy as np

SEED = 2026
URL = ("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
       "master/data/mnist.pkl.gz")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "mnist.pkl.gz")
OUT = os.path.join(DATA, "mnist19.npz")

DICTIONARY = """# Data dictionary (P7 - Handwritten Sudoku Reader and Solver)

| Item | Real or generated | Description |
|---|---|---|
| x_train/x_val/x_test | REAL | 784 floats in [0,1] = a 28x28 MNIST digit image (flattened row by row). |
| y_train/y_val/y_test | REAL | Digit label 1..9 (digit 0 removed). |
| Split sizes | REAL | ~45k train / ~9k validation / ~9k test after removing the digit 0. |
| Sudoku solution grid | GENERATED | Full valid 9x9 grid built by randomised backtracking (seeded). |
| Sudoku puzzle | GENERATED | Cells removed one by one while the solution stays UNIQUE, until the wanted number of givens (25/30/35) is reached. |
| Mosaic (252x252 image) | GENERATED | 81 tiles of 28x28. Given cell -> a real MNIST TEST image of that digit. Blank cell -> zeros + tiny Gaussian noise (sigma 0.01). |
| Clean digit | GENERATED (selection rule) | A test image of digit d that the network reads correctly, so the number of misreads in a puzzle equals exactly the number we inject. |
| Injected hard digit | GENERATED (selection rule) | A test image of digit d that the network misreads AND where d is the network's second-best guess (so a correction is possible). |
| Blank-cell rule | RULE | A tile is blank if its mean brightness < 0.02 (not decided by the network). |
"""

def main():
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(RAW):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, RAW)
    with gzip.open(RAW, "rb") as f:
        tr, va, te = pickle.load(f, encoding="latin1")
    def prep(d):
        x, y = d
        keep = y != 0
        return x[keep].astype(np.float32), y[keep].astype(np.int64)
    (xtr, ytr), (xva, yva), (xte, yte) = prep(tr), prep(va), prep(te)
    np.savez_compressed(OUT, x_train=xtr, y_train=ytr, x_val=xva, y_val=yva,
                        x_test=xte, y_test=yte, seed=SEED)
    with open(os.path.join(DATA, "DATA_DICTIONARY.md"), "w") as f:
        f.write(DICTIONARY)
    print("train", xtr.shape, "val", xva.shape, "test", xte.shape, "->", OUT)

if __name__ == "__main__":
    main()
