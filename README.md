# P7 - Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)
Ibrahim Mohamed - Section 4 - AID

## Setup
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## Run (exact commands, in order)
    python make_data.py            # downloads MNIST mirror, preprocesses (seed 2026), writes data/DATA_DICTIONARY.md
    python eval/train_nn.py        # trains MLP + logistic regression, calibrates, writes results/nn_metrics.json, nn_eval.png
    python run_demo.py             # integrated demo (console log + results/demo.png); options: --seed --misreads --strategy
    python eval/experiments.py     # 3 experiments x 5 seeds (~5 min on 1 CPU) -> results/*.json, *.png, correction_log.csv
    python -m pytest tests -q      # 16 tests: 5 normal, 3 edge, 2 impossible, search known-answer + brute force
    python eval/build_report.py    # builds report/report.pdf from the results

## Modules
- `sudoku/nn.py` MLP + temperature scaling, ECE | `sudoku/csp.py` BT / FC / AC-3 (switchable, counters)
- `sudoku/search.py` UCS / Greedy / A* over correction sets | `sudoku/generator.py` puzzle + mosaic generator (seeded)
- `sudoku/integration.py` full pipeline | `eval/` experiments and report | `tests/`
