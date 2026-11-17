# Data dictionary (P7 - Handwritten Sudoku Reader and Solver)

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
