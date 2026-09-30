# EXP-03 Multi-Finger Aggregation Ablation

Comparison of diagnostic screening accuracy as a function of the number of fingernail captures per patient ($K \in \{1, 2, 4, 8\}$) and aggregation strategy:

| Fingers Evaluated ($K$) | Aggregation Method | Accuracy | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `mean` | 98.2% | 97.2% | 100.0% |
| 1 | `median` | 98.2% | 97.2% | 100.0% |
| 1 | `majority_vote` | 98.2% | 97.2% | 100.0% |
| 2 | `mean` | 92.9% | 88.9% | 100.0% |
| 2 | `median` | 92.9% | 88.9% | 100.0% |
| 2 | `majority_vote` | 92.9% | 88.9% | 100.0% |
| 4 | `mean` | 91.1% | 86.1% | 100.0% |
| 4 | `median` | 98.2% | 97.2% | 100.0% |
| 4 | `majority_vote` | 96.4% | 94.4% | 100.0% |
| 8 | `mean` | 94.6% | 91.7% | 100.0% |
| 8 | `median` | 100.0% | 100.0% | 100.0% |
| 8 | `majority_vote` | 100.0% | 100.0% | 100.0% |
