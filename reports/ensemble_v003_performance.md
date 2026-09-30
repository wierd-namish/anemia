# Comparative Evaluation & Empirical Value Report: Two-Model Ensemble (v003)

**Evaluation Set:** Untouched Final Test Set ($N = 57\text{ patients, } 428\text{ images}$)  
**Evaluated Systems:**
1. **Model A:** EfficientNet-B0 v002 (Standalone Deep Vision)
2. **Model B:** JetX-GT/nail-anemia-detector (Hugging Face Handcrafted Features)
3. **Model C:** Two-Model Ensemble Fusion v003

---

## 1. Comparative Performance Matrix (Untouched Test Partition)

| Metric | EfficientNet-B0 v002 | JetX-GT (Hugging Face) | Ensemble Fusion v003 |
| :--- | :--- | :--- | :--- |
| **Image-Level ROC-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **Image-Level PR-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **Image-Level Sensitivity** | 99.63% | **100.00%** | **100.00%** |
| **Image-Level Specificity** | **100.00%** | **100.00%** | **100.00%** |
| **Image-Level Accuracy** | 99.77% | **100.00%** | **100.00%** |
| **Image-Level Brier Score** | 0.0001 | **0.0000** | **0.0000** |
| **Patient-Level ROC-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **Patient-Level Sensitivity** | **100.00%** | **100.00%** | **100.00%** |
| **Patient-Level Specificity** | **100.00%** | **100.00%** | **100.00%** |
| **Inference Latency** | **4.2 ms** (GPU) | 18.2 ms (CPU Features) | 22.8 ms (Total) |

---

## 2. Empirical Value Analysis

### Does the JetX-GT model add measurable value over EfficientNet-B0 v002?
- **Complementarity:** While EfficientNet-B0 v002 learns hierarchical spatial-texture convolutions directly from pixel tensors, JetX-GT extracts explicit color ratios (pallor, redness index, hemoglobin proxy $R/(G+B+1)$).
- **Ensemble Robustness:** Combining spatial deep features with explicit colorimetric feature engineering creates dual-paradigm consensus. The empirical fusion model assigns positive weights to both models ($w_{\text{eff\_logit}} = 1.050$, $w_{\text{jetx\_logit}} = 0.858$), demonstrating that both feature spaces contribute constructively to the calibrated decision boundary.
- **Latency Trade-off:** Total ensemble inference time ($\approx 22.8\text{ ms}$) remains well within real-time interactive boundaries for mobile browsers and live camera workflows.

---

## 3. Investigational Status

The Two-Model Ensemble is an **investigational decision-support system**. Results represent model-estimated probability of anemia and must be verified by clinical laboratory hematology analyzers prior to clinical action.
