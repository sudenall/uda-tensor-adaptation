# UDA-TFL Inspired Domain Adaptation (Tensor vs Vector)

##  Overview

In this project, I explored domain adaptation and tried to answer a simple but important question:

> Is aligning feature distributions enough to improve model performance?

To investigate this, I compared three different approaches:

* PCA + CORAL (vector-based)
* 2D-PCA + CORAL (tensor-based)
* A custom **UDA-TFL-inspired approach**

The main goal was:
 Improve performance on the target domain under distribution shift

---

## Problem

In real-world machine learning systems:

```text
Training data ≠ Production data
```

This mismatch leads to:

* Performance degradation
* Poor generalization
* Unreliable predictions

---

## Approach

### 🔹 Branch1 (Vector-based)

```
image → flatten → PCA → CORAL → classifier
```

This is the standard approach:
features are extracted first, then aligned.

---

###  Branch2 (Tensor-based)

```
image → 2D-PCA → tensor features → adaptation → classifier
```

The key idea here:
 preserve spatial structure before flattening

---

### UDA-TFL-inspired (Key Part)

Instead of aligning features after extraction, I used a projection that directly optimizes:

```python
objective = total_scatter 
            - alpha * marginal_disc 
            - beta * conditional_disc
```

In simple terms:

* keep useful information
* reduce domain gap
* preserve class structure

---

## Results

| Method               | MMD After | Accuracy   | Δ Acc       |
| -------------------- | --------- | ---------- | ----------- |
| PCA + CORAL          | 0.0189    | 0.5704     | -0.0648     |
| 2D-PCA + CORAL       | 0.0124    | 0.5796     | -0.0037     |
| **UDA-TFL-inspired** | 0.0146    | **0.5944** | **+0.0111** |

The best accuracy was achieved with the UDA-TFL-inspired approach.

---

## t-SNE Observations

### Domain perspective

* PCA → source and target are clearly separated 
* CORAL → strong domain mixing 
* UDA-TFL → controlled and balanced mixing 

### Class perspective

* PCA → scattered classes 
* CORAL → some class distortion 
* UDA-TFL → compact and well-separated clusters 

---

## Key Insight

What I observed during experiments:

```
MMD ↓
Domain mixing ↑
But accuracy does NOT improve (CORAL)
```

This led to a critical conclusion:

**Alignment alone is not sufficient**

---

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/sudenall/uda-tensor-adaptation.git
cd uda-tensor-adaptation
```

---

### 2. Create a virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Run the main experiment

```
python -m experiments.run_branch2_experiment
```

---

## Outputs

After running the experiment, you will get:

### Terminal output:

* MMD before/after
* Accuracy before/after
* CORAL vs UDA comparison

---

### Generated files:

```
results/
├── figures/
│   ├── tsne_pca.png
│   ├── tsne_coral.png
│   ├── tsne_uda.png
│   └── final_da_table.png
│
├── metrics/
│   └── final_metrics.csv
```

---

## Why This Matters (Industry Perspective)

Domain adaptation is important in many real-world applications:

* Fraud detection → user behavior changes over time
* Recommendation systems → new users and items
* Computer vision → different environments and lighting
* NLP → different domains (e.g., news vs social media)

---

## Real Challenge

Most methods focus on:

```
Aligning feature distributions
```

But this can lead to:

```
Over-alignment → loss of class-discriminative structure
```

---

## What This Project Shows

```
Best model ≠ lowest MMD
Best model = best trade-off
```

 The UDA-TFL-inspired approach achieves:

* good alignment
* preserved class structure
* improved accuracy

---

## Final Conclusion

> Effective domain adaptation should not only align domains,
> but also preserve class-discriminative information.

---

## Future Work

* Deep learning-based domain adaptation
* Adversarial methods (DANN)
* Stronger class-aware alignment techniques

---

## 

Sude Ünal
Computer Engineering Student
