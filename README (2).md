# Equity-Aware Deep Reinforcement Learning for Human-Robot Collaborative Task Allocation

**Paper:** Equity-Aware Deep Reinforcement Learning for Human-Robot Collaborative Task Allocation in Resource-Constrained Manufacturing: A Four-Algorithm Comparative Study

**Authors:** Salisu Auwal Musa, Dr. Sanjay Choudhary
**Institution:** Vivekananda Global University, Jaipur, India
**Status:**

\---

## Overview

This repository contains the complete simulation codebase for a five-algorithm comparative study benchmarking an equity-aware D3QN framework against PPO, PER-n³-D3QN, EBQ-lite, and a D3QN-NStep ablation condition, in a cement bagging Human-Robot Collaboration (HRC) simulation representative of resource-constrained developing-economy manufacturing.

The framework explicitly models worker fatigue, task equity bias, operational throughput, and quality error rate as parametric objectives within a composite reward function. All experiments are run across 20 independent random seeds with statistical inference via paired t-tests and Cohen's d effect sizes.

\---

## Repository Structure

```
├── experiment\_gpu\_final.py          # Step 1 — Main training run (all 5 algorithms)
├── robustness\_and\_figures.py        # Step 2 — Robustness testing + Figures 2–5
├── regenerate\_figures.py            # Step 3 — Regenerate Figure 1 (reward ablation)
├── equitable\_d3qn\_hrc\_v2\_\_1\_.py    # Step 4 — Sensitivity analysis + skill generalisation
└── README.md
```

\---

## Requirements

All experiments were run on **Kaggle** (free cloud GPU). No local GPU is required.

**Python:** 3.9  
**Key dependencies:**

```
torch >= 1.13
numpy
matplotlib
pandas
scipy
openpyxl
```

These are all pre-installed in the Kaggle environment. No `pip install` step is needed if you run on Kaggle.

\---

## How to Reproduce Results

All four scripts are designed to be run on **Kaggle Notebooks** in sequence. Follow the steps below exactly.

\---

### Step 1 — Train All Five Algorithms

**Script:** `experiment\_gpu\_final.py`
**Produces:** `hrc\_checkpoint.pkl` (saved to `/kaggle/working/`)
**Runtime:** \~60–90 minutes (GPU T4) | \~8–12 hours (CPU — not recommended)
**Source of:** Tables 3 and 4 in the paper (final performance and statistical comparisons)

**Instructions:**

1. Go to [kaggle.com](https://kaggle.com) → **Create New Notebook**
2. Upload `experiment\_gpu\_final.py` or paste its contents into a code cell
3. In **Settings → Accelerator**, select **GPU T4 x2**
4. Ensure **Internet is ON** in Settings
5. Click **Run All**
6. After completion, click **Save Version** in Kaggle to commit outputs to storage — this preserves the checkpoint if your session restarts

**Output files:**

```
/kaggle/working/hrc\_checkpoint.pkl
```

> \*\*Important:\*\* The checkpoint file must exist before running Step 2. If your Kaggle session restarts before Step 2, reload the saved version to restore it.

\---

### Step 2 — Robustness Testing and Figures 2–5

**Script:** `robustness\_and\_figures.py`
**Requires:** `hrc\_checkpoint.pkl` from Step 1
**Runtime:** \~25–35 minutes (CPU) | \~8 minutes (GPU)
**Source of:** Table 5 (robustness), Figures 2, 3, 4, 5, and the Excel/CSV result tables

**Instructions:**

1. In the same Kaggle session (or a new one with the saved dataset), open a new cell
2. Paste the contents of `robustness\_and\_figures.py` and run
3. The script loads the checkpoint automatically from `/kaggle/working/`

**Output files:**

```
/kaggle/working/fig2\_training\_curves\_FINAL.png
/kaggle/working/fig3\_performance\_bars\_FINAL.png
/kaggle/working/fig4\_robustness\_heatmap\_FINAL.png
/kaggle/working/fig5\_radar\_chart\_FINAL.png
/kaggle/working/extended\_results.xlsx
/kaggle/working/performance\_summary.csv
/kaggle/working/statistical\_tests.csv
/kaggle/working/robustness\_testing.csv
```

\---

### Step 3 — Regenerate Figure 1 (Reward Function Ablation)

**Script:** `regenerate\_figures.py`
**Requires:** `hrc\_checkpoint.pkl` from Step 1
**Runtime:** \~2 minutes
**Source of:** Figure 1 in the paper (Equity D3QN vs Productivity-Only vs PPO)

**Instructions:**

1. In the same Kaggle session, paste `regenerate\_figures.py` into a new cell and run

**Output files:**

```
/kaggle/working/fig1\_ablation\_FINAL.png
```

> \*\*Note:\*\* The Productivity-Only condition in Figure 1 is derived from the D3QN training run via reward-scaling approximation, as no separate Productivity-Only agent was trained independently. This is a representative simulation of the welfare-productivity trade-off and is described as such in Section 4.1 of the paper.

\---

### Step 4 — Sensitivity Analysis and Skill-Level Generalisation

**Script:** `equitable\_d3qn\_hrc\_v2\_\_1\_.py`
**Runs independently** (does not require the checkpoint from Step 1)
**Runtime:** \~20–25 minutes (CPU)
**Source of:** Tables 6 and 7 in the paper (weight sensitivity and skill generalisation)

**Instructions:**

1. Create a new Kaggle Notebook (CPU is sufficient — no GPU needed)
2. Paste the contents of `equitable\_d3qn\_hrc\_v2\_\_1\_.py` into a code cell
3. Run All
4. Copy the printed output tables directly into manuscript tables

**Output:** Printed numerical results (no figure files)

\---

## Key Results Summary

|Metric|D3QN (Proposed)|PPO|PER-n³-D3QN|EBQ-lite|
|-|-|-|-|-|
|Error Rate ↓|**0.070 ± 0.023**|0.244 ± 0.099|0.099 ± 0.047|0.070 ± 0.024|
|Fatigue Index ↓|**0.048 ± 0.027**|0.658 ± 0.368|0.162 ± 0.139|0.048 ± 0.037|
|Throughput ↑|**0.623 ± 0.153**|0.619 ± 0.152|0.601 ± 0.155|0.623 ± 0.154|

All comparisons: 20 independent random seeds, episodes 71–100 (convergence zone), paired t-tests (α = 0.05).

D3QN achieves **71.3% lower error rate** and **92.8% lower worker fatigue** than PPO (p < 0.001, Cohen's d > 1.6).

\---

## Experimental Configuration

|Parameter|Value|
|-|-|
|Algorithms|D3QN, PPO, PER-n³-D3QN, EBQ-lite, D3QN-NStep|
|Random seeds|20 (seeds 0–19)|
|Episodes per seed|100 (500 steps each)|
|Convergence zone|Episodes 71–100|
|Network architecture|5-128-128-4 (value + advantage streams)|
|Optimizer|Adam (lr = 0.001)|
|Batch size|256 (GPU) / 32 (CPU)|
|Discount factor γ|0.95|
|Reward weights (baseline)|w = (0.5, 0.3, 0.1, 0.1)|
|Platform|Kaggle (GPU T4 x2)|
|Framework|PyTorch + CUDA|

\---

## Simulation Environment

The cement bagging HRC environment models:

* Poisson bag arrivals (λ = 1.2 bags/min)
* Three worker skill levels: Junior (α = 1.2), Intermediate (α = 1.0), Senior (α = 0.8)
* Exponential fatigue accumulation and recovery (β = 0.10)
* Stochastic robot downtime: p \~ Uniform(0.05, 0.20)
* Gaussian sensor noise (σ = 0.10 for robustness testing)
* Four discrete actions: Idle, Assist, TakeOver, SuggestBreak

\---

## Citation

If you use this code or build on this framework, please cite:

```
Musa, S. A., \& Choudhary, S. (2026). Equity-Aware Deep Reinforcement Learning 
for Human-Robot Collaborative Task Allocation in Resource-Constrained Manufacturing: 
A Four-Algorithm Comparative Study. International Journal of Robotics and Automation.
\[Under review]
```

\---

## Acknowledgements

GPU computational resources were provided via Kaggle's free-tier accelerator programme. The first author is supported by Dangote Cement PLC under a formal study leave arrangement.

\---

## License

This code is released for academic reproducibility. For commercial use or adaptation, please contact the corresponding author.

