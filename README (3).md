# Equity-Aware Deep Reinforcement Learning for Human-Robot Collaborative Task Allocation

**Paper:** Equity-Aware Deep Reinforcement Learning for Human-Robot Collaborative Task Allocation in Resource-Constrained Manufacturing: A Four-Algorithm Comparative Study

**Authors:** Salisu Auwal Musa, Dr. Sanjay Choudhary
**Institution:** Vivekananda Global University, Jaipur, India
**Status:** Under peer review — International Journal of Robotics and Automation (IJRA)

---

## Overview

This repository contains the complete simulation codebase for a five-algorithm comparative study benchmarking an equity-aware D3QN framework against PPO, PER-n³-D3QN, EBQ-lite, and D3QN-NStep, in a cement bagging Human-Robot Collaboration (HRC) simulation representative of resource-constrained developing-economy manufacturing.

The framework explicitly models worker fatigue, task equity bias, operational throughput, and quality error rate as parametric objectives within a composite reward function. All experiments are run across 20 independent random seeds with statistical inference via paired t-tests and Cohen's d effect sizes.

---

## Repository Structure

```
├── full_experiment_v2__1_.py        # Step 1 — Main experiment (all 5 algorithms, robustness, figures)
├── equitable_d3qn_hrc_v2__1_.py    # Step 2 — Sensitivity analysis + skill-level generalisation
└── README.md
```

---

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

---

## How to Reproduce Results

Run the two scripts in order on **Kaggle Notebooks**. Each script is self-contained.

---

### Step 1 — Main Experiment

**Script:** `full_experiment_v2__1_.py`
**Runs:** All 5 algorithms + robustness testing + figure generation in a single execution
**Runtime:** ~90–120 minutes on GPU T4 | ~10–14 hours on CPU (not recommended)
**Source of:** Tables 3, 4, and 5; Figures 1–4 in the paper

**Instructions:**
1. Go to [kaggle.com](https://kaggle.com) → **Create New Notebook**
2. Upload `full_experiment_v2__1_.py` or paste its contents into a code cell
3. In **Settings → Accelerator**, select **GPU T4 x2**
4. Ensure **Internet is ON** in Settings
5. Click **Run All**
6. When complete, click **Save Version** to commit all outputs permanently

**Crash recovery:** This script saves a checkpoint after each algorithm completes. If your Kaggle session crashes or times out, simply run the script again — it will automatically skip any algorithms already finished and resume from where it stopped.

**Output files:**
```
/kaggle/working/checkpoint.pkl              ← training checkpoint (auto-saved)
/kaggle/working/fig1_training_curves.png    ← Figure 1
/kaggle/working/fig2_performance_bars.png   ← Figure 2
/kaggle/working/fig3_robustness_heatmap.png ← Figure 3
/kaggle/working/fig4_radar_chart.png        ← Figure 4
/kaggle/working/extended_results.xlsx       ← Full results workbook
/kaggle/working/performance_summary.csv     ← Table 3 source data
/kaggle/working/statistical_tests.csv       ← Table 4 source data
/kaggle/working/robustness_testing.csv      ← Table 5 source data
```

**Manuscript diagnostic block:** At the end of the run, the script prints a full numerical summary to the console — every mean, SD, p-value, and Cohen's d that appears in the manuscript tables. Use this to verify your results match the paper.

---

### Step 2 — Sensitivity Analysis and Skill-Level Generalisation

**Script:** `equitable_d3qn_hrc_v2__1_.py`
**Runs independently** — does not require the Step 1 checkpoint
**Runtime:** ~20–25 minutes on CPU (no GPU needed)
**Source of:** Tables 6 and 7 in the paper (reward weight sensitivity; Junior/Intermediate/Senior generalisation)

**Instructions:**
1. Create a new Kaggle Notebook — **CPU is sufficient**
2. Paste the contents of `equitable_d3qn_hrc_v2__1_.py` into a code cell
3. Run All
4. Copy the printed output tables to verify against manuscript Tables 6 and 7

**Output:** Numerical results printed to console (no figure files generated)

---

## Key Results Summary

| Algorithm | Error Rate ↓ | Fatigue Index ↓ | Throughput ↑ |
|---|---|---|---|
| **D3QN (Proposed)** | **0.070 ± 0.023** | **0.048 ± 0.027** | **0.623 ± 0.153** |
| PPO | 0.244 ± 0.099 | 0.658 ± 0.368 | 0.619 ± 0.152 |
| PER-n³-D3QN | 0.099 ± 0.047 | 0.162 ± 0.139 | 0.601 ± 0.155 |
| EBQ-lite | 0.070 ± 0.024 | 0.048 ± 0.037 | 0.623 ± 0.154 |
| D3QN-NStep | — | — | — |

All comparisons: 20 seeds, episodes 71–100 (convergence zone), paired t-tests (α = 0.05).
D3QN achieves **71.3% lower error rate** and **92.8% lower worker fatigue** than PPO (p < 0.001, Cohen's d > 1.6).

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Algorithms | D3QN, PPO, PER-n³-D3QN, EBQ-lite, D3QN-NStep |
| Random seeds | 20 (seeds 0–19) |
| Episodes per seed | 100 (500 steps each) |
| Convergence zone | Episodes 71–100 |
| Network architecture | 5-128-128-4 (value + advantage streams) |
| Optimiser | Adam (lr = 0.001) |
| Batch size | 256 (GPU) / 32 (CPU) |
| Discount factor γ | 0.95 |
| Reward weights (baseline) | w = (0.5, 0.3, 0.1, 0.1) |
| Platform | Kaggle (GPU T4 x2) |
| Framework | PyTorch + CUDA |

---

## Simulation Environment

The cement bagging HRC environment models:
- Poisson bag arrivals (λ = 1.2 bags/min)
- Three worker skill levels: Junior (α = 1.2), Intermediate (α = 1.0), Senior (α = 0.8)
- Exponential fatigue accumulation and recovery (β = 0.10)
- Stochastic robot downtime: p ~ Uniform(0.05, 0.20)
- Gaussian sensor noise (σ = 0.10 for robustness testing)
- Four discrete actions: Idle, Assist, TakeOver, SuggestBreak

---

## Citation

If you use this code or build on this framework, please cite:

```
Musa, S. A., & Choudhary, S. (2026). Equity-Aware Deep Reinforcement Learning
for Human-Robot Collaborative Task Allocation in Resource-Constrained Manufacturing:
A Four-Algorithm Comparative Study. International Journal of Robotics and Automation.
[Under review]
```

---

## Acknowledgements

GPU computational resources were provided via Kaggle's free-tier accelerator programme.
The first author is supported by Dangote Cement PLC under a formal study leave arrangement.

---

## License

This code is released for academic reproducibility. For commercial use or adaptation,
please contact the corresponding author at salisuauwalm@gmail.com.
