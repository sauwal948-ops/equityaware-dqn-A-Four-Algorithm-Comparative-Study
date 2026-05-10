# Equity-Aware Deep Reinforcement Learning for Human-Robot Collaborative Task Allocation



This repository contains the full experimental code for the paper:

> \*\*"Equity-Aware Deep Reinforcement Learning for Human-Robot Collaborative Task Allocation in Resource-Constrained Manufacturing"\*\*

The framework integrates worker fatigue, task equity bias, operational throughput, and quality error rate as explicit parametric objectives within a Double Dueling Deep Q-Network (D3QN) architecture, validated through a five-algorithm comparative study on a cement bagging simulation.

\---

## Files in This Repository

|File|Purpose|
|-|-|
|`hrc\_experiment\_v2.py`|**Main experiment** — trains all 5 algorithms, generates Tables 3, 4, 6, 7 and Figures 1–4|
|`rerun\_robustness\_100ep.py`|**Robustness rerun** — generates Table 5 and Figure 5 with corrected settings|
|`README.md`|This file|

> ⚠️ \*\*Important:\*\* Run `hrc\_experiment\_v2.py` first (produces the checkpoint), then run `rerun\_robustness\_100ep.py` for the published robustness results. Do \*\*not\*\* use the built-in robustness output from `hrc\_experiment\_v2.py` for Table 5 — it uses N\_ROB=3 and ROB\_EPISODES=50, which were superseded by the dedicated rerun script (N\_ROB=7, ROB\_EPISODES=100).

\---

## Table of Contents

* [Requirements](#requirements)
* [Quick Start on Kaggle](#quick-start-on-kaggle)
* [Running Locally](#running-locally)
* [Code Structure](#code-structure)
* [Which Script Produces Which Output](#which-script-produces-which-output)
* [Output Files](#output-files)
* [Reproducing Key Results](#reproducing-key-results)
* [Implementation Notes and Bug Fixes](#implementation-notes-and-bug-fixes)
* [Checkpoint Resume](#checkpoint-resume)
* [Key Findings](#key-findings)
* [Citation](#citation)

\---

## Requirements

```bash
pip install torch numpy pandas matplotlib scipy openpyxl
```

|Environment|Expected Runtime|
|-|-|
|Kaggle GPU T4 x2 (recommended)|\~5–6 hours (main) + \~3.5 hours (rerun)|
|Local GPU (CUDA)|\~4–5 hours (main) + \~3 hours (rerun)|
|CPU only|\~8–10 hours (main) + \~4 hours (rerun)|

Python 3.10+, PyTorch 2.0+ required.

\---

## Quick Start on Kaggle

### Step 1 — Enable GPU

**Settings → Accelerator → GPU T4 x2**, Internet ON.

### Step 2 — Run the main experiment

Paste the entire contents of `hrc\_experiment\_v2.py` into a single notebook cell and run. Expected output:

```
Device: cuda
GPU: Tesla T4
\[no checkpoint — starting fresh]

\[1/6] Main training...
  D3QN (10 seeds × 200 ep)...
    D3QN DONE in \~28 min
  \[checkpoint saved — 95 KB]
  ...
\[6/6] Generating all figures and tables...
  Saved: fig1\_ablation\_FINAL.png
  ...
Total time: \~288 min
```

### Step 3 — Save Version immediately

Click **Save Version → Save \& Run All** before closing the tab. This commits the checkpoint and all outputs to permanent storage.

> ⚠️ Kaggle's `/kaggle/working` directory is wiped when a session ends. Save Version is the only way to keep your checkpoint.

### Step 4 — Run the robustness rerun

In a new cell (same notebook, new session), first copy the checkpoint:

```python
import shutil, os
# Find your checkpoint — run this to get the exact path
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        print(os.path.join(root, f))
```

Then copy it to working:

```python
shutil.copy('/kaggle/input/datasets/YOUR\_USERNAME/hrc-checkpoint-v2/hrc\_checkpoint\_v2.pkl',
            '/kaggle/working/hrc\_checkpoint\_v2.pkl')
```

Then paste and run `rerun\_robustness\_100ep.py`. Expected output:

```
Robustness Re-run FINAL
N\_ROB=7 seeds × ROB\_EPISODES=100 ep
...
Robustness complete in \~192 min
Checkpoint updated
Saved: fig5\_robustness\_heatmap\_FINAL.png
```

\---

## Running Locally

```bash
git clone https://github.com/YOUR\_USERNAME/hrc-equity-drl.git
cd hrc-equity-drl
pip install torch numpy pandas matplotlib scipy openpyxl

# Step 1: Run main experiment
# Edit OUTPUT\_DIR = './outputs' at top of script if needed
python hrc\_experiment\_v2.py

# Step 2: Run robustness rerun (uses checkpoint from Step 1)
# Edit OUTPUT\_DIR and CKPT\_FILE to match your local paths
python rerun\_robustness\_100ep.py
```

\---

## Code Structure

### hrc\_experiment\_v2.py (1,180 lines)

|Class / Function|Description|
|-|-|
|`CementBaggingEnv`|Simulation environment — MDP dynamics, fatigue model (Eq. 2), reward function (Eq. 1)|
|`D3QNetwork`|5-128-128-4 dueling neural network with kaiming initialization|
|`UniformBuffer`|Standard experience replay buffer|
|`PERBuffer` + `\_SumTree`|Prioritized Experience Replay via sum-tree data structure|
|`EBQBuffer`|Episodic buffer with episode-end reward boosting and positive-experience duplication|
|`\_NStep`|n-step return accumulator|
|`D3QNAgent`|Base D3QN agent with double-network target updates|
|`PERnAgent`|PER + n-step extension of D3QNAgent|
|`EBQAgent`|EBQ buffer extension of D3QNAgent|
|`NStepAgent`|n-step only extension of D3QNAgent (ablation)|
|`PPOAgent` + `PPONet`|Proximal Policy Optimization with clipped surrogate|
|`run\_algo()`|Multi-seed training orchestrator|
|`robustness()`|Robustness testing across 5 conditions (N\_ROB=3, 50ep — draft only)|
|`run\_sensitivity()`|Table 6: reward weight sensitivity analysis|
|`run\_skill\_gen()`|Table 7: skill-level generalization|
|`make\_fig1()` – `make\_fig5()`|Figure generation|
|`stats\_table()`|Paired t-tests and Cohen's d (Table 4)|
|`save\_ckpt()` / `load\_ckpt()`|Checkpoint save/resume|
|`main()`|6-stage orchestrator with per-algorithm checkpoint saves|

### rerun\_robustness\_100ep.py (634 lines)

Contains a standalone copy of the environment, all agents, and a dedicated robustness runner with:

* Bug 1 fixed (observation clipping)
* Bug 2 fixed (no double gradient update at episode end)
* N\_ROB=7 seeds, ROB\_EPISODES=100, convergence zone ep 71–100
* Mid-run checkpoint saves (resumes automatically if session is interrupted)

\---

## Which Script Produces Which Output

|Output|Script|Notes|
|-|-|-|
|Table 3 — Final performance|`hrc\_experiment\_v2.py`|✅ Use directly|
|Table 4 — Statistics|`hrc\_experiment\_v2.py`|✅ Use directly|
|**Table 5 — Robustness**|**`rerun\_robustness\_100ep.py`**|⚠️ Must use rerun|
|Table 6 — Sensitivity|`hrc\_experiment\_v2.py`|✅ Use directly|
|Table 7 — Skill-level|`hrc\_experiment\_v2.py`|✅ Use directly|
|Figures 1–4|`hrc\_experiment\_v2.py`|✅ Use directly|
|**Figure 5 — Robustness heatmap**|**`rerun\_robustness\_100ep.py`**|⚠️ Must use rerun|

\---

## Output Files

### Figures (generated by main script)

|File|Description|
|-|-|
|`fig1\_ablation\_FINAL.png`|Reward function ablation — Equity D3QN vs Productivity-Only vs PPO|
|`fig2\_training\_curves\_FINAL.png`|Training dynamics — all 5 algorithms, 4 metrics, ±1 SD|
|`fig3\_performance\_bars\_FINAL.png`|Final performance bar charts with broken y-axis|
|`fig4\_radar\_chart\_FINAL.png`|Multi-metric performance radar|
|`fig5\_robustness\_heatmap\_FINAL.png`|**Generated by rerun script** — robustness degradation heatmap|

### Data Tables

|File|Contents|
|-|-|
|`HRC\_Results\_ALL\_TABLES.xlsx`|Tables 3–7 in one Excel file (5 sheets; Table 5 updated by rerun)|
|`table3\_performance.csv`|Final performance — all algorithms|
|`table4\_statistics.csv`|Paired t-tests and Cohen's d|
|`table5\_robustness\_d3qn.csv`|**Generated by rerun script** — D3QN robustness|
|`table6\_sensitivity.csv`|Reward weight sensitivity|
|`table7\_skillgen.csv`|Skill-level generalization|
|`hrc\_checkpoint\_v2.pkl`|Full training checkpoint — all results, resumable|

\---

## Reproducing Key Results

### Table 3 — Final performance (episodes 141–200, 10 seeds)

|Algorithm|Throughput|Error Rate|Fatigue|
|-|-|-|-|
|D3QN|0.641 ± 0.016|0.070 ± 0.003|0.047 ± 0.002|
|PPO|0.640 ± 0.016|0.209 ± 0.104|0.550 ± 0.397|
|PER-n³-D3QN|0.628 ± 0.014|0.071 ± 0.006|0.077 ± 0.043|
|EBQ-lite|0.641 ± 0.016|0.070 ± 0.003|0.045 ± 0.004|
|D3QN-NStep|0.641 ± 0.016|0.070 ± 0.003|0.047 ± 0.003|

### Table 5 — Robustness (D3QN, 7 seeds × 100 ep)

|Condition|Error Degradation|Status|
|-|-|-|
|Clean baseline|0.0%|✓ PASS|
|Sensor noise 10%|−2.6%|✓ PASS|
|Sensor noise 20%|−4.1%|✓ PASS|
|Downtime 5–20%|+19.6%|✗ FAIL|
|Compound (noise+DT)|−10.8%|✓ PASS|

\---

## Implementation Notes and Bug Fixes

Two implementation issues were identified and corrected during the study. Both fixes are incorporated in the published code.

### Bug 1 — Observation clipping during noise conditions

**Location:** `CementBaggingEnv.\_observe()` in both scripts.

**Issue:** The original clipping line applied `np.clip()` to the entire observation vector, including `machine\_spd` (valid range 0–1.2). During noise conditions, this inadvertently capped `machine\_spd` at 1.0, distorting the observation distribution and producing artefactual negative degradation values in robustness results.

```python
# INCORRECT (original):
obs\[\[1, 3]] += noise
obs = np.clip(obs, 0.0, 1.0)   # clips machine\_spd incorrectly

# CORRECT (in both published files):
obs\[\[1, 3]] += noise
obs\[\[1, 3]] = np.clip(obs\[\[1, 3]], 0.0, 1.0)  # only clip noised channels
```

**Affected results:** Table 5 and Figure 5 only. Tables 3, 4, 6, 7 and Figures 1–4 are unaffected (main training uses no sensor noise).

### Bug 2 — No double gradient update

**Status:** Not present in the published `hrc\_experiment\_v2.py`. The training loop issues `learn()` every 4 steps and the `done` branch simply breaks without an additional call.

The `rerun\_robustness\_100ep.py` script adds an explicit final `learn()` call if the last step is not a multiple of `LEARN\_EVERY`, ensuring a gradient update on the terminal transition regardless of episode length.

\---

## Checkpoint Resume

If your session is interrupted, simply re-run the script. Completed stages are skipped:

```
\[checkpoint loaded — completed: \['D3QN', 'PPO', 'PER-n3-D3QN']]
  Skipping D3QN (checkpoint)
  Skipping PPO (checkpoint)
  Skipping PER-n3-D3QN (checkpoint)
  EBQ-lite (10 seeds × 200 ep)...   ← resumes here
```

The robustness rerun also saves progress after each algorithm, so interruption between algorithms loses no work.

### Persistent checkpointing on Kaggle

1. After the main run completes, go to kaggle.com → Datasets → **New Dataset**
2. Name it `hrc-checkpoint-v2`, upload `hrc\_checkpoint\_v2.pkl`
3. In your notebook, add it as an input (Settings → Add Input)
4. Add at the top of the rerun script:

```python
import shutil, os
src = '/kaggle/input/datasets/YOUR\_USERNAME/hrc-checkpoint-v2/hrc\_checkpoint\_v2.pkl'
dst = '/kaggle/working/hrc\_checkpoint\_v2.pkl'
if not os.path.exists(dst) and os.path.exists(src):
    shutil.copy(src, dst)
    print("Checkpoint loaded from dataset ✓")
```

\---

## Key Findings

1. **D3QN outperforms PPO** on welfare metrics (error rate 66.5% lower, fatigue 91.5% lower, p < 0.05, |d| > 1.2). Policy-gradient methods show bimodal convergence under composite welfare rewards (PPO fatigue SD = 0.397).
2. **Prioritized replay distorts welfare signals.** PER-n³-D3QN delivers significantly lower throughput (p < 0.001, d = 2.602) with no welfare benefit. The D3QN-NStep ablation confirms causal attribution: n-step returns alone are welfare-neutral; the priority mechanism is responsible.
3. **EBQ buffer components are unnecessary.** EBQ-lite is statistically indistinguishable from D3QN on all metrics (p > 0.13). The composite reward structure alone is sufficient.
4. **Sensor noise up to 20% is well-tolerated.** Equipment downtime (5–20%) is the primary deployment vulnerability (+19.6% error degradation). D3QN-NStep passes all five robustness conditions.
5. **The reward weight vector is adjustable without retraining.** Safety-First configuration (w₃ = 0.5) reduces fatigue by 82.6% with minimal throughput cost (−0.8%).

\---

## Citation

If you use this code or results, please cite:

```bibtex
@article{YOUR\_CITATION\_KEY,
  title   = {Equity-Aware Deep Reinforcement Learning for Human-Robot 
             Collaborative Task Allocation in Resource-Constrained Manufacturing},
  author  = {\[Author names]},
  journal = {\[Journal name]},
  year    = {2025},
}
```

\---

## License

MIT License. See [LICENSE](LICENSE) for details.

