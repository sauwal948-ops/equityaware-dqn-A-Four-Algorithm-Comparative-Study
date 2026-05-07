RESULT FROM KAGGLE

Device: cpu
WARNING: No GPU — this will be very slow on CPU.
  [no checkpoint — starting fresh]

=================================================================
 HRC Experiment v2  |  device=cpu  |  batch=64
 Main     : 10 seeds × 200 ep  |  conv zone ep 141–200
 Robustness: 3 seeds × 50 ep  |  conv zone ep 31–50
 Ablation : 10 seeds (Productivity-Only, Fig 1)
 Sens/Skill: 5 seeds each (Tables 6 & 7)
=================================================================

[1/6] Main training...
  D3QN (10 seeds × 200 ep)...
    D3QN DONE in 28.5 min
  [checkpoint saved — 95 KB at /kaggle/working/hrc_checkpoint_v2.pkl]
  PPO (10 seeds × 200 ep)...
    PPO DONE in 13.8 min
  [checkpoint saved — 190 KB at /kaggle/working/hrc_checkpoint_v2.pkl]
  PER-n3-D3QN (10 seeds × 200 ep)...
    PER-n3-D3QN DONE in 32.3 min
  [checkpoint saved — 285 KB at /kaggle/working/hrc_checkpoint_v2.pkl]
  EBQ-lite (10 seeds × 200 ep)...
    EBQ-lite DONE in 27.9 min
  [checkpoint saved — 380 KB at /kaggle/working/hrc_checkpoint_v2.pkl]
  D3QN-NStep (10 seeds × 200 ep)...
    D3QN-NStep DONE in 28.1 min
  [checkpoint saved — 475 KB at /kaggle/working/hrc_checkpoint_v2.pkl]

[2/6] Productivity-Only D3QN ablation (10 seeds, Fig 1)...
  D3QN (10 seeds × 200 ep)...
    D3QN DONE in 27.8 min
  [checkpoint saved — 570 KB at /kaggle/working/hrc_checkpoint_v2.pkl]

[3/6] Robustness (3 seeds × 50 ep × 5 conds × 5 algos)...
    Robustness done: D3QN
    Robustness done: PPO
    Robustness done: PER-n3-D3QN
    Robustness done: EBQ-lite
    Robustness done: D3QN-NStep
  [checkpoint saved — 571 KB at /kaggle/working/hrc_checkpoint_v2.pkl]

[4/6] Sensitivity analysis (5 seeds × 3 configs, Table 6)...
    Sensitivity done: Baseline (0.5,0.3,0.1,0.1)
    Sensitivity done: Safety-First (0.3,0.3,0.5,0.1)
    Sensitivity done: Production-Critical (0.8,0.15,0.05,0.0)
  [checkpoint saved — 572 KB at /kaggle/working/hrc_checkpoint_v2.pkl]

[5/6] Skill-level generalization (5 seeds × 3 levels, Table 7)...
    Skill-gen done: Junior (α=1.2)
    Skill-gen done: Intermediate (α=1.0)
    Skill-gen done: Senior (α=0.8)
  [checkpoint saved — 573 KB at /kaggle/working/hrc_checkpoint_v2.pkl]

[6/6] Generating all figures and tables...
  Saved: /kaggle/working/fig1_ablation_FINAL.png
  Saved: /kaggle/working/fig2_training_curves_FINAL.png
  Saved: /kaggle/working/fig3_performance_bars_FINAL.png
  Saved: /kaggle/working/fig4_radar_chart_FINAL.png
  Saved: /kaggle/working/fig5_robustness_heatmap_FINAL.png
  Saved: /kaggle/working/HRC_Results_ALL_TABLES.xlsx

Total time: 288 min

=================================================================
PASTE THIS OUTPUT TO CLAUDE
=================================================================

MEANS (ep 141–200, 10 seeds):

D3QN:
  throughput: 0.6411 +/- 0.0156
  error_rate: 0.0700 +/- 0.0027
  fatigue: 0.0468 +/- 0.0018
  action_conc: 0.7355 +/- 0.0009

PPO:
  throughput: 0.6400 +/- 0.0155
  error_rate: 0.2086 +/- 0.1036
  fatigue: 0.5501 +/- 0.3969
  action_conc: 0.5084 +/- 0.1834

PER-n3-D3QN:
  throughput: 0.6283 +/- 0.0143
  error_rate: 0.0708 +/- 0.0056
  fatigue: 0.0773 +/- 0.0430
  action_conc: 0.6482 +/- 0.0342

EBQ-lite:
  throughput: 0.6412 +/- 0.0156
  error_rate: 0.0697 +/- 0.0032
  fatigue: 0.0449 +/- 0.0038
  action_conc: 0.7357 +/- 0.0013

D3QN-NStep:
  throughput: 0.6409 +/- 0.0156
  error_rate: 0.0703 +/- 0.0028
  fatigue: 0.0467 +/- 0.0030
  action_conc: 0.7334 +/- 0.0043

STATISTICS TABLE (Table 4):
         Comparison     Metric  D3QN   PPO  Change_% p_value  Cohen_d Sig  PER-n3-D3QN  EBQ-lite  D3QN-NStep
        D3QN vs PPO error_rate 0.070 0.209     198.1   <0.05   -1.288 Yes          NaN       NaN         NaN
        D3QN vs PPO    fatigue 0.047 0.550    1076.7   <0.05   -1.207 Yes          NaN       NaN         NaN
        D3QN vs PPO throughput 0.641 0.640      -0.2   <0.05    0.878 Yes          NaN       NaN         NaN
D3QN vs PER-n3-D3QN error_rate 0.070   NaN       1.2  0.7244   -0.115  No        0.071       NaN         NaN
D3QN vs PER-n3-D3QN    fatigue 0.047   NaN      65.3  0.0598   -0.681  No        0.077       NaN         NaN
D3QN vs PER-n3-D3QN throughput 0.641   NaN      -2.0  <0.001    2.602 Yes        0.628       NaN         NaN
   D3QN vs EBQ-lite error_rate 0.070   NaN      -0.4  0.7037    0.124  No          NaN     0.070         NaN
   D3QN vs EBQ-lite    fatigue 0.047   NaN      -4.1  0.2312    0.406  No          NaN     0.045         NaN
   D3QN vs EBQ-lite throughput 0.641   NaN       0.0  0.1364   -0.517  No          NaN     0.641         NaN
 D3QN vs D3QN-NStep error_rate 0.070   NaN       0.5  0.3282   -0.327  No          NaN       NaN       0.070
 D3QN vs D3QN-NStep    fatigue 0.047   NaN      -0.2  0.9258    0.030  No          NaN       NaN       0.047
 D3QN vs D3QN-NStep throughput 0.641   NaN      -0.0   <0.05    0.859 Yes          NaN       NaN       0.641

D3QN ROBUSTNESS (Table 5):
          Condition  Throughput  Error Rate  Fatigue Index  Error Degradation (%) Status
     Clean baseline       0.621       0.152          0.293                    0.0 ✓ PASS
   Sensor noise 10%       0.603       0.093          0.109                  -39.0 ✓ PASS
   Sensor noise 20%       0.603       0.175          0.374                   15.2 ✗ FAIL
     Downtime 5-20%       0.633       0.148          0.277                   -2.3 ✓ PASS
Compound (noise+DT)       0.585       0.125          0.197                  -17.6 ✓ PASS

SENSITIVITY ANALYSIS (Table 6):
            Weight Config (w1,w2,w3,w4)  Throughput  Fatigue  Error Rate
             Baseline (0.5,0.3,0.1,0.1)       0.639    0.046       0.068
         Safety-First (0.3,0.3,0.5,0.1)       0.634    0.008       0.054
Production-Critical (0.8,0.15,0.05,0.0)       0.639    0.047       0.069

SKILL GENERALIZATION (Table 7):
         Skill Level  Throughput  Fatigue Index  Error Rate
      Junior (α=1.2)       0.639          0.052       0.070
Intermediate (α=1.0)       0.639          0.046       0.068
      Senior (α=0.8)       0.639          0.041       0.067

=================================================================
Output files in /kaggle/working/:
  fig1_ablation_FINAL.png
  fig2_training_curves_FINAL.png
  fig3_performance_bars_FINAL.png
  fig4_radar_chart_FINAL.png
  fig5_robustness_heatmap_FINAL.png
  HRC_Results_ALL_TABLES.xlsx  (Tables 3–7 + robustness all algos)
  table3_performance.csv
  table4_statistics.csv
  table5_robustness_d3qn.csv
  table6_sensitivity.csv
  table7_skillgen.csv
=================================================================