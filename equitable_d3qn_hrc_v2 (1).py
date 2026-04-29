"""
=============================================================================
Equity-Aware D3QN for Human-Robot Collaborative Task Allocation
Double Dueling Deep Q-Network (D3QN) + PPO Benchmark  — VERSION 2
=============================================================================
FIXES FROM V1:
  1. Numpy tensor warning fixed (np.array() before FloatTensor)
  2. Reward comparison removed — composite rewards not comparable across
     different weight configs. Stats now run on individual metrics only.
  3. Evaluation function added: both policies scored on same metric scale.

HOW TO RUN ON KAGGLE:
  1. kaggle.com → Create New Notebook
  2. Paste this entire file into a code cell
  3. Accelerator: None (CPU is fine, ~20–25 min)
  4. Run All
  5. Copy printed numbers into manuscript tables
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from scipy import stats

torch.backends.cudnn.deterministic = True


# =============================================================================
# ENVIRONMENT
# =============================================================================

class CementBaggingEnv:
    """
    State:   [machine_speed, worker_fatigue, queue_length, error_rate, skill_level]
    Actions: 0=Idle, 1=Assist, 2=TakeOver, 3=SuggestBreak
    Reward:  R = w1*throughput - w2*error - w3*fatigue - w4*bias  (Eq.1)
    """
    def __init__(self, skill_level=1, weights=(0.5, 0.3, 0.1, 0.1),
                 noise_std=0.0, downtime_range=(0.05, 0.05)):
        self.skill          = skill_level
        self.w              = weights
        self.noise_std      = noise_std
        self.downtime_range = downtime_range
        self.alpha          = [1.2, 1.0, 0.8][skill_level]  # fatigue accumulation
        self.beta           = 0.3                             # fatigue recovery
        self.reset()

    def reset(self):
        self.machine_speed = 1.0
        self.fatigue       = 0.0
        self.queue         = 10.0
        self.error_rate    = 0.05
        self.t             = 0
        self.max_t         = 500
        self.action_counts = [0, 0, 0, 0]
        return self._obs()

    def _obs(self):
        obs = np.array([
            self.machine_speed / 1.2,
            np.clip(self.fatigue, 0, 1),
            self.queue / 50.0,
            np.clip(self.error_rate, 0, 1),
            self.skill / 2.0
        ], dtype=np.float32)
        if self.noise_std > 0:
            obs[1] += np.random.normal(0, self.noise_std)
            obs[3] += np.random.normal(0, self.noise_std)
            obs = np.clip(obs, 0.0, 1.0)
        return obs

    def step(self, action):
        # Stochastic robot downtime
        p_down     = np.random.uniform(*self.downtime_range)
        robot_down = np.random.random() < p_down

        load_map  = [1.0, 0.6, 0.0, 0.0]
        speed_map = [1.0, 1.1, 0.7, 0.0]
        error_red = [0.0, 0.3, 0.5, 0.0]

        load  = load_map[action]  if not robot_down else 1.0
        speed = speed_map[action] if not robot_down else 0.8

        # Fatigue dynamics: df/dt = alpha*Load - beta*f  (Eq.2)
        if action == 3:
            self.fatigue = max(0.0, self.fatigue - self.beta * 2.0)
        else:
            self.fatigue += self.alpha * load * 0.02
            self.fatigue  = max(0.0, self.fatigue - self.beta * 0.01)
        self.fatigue = min(1.0, self.fatigue)

        # Throughput
        bags       = np.random.poisson(1.2) * speed * (1.0 - self.fatigue * 0.3)
        throughput = float(np.clip(bags / 1.2, 0.0, 1.0))

        # Error rate
        base_error      = 0.05 + self.fatigue * 0.10
        reduction       = error_red[action] * (1.0 - self.fatigue)
        self.error_rate = float(np.clip(base_error - reduction, 0.01, 0.50))
        if self.noise_std > 0:
            self.error_rate = float(np.clip(
                self.error_rate + np.random.normal(0, self.noise_std * 0.5),
                0.01, 0.50))

        # Queue
        self.queue = float(np.clip(
            self.queue + np.random.poisson(1.2) - bags, 0.0, 50.0))

        # Equity bias penalty
        self.action_counts[action] += 1
        bias = float(np.var(self.action_counts) / (max(self.action_counts) + 1e-6))
        bias = min(1.0, bias)

        # Composite reward (Eq.1)
        w1, w2, w3, w4 = self.w
        reward = w1 * throughput - w2 * self.error_rate \
               - w3 * self.fatigue  - w4 * bias

        self.t += 1
        done = (self.t >= self.max_t)
        info = {
            'throughput': throughput,
            'error':      self.error_rate,
            'fatigue':    self.fatigue,
            'bias':       bias,
            'action':     action
        }
        return self._obs(), reward, done, info


# =============================================================================
# D3QN NETWORK
# =============================================================================

class DuelingDQN(nn.Module):
    """
    Double Dueling DQN (D3QN):
      Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

    Dueling streams stabilise Q-estimates when multiple actions have similar
    welfare-constrained values (e.g. Idle vs Assist under low fatigue).
    Double DQN correction applied in training loop (online selects, target evaluates).
    """
    def __init__(self, state_dim=5, action_dim=4, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        shared    = self.shared(x)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# D3QN AGENT
# =============================================================================

class D3QNAgent:
    def __init__(self, state_dim=5, action_dim=4,
                 lr=0.001, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay_episodes=60,
                 buffer_size=10000, batch_size=32, target_update=100):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = (eps_start - eps_end) / eps_decay_episodes
        self.steps         = 0

        self.online_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self.online_net(s).argmax().item())

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch       = self.buffer.sample(self.batch_size)

        # FIX: convert to np.array first to avoid slow tensor warning
        states      = torch.FloatTensor(np.array([t.state      for t in batch]))
        actions     = torch.LongTensor( np.array([t.action     for t in batch])).unsqueeze(1)
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones       = torch.FloatTensor(np.array([t.done       for t in batch]))

        # Double DQN: online selects action, target evaluates it
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(states).gather(1, actions).squeeze(1)
        loss      = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())


# =============================================================================
# PPO AGENT  (benchmark only)
# =============================================================================

class PPOAgent:
    """
    Lightweight PPO for discrete actions — benchmark comparison only.
    Key limitation vs D3QN: implicit value estimation, no inspectable
    Q(s,a) per action, higher policy variance under welfare constraints.
    """
    def __init__(self, state_dim=5, action_dim=4,
                 lr=3e-4, gamma=0.95, clip_eps=0.2, ppo_epochs=4):
        self.gamma      = gamma
        self.clip_eps   = clip_eps
        self.ppo_epochs = ppo_epochs

        self.net         = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head  = nn.Linear(128, 1)

        all_params = (list(self.net.parameters()) +
                      list(self.policy_head.parameters()) +
                      list(self.value_head.parameters()))
        self.optimizer = optim.Adam(all_params, lr=lr)
        self.memory    = []

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            feat     = self.net(s)
            logits   = self.policy_head(feat)
            dist     = torch.distributions.Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def store(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))

    def update(self):
        if len(self.memory) < 32:
            return
        states, actions, old_log_probs, rewards, dones = zip(*self.memory)
        states        = torch.FloatTensor(np.array(states))
        actions       = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))

        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - float(d))
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            feat      = self.net(states)
            logits    = self.policy_head(feat)
            dist      = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            values    = self.value_head(feat).squeeze(1)

            ratio     = torch.exp(log_probs - old_log_probs.detach())
            advantage = (returns - values.detach())
            surr1     = ratio * advantage
            surr2     = torch.clamp(ratio,
                                    1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * advantage
            loss = (-torch.min(surr1, surr2).mean()
                    + 0.5 * nn.MSELoss()(values, returns))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.memory = []


# =============================================================================
# TRAINING RUNNERS
# =============================================================================

def run_d3qn(seed, weights=(0.5, 0.3, 0.1, 0.1), skill=1,
             noise_std=0.0, downtime_range=(0.05, 0.05), n_episodes=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env   = CementBaggingEnv(skill, weights, noise_std, downtime_range)
    agent = D3QNAgent()

    ep_rewards, ep_throughputs, ep_errors, ep_fatigues, ep_actions = \
        [], [], [], [], []

    for ep in range(n_episodes):
        state        = env.reset()
        total_reward = 0.0
        infos        = []

        while True:
            action                       = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            state        = next_state
            total_reward += reward
            infos.append(info)
            if done:
                break

        agent.decay_epsilon()
        ep_rewards.append(total_reward)
        ep_throughputs.append(float(np.mean([i['throughput'] for i in infos])))
        ep_errors.append(    float(np.mean([i['error']      for i in infos])))
        ep_fatigues.append(  float(np.mean([i['fatigue']    for i in infos])))
        ep_actions.append(   [i['action'] for i in infos])

    final_actions = [a for ep in ep_actions[-30:] for a in ep]
    action_dist   = [final_actions.count(a) / len(final_actions) for a in range(4)]

    return {
        'rewards':          ep_rewards,
        'throughputs':      ep_throughputs,
        'errors':           ep_errors,
        'fatigues':         ep_fatigues,
        'action_dist':      action_dist,
        'final_throughput': float(np.mean(ep_throughputs[-30:])),
        'final_error':      float(np.mean(ep_errors[-30:])),
        'final_fatigue':    float(np.mean(ep_fatigues[-30:])),
        'final_reward_sd':  float(np.std(ep_rewards[-30:])),
    }


def run_ppo(seed, weights=(0.5, 0.3, 0.1, 0.1), n_episodes=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env   = CementBaggingEnv(1, weights)
    agent = PPOAgent()

    ep_rewards, ep_throughputs, ep_errors, ep_fatigues = [], [], [], []

    for ep in range(n_episodes):
        state        = env.reset()
        total_reward = 0.0
        infos        = []

        while True:
            action, log_prob               = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, log_prob, reward, done)
            state        = next_state
            total_reward += reward
            infos.append(info)
            if done:
                break

        agent.update()
        ep_rewards.append(total_reward)
        ep_throughputs.append(float(np.mean([i['throughput'] for i in infos])))
        ep_errors.append(    float(np.mean([i['error']      for i in infos])))
        ep_fatigues.append(  float(np.mean([i['fatigue']    for i in infos])))

    return {
        'rewards':          ep_rewards,
        'throughputs':      ep_throughputs,
        'errors':           ep_errors,
        'fatigues':         ep_fatigues,
        'final_throughput': float(np.mean(ep_throughputs[-30:])),
        'final_error':      float(np.mean(ep_errors[-30:])),
        'final_fatigue':    float(np.mean(ep_fatigues[-30:])),
        'final_reward_sd':  float(np.std(ep_rewards[-30:])),
    }


def cohens_d_paired(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    return float(np.mean(diffs) / (np.std(diffs) + 1e-9))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    N_SEEDS     = 20
    EQUITY_W    = (0.5, 0.3, 0.1, 0.1)
    PROD_ONLY_W = (0.5, 0.3, 0.0, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("EXPERIMENT 1: Equity D3QN vs Productivity-Only Baseline")
    print(f"  {N_SEEDS} seeds | 100 episodes each | metrics compared independently")
    print("=" * 65)

    equity_res, prod_res = [], []
    for seed in range(N_SEEDS):
        print(f"  Training seed {seed+1}/{N_SEEDS}...", end='\r')
        equity_res.append(run_d3qn(seed, EQUITY_W))
        prod_res.append(  run_d3qn(seed, PROD_ONLY_W))
    print()

    eq_t = [r['final_throughput'] for r in equity_res]
    pr_t = [r['final_throughput'] for r in prod_res]
    eq_e = [r['final_error']      for r in equity_res]
    pr_e = [r['final_error']      for r in prod_res]
    eq_f = [r['final_fatigue']    for r in equity_res]
    pr_f = [r['final_fatigue']    for r in prod_res]
    eq_sd = np.mean([r['final_reward_sd'] for r in equity_res])
    pr_sd = np.mean([r['final_reward_sd'] for r in prod_res])

    # Statistical tests on individual metrics (valid — same measurement scale)
    t_t, p_t = stats.ttest_rel(eq_t, pr_t)
    t_e, p_e = stats.ttest_rel(eq_e, pr_e)
    t_f, p_f = stats.ttest_rel(eq_f, pr_f)

    d_t = cohens_d_paired(eq_t, pr_t)
    d_e = cohens_d_paired(pr_e, eq_e)   # reversed: reduction is positive effect
    d_f = cohens_d_paired(pr_f, eq_f)   # reversed: reduction is positive effect

    print(f"\n{'Metric':<26} {'Equity D3QN':>13} {'Prod. Only':>12} {'Change':>9}  p-value   d")
    print("-" * 82)

    for name, a, b, p, d in [
        ('Normalized Throughput', eq_t, pr_t, p_t, d_t),
        ('Error Rate',            eq_e, pr_e, p_e, d_e),
        ('Worker Fatigue Index',  eq_f, pr_f, p_f, d_f),
    ]:
        pct  = (np.mean(a) - np.mean(b)) / (np.mean(b) + 1e-9) * 100
        sig  = '*' if p < 0.05 else ' '
        print(f"  {name:<24} {np.mean(a):>13.3f} {np.mean(b):>12.3f} "
              f"{pct:>+8.1f}%  {p:.5f}{sig}  {d:+.3f}")

    print(f"\n  Policy Variance (reward SD):"
          f"  Equity D3QN={eq_sd:.2f}  Prod. Only={pr_sd:.2f}"
          f"  Reduction={(pr_sd-eq_sd)/pr_sd*100:.1f}%")

    print(f"\n  NOTE: Composite reward not compared — reward functions differ")
    print(f"  (equity agent subtracts fatigue/bias penalties; baseline does not).")
    print(f"  Individual metrics above are the correct comparison.\n")

    # Action distribution
    avg_dist = np.mean([r['action_dist'] for r in equity_res], axis=0)
    labels   = ['Idle', 'Assist', 'TakeOver', 'SuggestBreak']
    print(f"  Action distribution (Equity D3QN, final 30 eps):")
    for lbl, pct in zip(labels, avg_dist):
        print(f"    {lbl:<14}: {pct*100:.1f}%")
    augmentation = (avg_dist[1] + avg_dist[3]) * 100
    print(f"  Augmentation-focused (Assist+SuggestBreak): {augmentation:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: D3QN vs PPO Benchmark")
    print(f"  Same equity reward function | {N_SEEDS} seeds each")
    print("=" * 65)

    ppo_res = []
    for seed in range(N_SEEDS):
        print(f"  Training PPO seed {seed+1}/{N_SEEDS}...", end='\r')
        ppo_res.append(run_ppo(seed, EQUITY_W))
    print()

    ppo_t  = [r['final_throughput'] for r in ppo_res]
    ppo_e  = [r['final_error']      for r in ppo_res]
    ppo_f  = [r['final_fatigue']    for r in ppo_res]
    ppo_sd = np.mean([r['final_reward_sd'] for r in ppo_res])

    t2_t, p2_t = stats.ttest_ind(eq_t, ppo_t)
    t2_e, p2_e = stats.ttest_ind(eq_e, ppo_e)
    t2_f, p2_f = stats.ttest_ind(eq_f, ppo_f)

    print(f"\n{'Metric':<26} {'D3QN':>10} {'PPO':>10} {'Better':>8}  {'Change':>9}  p-value")
    print("-" * 76)

    bench_metrics = [
        ('Normalized Throughput', eq_t, ppo_t, False, p2_t),
        ('Error Rate',            eq_e, ppo_e, True,  p2_e),
        ('Worker Fatigue Index',  eq_f, ppo_f, True,  p2_f),
    ]
    for name, a, b, lower_better, p in bench_metrics:
        a_m, b_m = np.mean(a), np.mean(b)
        pct      = (a_m - b_m) / (b_m + 1e-9) * 100
        winner   = 'D3QN' if (lower_better and a_m < b_m) \
                           or (not lower_better and a_m > b_m) else 'PPO'
        sig      = '*' if p < 0.05 else ' '
        print(f"  {name:<24} {a_m:>10.3f} {b_m:>10.3f} {winner:>8}  "
              f"{pct:>+8.1f}%  {p:.5f}{sig}")

    print(f"  {'Policy Variance (SD)':<24} {eq_sd:>10.3f} {ppo_sd:>10.3f} "
          f"{'D3QN' if eq_sd < ppo_sd else 'PPO':>8}  "
          f"{(eq_sd-ppo_sd)/(ppo_sd+1e-9)*100:>+8.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: Robustness Testing (5 seeds per condition)")
    print("=" * 65)

    baseline_t = np.mean(eq_t)
    baseline_e = np.mean(eq_e)
    baseline_f = np.mean(eq_f)

    robustness_configs = [
        ('Clean baseline',       0.00, (0.05, 0.05)),
        ('Sensor noise 10%',     0.10, (0.05, 0.05)),
        ('Sensor noise 20%',     0.20, (0.05, 0.05)),
        ('Downtime 5-20%',       0.00, (0.05, 0.20)),
        ('Compound (noise+DT)',  0.10, (0.05, 0.20)),
    ]

    print(f"\n  {'Condition':<24} {'Throughput':>12} {'Error':>8} {'Fatigue':>9} "
          f"{'Err.Degrad':>12} {'Status':>8}")
    print("  " + "-" * 77)

    for name, noise, dt in robustness_configs:
        res = [run_d3qn(s, EQUITY_W, noise_std=noise,
                        downtime_range=dt) for s in range(5)]
        t = float(np.mean([r['final_throughput'] for r in res]))
        e = float(np.mean([r['final_error']      for r in res]))
        f = float(np.mean([r['final_fatigue']    for r in res]))
        # Measure degradation on error reduction (key welfare metric)
        err_degrad = (e - baseline_e) / (baseline_e + 1e-9) * 100
        status     = 'PASS' if abs(err_degrad) < 15 else 'FAIL'
        print(f"  {name:<24} {t:>12.3f} {e:>8.3f} {f:>9.3f} "
              f"{err_degrad:>+11.1f}%  {status:>8}")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SENSITIVITY ANALYSIS: Weight Configurations")
    print("=" * 65)

    scenarios = [
        ('Baseline (0.5,0.3,0.1,0.1)',      (0.5, 0.3, 0.1, 0.1)),
        ('Safety-First (0.3,0.3,0.5,0.1)',  (0.3, 0.3, 0.5, 0.1)),
        ('Production (0.8,0.15,0.05,0.0)',  (0.8, 0.15,0.05,0.0)),
    ]
    print(f"\n  {'Scenario':<38} {'Throughput':>12} {'Fatigue':>10} {'Error':>8}")
    print("  " + "-" * 70)
    for name, w in scenarios:
        res = [run_d3qn(s, w) for s in range(5)]
        t   = np.mean([r['final_throughput'] for r in res])
        f   = np.mean([r['final_fatigue']    for r in res])
        e   = np.mean([r['final_error']      for r in res])
        print(f"  {name:<38} {t:>12.3f} {f:>10.3f} {e:>8.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SKILL-LEVEL GENERALISATION")
    print("=" * 65)
    print(f"\n  {'Skill Level':<20} {'Throughput':>12} {'Fatigue':>10} {'Error':>8}")
    print("  " + "-" * 52)
    for skill_id, skill_name in enumerate(['Junior', 'Intermediate', 'Senior']):
        res = [run_d3qn(s, EQUITY_W, skill=skill_id) for s in range(5)]
        t   = np.mean([r['final_throughput'] for r in res])
        f   = np.mean([r['final_fatigue']    for r in res])
        e   = np.mean([r['final_error']      for r in res])
        print(f"  {skill_name:<20} {t:>12.3f} {f:>10.3f} {e:>8.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURES
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Equity-Aware D3QN: Experimental Results (v2)',
                 fontsize=13, fontweight='bold')

    # Fig 1: Training convergence — fatigue index
    ax = axes[0, 0]
    eq_mean_f  = np.mean([r['fatigues'] for r in equity_res], axis=0)
    pr_mean_f  = np.mean([r['fatigues'] for r in prod_res],   axis=0)
    ppo_mean_f = np.mean([r['fatigues'] for r in ppo_res],    axis=0)
    eq_std_f   = np.std( [r['fatigues'] for r in equity_res], axis=0)
    eps        = np.arange(len(eq_mean_f))
    ax.plot(eq_mean_f,  color='#2E86AB', lw=2, label='Equity D3QN')
    ax.fill_between(eps, eq_mean_f - eq_std_f, eq_mean_f + eq_std_f,
                    color='#2E86AB', alpha=0.2)
    ax.plot(pr_mean_f,  color='#E84855', lw=2, linestyle='--', label='Prod. Only')
    ax.plot(ppo_mean_f, color='#F18F01', lw=2, linestyle=':',  label='PPO')
    ax.set_title('Worker Fatigue Index over Training (mean ± SD, 20 seeds)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Fatigue Index')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Fig 2: Error rate over training
    ax = axes[0, 1]
    eq_mean_e  = np.mean([r['errors'] for r in equity_res], axis=0)
    pr_mean_e  = np.mean([r['errors'] for r in prod_res],   axis=0)
    ppo_mean_e = np.mean([r['errors'] for r in ppo_res],    axis=0)
    ax.plot(eq_mean_e,  color='#2E86AB', lw=2, label='Equity D3QN')
    ax.plot(pr_mean_e,  color='#E84855', lw=2, linestyle='--', label='Prod. Only')
    ax.plot(ppo_mean_e, color='#F18F01', lw=2, linestyle=':',  label='PPO')
    ax.set_title('Error Rate over Training (mean, 20 seeds)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Error Rate')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Fig 3: Final metric comparison bar chart
    ax = axes[1, 0]
    metric_names  = ['Throughput', 'Error Rate', 'Fatigue Index']
    eq_vals  = [np.mean(eq_t), np.mean(eq_e), np.mean(eq_f)]
    pr_vals  = [np.mean(pr_t), np.mean(pr_e), np.mean(pr_f)]
    ppo_vals = [np.mean(ppo_t), np.mean(ppo_e), np.mean(ppo_f)]
    x     = np.arange(len(metric_names))
    width = 0.28
    ax.bar(x - width,   eq_vals,  width, label='Equity D3QN', color='#2E86AB', alpha=0.85)
    ax.bar(x,           pr_vals,  width, label='Prod. Only',   color='#E84855', alpha=0.85)
    ax.bar(x + width,   ppo_vals, width, label='PPO',          color='#F18F01', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_title('Final Metric Comparison (last 30 eps, 20 seeds)\n'
                 'Lower is better for Error and Fatigue')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Fig 4: Action distribution
    ax = axes[1, 1]
    action_labels = ['Idle', 'Assist', 'TakeOver', 'SuggestBreak']
    eq_ad  = np.mean([r['action_dist'] for r in equity_res], axis=0)
    x      = np.arange(len(action_labels))
    colors = ['#5B9BD5', '#70AD47', '#ED7D31', '#FFC000']
    bars   = ax.bar(x, eq_ad * 100, color=colors, alpha=0.85)
    for bar, pct in zip(bars, eq_ad):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{pct*100:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(action_labels)
    ax.set_title('Action Distribution — Equity D3QN\n(final 30 episodes, 20 seeds)')
    ax.set_ylabel('Frequency (%)'); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('hrc_d3qn_results_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved: hrc_d3qn_results_v2.png")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MANUSCRIPT TABLE NUMBERS — copy these directly")
    print("=" * 65)
    print(f"\nTable 3 (KPIs — Equity D3QN):")
    print(f"  Normalized Throughput : {np.mean(eq_t):.3f}")
    print(f"  Error Rate            : {np.mean(eq_e)*100:.2f}%")
    print(f"  Worker Fatigue Index  : {np.mean(eq_f):.3f}")
    print(f"  Policy Variance (SD)  : {eq_sd:.2f}")

    print(f"\nTable 4 (Equity D3QN vs Productivity-Only):")
    print(f"  Throughput change : {(np.mean(eq_t)-np.mean(pr_t))/np.mean(pr_t)*100:+.1f}%"
          f"  (p={p_t:.4f}, d={d_t:+.3f})")
    print(f"  Error reduction   : {(np.mean(pr_e)-np.mean(eq_e))/np.mean(pr_e)*100:.1f}%"
          f"  (p={p_e:.5f}, d={d_e:+.3f})")
    print(f"  Fatigue reduction : {(np.mean(pr_f)-np.mean(eq_f))/np.mean(pr_f)*100:.1f}%"
          f"  (p={p_f:.5f}, d={d_f:+.3f})")
    print(f"  SD reduction      : {(pr_sd-eq_sd)/pr_sd*100:.1f}%")

    print(f"\nTable 5 (D3QN vs PPO):")
    print(f"  Throughput  — D3QN: {np.mean(eq_t):.3f}  PPO: {np.mean(ppo_t):.3f}"
          f"  ({(np.mean(eq_t)-np.mean(ppo_t))/np.mean(ppo_t)*100:+.1f}%,  p={p2_t:.4f})")
    print(f"  Error Rate  — D3QN: {np.mean(eq_e):.3f}  PPO: {np.mean(ppo_e):.3f}"
          f"  ({(np.mean(eq_e)-np.mean(ppo_e))/np.mean(ppo_e)*100:+.1f}%,  p={p2_e:.4f})")
    print(f"  Fatigue     — D3QN: {np.mean(eq_f):.3f}  PPO: {np.mean(ppo_f):.3f}"
          f"  ({(np.mean(eq_f)-np.mean(ppo_f))/np.mean(ppo_f)*100:+.1f}%,  p={p2_f:.4f})")
    print(f"  Policy SD   — D3QN: {eq_sd:.2f}   PPO: {ppo_sd:.2f}"
          f"  ({(eq_sd-ppo_sd)/ppo_sd*100:+.1f}%)")
    print("=" * 65)
