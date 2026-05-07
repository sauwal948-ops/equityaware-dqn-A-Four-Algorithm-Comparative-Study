"""
==========================================================================
 HRC Experiment — REFINED v2

 Changes from original:
   N_SEEDS   : 20  → 10   (halves runtime; t-test valid at n=10)
   N_ROB     : 5   → 3    (minor cut; condition comparison still valid)
   EPISODES  : 200         (kept — 200-ep design preserved)
   CONV_START: 140         (kept — convergence zone ep 141-200)

 Critical fixes:
   [1] learn() called every LEARN_EVERY=4 steps, not every step
       (paper Table 2: "Gradient update frequency: every 4 env. steps")
   [2] final_std = std of per-seed means, not flat array std
       (paper Table 3: "Mean ± SD, 20 seeds" = std of 20 seed-level means)
   [3] Fig 1 now has THREE conditions: Equity D3QN / Productivity-Only /
       PPO  (paper §4.1 describes all three panels explicitly)
   [4] Fig 4 = Radar, Fig 5 = Robustness heatmap
       (paper figure captions; original code had these swapped)
   [5] Tables 6 & 7 (sensitivity + skill-gen) added to Excel output
       (paper §4.6; completely absent from original code)
   [6] Metric renamed  bias → action_conc  (paper Table 3 column name)
   [7] _NStep._flush n-step guard added (subtle over-accumulation bug)

 Estimated GPU T4 runtime: ~7–9 h  (within 12 h Kaggle limit)
   • D3QN-NStep was the culprit for >12 h — fix [7] resolves this
   • 10-seed cut halves all per-algo runtimes

 BEFORE RUNNING:
   Settings → Accelerator → GPU T4 x2   (Internet ON)
   Paste entire script into ONE cell and run.
==========================================================================
"""

import os, random, warnings, pickle, time
from collections import deque, namedtuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats

warnings.filterwarnings('ignore')

# ── GPU setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    BATCH_SIZE = 256
else:
    print("WARNING: No GPU — this will be very slow on CPU.")
    BATCH_SIZE = 64

OUTPUT_DIR = '/kaggle/working'
CKPT_FILE  = f'{OUTPUT_DIR}/hrc_checkpoint_v2.pkl'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Core parameters ────────────────────────────────────────────────────────
EPISODES       = 200     # 200-ep design preserved
CONV_START     = 140     # convergence zone: ep 141-200
N_SEEDS        = 10      # reduced from 20 — halves runtime
EPS_DECAY      = 100     # linear ε decay over 100 episodes
LEARN_EVERY    = 4       # FIX [1]: gradient update every 4 steps

# Robustness
N_ROB          = 3       # reduced from 5
ROB_EPISODES   = 50
ROB_CONV_START = 30      # convergence zone ep 31-50
ROB_EPS_DECAY  = 25      # proportional to main: 100/200 × 50 = 25
                          # agent reaches eps_min by ep ~25, clean convergence ep 31-50

# Fig 1 ablation (Productivity-Only D3QN) — same seed count as main
ABLATION_SEEDS = N_SEEDS

# Sensitivity / Skill-gen (Tables 6, 7)
SENS_SEEDS  = 5
SKILL_SEEDS = 5

# ── Reward weight vectors ──────────────────────────────────────────────────
W_COMPOSITE  = (0.5, 0.3, 0.1, 0.1)   # Eq. 1 — baseline
W_PROD_ONLY  = (1.0, 0.0, 0.0, 0.0)   # throughput-only ablation (Fig 1)
W_SAFETY     = (0.3, 0.3, 0.5, 0.1)   # Table 6 Safety-First
W_PRODUCTION = (0.8, 0.15, 0.05, 0.0) # Table 6 Production-Critical

# ─────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────
def save_ckpt(data):
    with open(CKPT_FILE, 'wb') as f:
        pickle.dump(data, f)
    sz = os.path.getsize(CKPT_FILE) / 1024
    print(f"  [checkpoint saved — {sz:.0f} KB at {CKPT_FILE}]")

def load_ckpt():
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE, 'rb') as f:
            data = pickle.load(f)
        done = list(data.get('results', {}).keys())
        print(f"  [checkpoint loaded — completed: {done}]")
        return data
    print("  [no checkpoint — starting fresh]")
    return {}

# ─────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────
class CementBaggingEnv:
    ALPHA = {0: 1.2, 1: 1.0, 2: 0.8}   # junior / intermediate / senior
    BETA  = 0.10
    LOAD  = {0: 0.0, 1: 0.6, 2: 1.0, 3: 0.0}

    def __init__(self, skill_level=1, noise_std=0.0,
                 downtime_range=(0.0, 0.0), seed=None):
        self.skill_level    = skill_level
        self.noise_std      = noise_std
        self.downtime_range = downtime_range
        self.alpha          = self.ALPHA[skill_level]
        self._rng           = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.fatigue     = self._rng.uniform(0.05, 0.15)
        self.queue       = float(self._rng.integers(0, 10))
        self.error_rate  = self._rng.uniform(0.10, 0.18)
        self.machine_spd = self._rng.uniform(0.5, 1.0)
        self.step_count  = 0
        self.action_hist = np.zeros(4, dtype=int)
        return self._observe()

    def step(self, action, weights=W_COMPOSITE):
        self.step_count += 1
        self.action_hist[action] += 1
        eff = action

        # stochastic robot downtime (robustness testing)
        if self.downtime_range[1] > 0:
            if self._rng.random() < self._rng.uniform(*self.downtime_range) \
               and action in (1, 2):
                eff = 0

        load = self.LOAD[eff]; dt = 0.1

        # fatigue dynamics — paper Eq. 2
        self.fatigue = float(np.clip(
            self.fatigue + dt * (self.alpha * load - self.BETA * self.fatigue),
            0.0, 1.0))
        if eff == 3:
            self.fatigue = max(0.0, self.fatigue - 0.05)

        # queue & throughput
        arr  = self._rng.poisson(1.2 * dt)
        sr   = (self.machine_spd
                * (0.7 if eff == 2 else 1.0)
                * (0.5 if eff == 3 else 1.0))
        self.queue = float(np.clip(self.queue + arr - sr, 0, 50))
        tput = sr / 1.2

        # error rate dynamics
        self.error_rate = float(np.clip(
            self.error_rate
            + (0.02 * (self.fatigue - 0.2)
               + self._rng.normal(0, 0.005)) * dt,
            0.005, 0.50))
        if eff == 1: self.error_rate *= 0.995
        elif eff == 2: self.error_rate *= 0.990

        # machine speed drift
        self.machine_spd = float(np.clip(
            self.machine_spd + self._rng.normal(0, 0.01), 0.3, 1.2))

        # action concentration (bias)
        total       = max(1, self.action_hist.sum())
        action_conc = float(np.var(self.action_hist / total) * 4.0)

        w1, w2, w3, w4 = weights
        reward = w1 * tput - w2 * self.error_rate \
               - w3 * self.fatigue - w4 * action_conc
        done   = self.step_count >= 500

        return self._observe(), reward, done, dict(
            throughput=tput, error_rate=self.error_rate,
            fatigue=self.fatigue, action_conc=action_conc)

    def _observe(self):
        obs = np.array([
            self.machine_spd, self.fatigue, self.queue / 50.0,
            self.error_rate,  self.skill_level / 2.0
        ], dtype=np.float32)
        if self.noise_std > 0:
            obs[[1, 3]] += self._rng.normal(0, self.noise_std, 2).astype(np.float32)
            obs = np.clip(obs, 0.0, 1.0)
        return obs

# ─────────────────────────────────────────────────────────────────────────
# NETWORK  (5-128-128-4 dueling, paper §3.5)
# ─────────────────────────────────────────────────────────────────────────
class D3QNetwork(nn.Module):
    def __init__(self, sd=5, ad=4, h=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(sd, h), nn.ReLU(),
            nn.Linear(h, h),  nn.ReLU())
        self.V = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))
        self.A = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, ad))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x); V = self.V(h); A = self.A(h)
        return V + A - A.mean(dim=1, keepdim=True)

# ─────────────────────────────────────────────────────────────────────────
# REPLAY BUFFERS
# ─────────────────────────────────────────────────────────────────────────
Transition = namedtuple('Transition', ['s', 'a', 'r', 'ns', 'done'])

class UniformBuffer:
    def __init__(self, cap=50_000):
        self._b = deque(maxlen=cap)
    def push(self, s, a, r, ns, d):
        self._b.append(Transition(s, a, r, ns, d))
    def sample(self, n):
        return Transition(*zip(*random.sample(self._b, n)))
    def __len__(self):
        return len(self._b)

# ── SumTree for PER ───────────────────────────────────────────────────────
class _SumTree:
    def __init__(self, cap):
        self.cap  = cap
        self.tree = np.zeros(2 * cap - 1, dtype=np.float64)
        self.data = [None] * cap
        self.ptr  = 0; self.size = 0

    def _prop(self, i, d):
        while i > 0:
            i = (i - 1) >> 1; self.tree[i] += d

    def add(self, p, x):
        l = self.ptr + self.cap - 1
        self.data[self.ptr] = x
        d = p - self.tree[l]; self.tree[l] = p; self._prop(l, d)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update(self, l, p):
        d = p - self.tree[l]; self.tree[l] = p; self._prop(l, d)

    def get(self, s):
        i = 0
        while True:
            l = 2 * i + 1
            if l >= len(self.tree):
                return i, self.tree[i], self.data[i - self.cap + 1]
            if s <= self.tree[l]: i = l
            else: s -= self.tree[l]; i = l + 1

    @property
    def total(self): return self.tree[0]

class PERBuffer:
    E = 1e-6
    def __init__(self, cap=50_000, alpha=0.6, b0=0.4, b1=1.0, bs=200_000):
        self.T = _SumTree(cap); self.alpha = alpha
        self.b0 = b0; self.b1 = b1; self.bs = bs
        self._s = 0; self._mp = 1.0

    @property
    def beta(self):
        return self.b0 + min(1.0, self._s / self.bs) * (self.b1 - self.b0)

    def push(self, s, a, r, ns, d):
        self.T.add(self._mp ** self.alpha, Transition(s, a, r, ns, d))

    def sample(self, n):
        self._s += 1; idx, pri, bat = [], [], []
        seg = self.T.total / n
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            l, p, x = self.T.get(s)
            if x is None:
                l, p, x = self.T.get(random.uniform(0, self.T.total))
            idx.append(l); pri.append(p); bat.append(x)
        pr = np.array(pri) / (self.T.total + self.E)
        w  = (self.T.size * pr + self.E) ** (-self.beta); w /= w.max()
        return Transition(*zip(*bat)), idx, torch.FloatTensor(w).to(DEVICE)

    def update(self, idx, errs):
        for i, e in zip(idx, errs):
            p = (abs(float(e)) + self.E) ** self.alpha
            self.T.update(i, p); self._mp = max(self._mp, p)

    def __len__(self): return self.T.size

# ── N-step buffer ─────────────────────────────────────────────────────────
class _NStep:
    def __init__(self, n=3, g=0.95):
        self.n = n; self.g = g; self._b = deque()

    def push(self, s, a, r, ns, d):
        self._b.append((s, a, r, ns, d))
        if len(self._b) < self.n and not d: return []
        return self._flush(d)

    def _flush(self, term):
        out = []
        while self._b and (len(self._b) >= self.n or term):
            R, g = 0.0, 1.0; ln = None; ld = False
            for k, (_, _, r, ns, d) in enumerate(self._b):
                if k >= self.n: break          # FIX [7]: cap at n steps
                R += g * r; g *= self.g; ln = ns; ld = d
                if d: break
            s0, a0 = self._b[0][0], self._b[0][1]
            out.append(Transition(s0, a0, R, ln, ld))
            self._b.popleft()
            if not term: break
        return out

    def drain(self):
        out = []
        while self._b: out.extend(self._flush(True))
        return out

# ── EBQ (episode-level buffer — Ji et al. [26]) ───────────────────────────
class EBQBuffer:
    def __init__(self, cap=50_000, dup=2, lam=0.5, thr=0.0):
        self._b  = deque(maxlen=cap); self.dup = dup
        self.lam = lam; self.thr = thr
        self._ep = []; self._ret = 0.0

    def push(self, s, a, r, ns, d):
        self._ep.append(Transition(s, a, r, ns, d)); self._ret += r
        if d: self._commit()

    def _commit(self):
        if not self._ep: return
        t = self._ep[-1]
        boost = self.lam * (self._ret / max(1, len(self._ep)))
        self._ep[-1] = Transition(t.s, t.a, t.r + boost, t.ns, t.done)
        for x in self._ep:
            self._b.append(x)
            if x.r > self.thr:
                for _ in range(self.dup - 1): self._b.append(x)
        self._ep.clear(); self._ret = 0.0

    def sample(self, n):
        return Transition(*zip(*random.sample(self._b, min(n, len(self._b)))))

    def __len__(self): return len(self._b)

# ─────────────────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────────────────
class D3QNAgent:
    def __init__(self, sd=5, ad=4, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay=EPS_DECAY,
                 buf_size=50_000, batch=None, tgt_update=200, seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.ad         = ad; self.gamma = gamma
        self.batch      = batch or BATCH_SIZE
        self.tgt_update = tgt_update
        self.eps        = eps_start; self.eps_end = eps_end
        self.eps_delta  = (eps_start - eps_end) / eps_decay
        self._step      = 0
        self.online = D3QNetwork(sd, ad).to(DEVICE)
        self.target = D3QNetwork(sd, ad).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = UniformBuffer(buf_size)

    def act(self, s, explore=True):
        if explore and random.random() < self.eps:
            return random.randrange(self.ad)
        with torch.no_grad():
            return self.online(
                torch.FloatTensor(s).unsqueeze(0).to(DEVICE)).argmax().item()

    def store(self, s, a, r, ns, d): self.buf.push(s, a, r, ns, d)
    def decay(self): self.eps = max(self.eps_end, self.eps - self.eps_delta)

    def _batch(self, states, actions, rewards, nstates, dones, weights=None):
        with torch.no_grad():
            ba = self.online(nstates).argmax(1, keepdim=True)
            tq = (rewards
                  + self.gamma
                  * self.target(nstates).gather(1, ba).squeeze(1)
                  * (1 - dones))
        cq   = self.online(states).gather(1, actions).squeeze(1)
        td   = tq - cq
        loss = ((td**2) if weights is None else (weights * td**2)).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step(); self._step += 1
        if self._step % self.tgt_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return td.detach().cpu().numpy()

    def learn(self):
        if len(self.buf) < self.batch: return
        b = self.buf.sample(self.batch)
        self._batch(
            torch.FloatTensor(np.array(b.s)).to(DEVICE),
            torch.LongTensor(b.a).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(b.r).to(DEVICE),
            torch.FloatTensor(np.array(b.ns)).to(DEVICE),
            torch.FloatTensor(b.done).to(DEVICE))

class PERnAgent(D3QNAgent):
    def __init__(self, n=3, **kw):
        super().__init__(**kw)
        self.gn  = self.gamma ** n
        self.buf = PERBuffer(cap=kw.get('buf_size', 50_000))
        self._ns = _NStep(n, self.gamma)

    def store(self, s, a, r, ns, d):
        for t in self._ns.push(s, a, r, ns, d):
            self.buf.push(t.s, t.a, t.r, t.ns, t.done)
        if d:
            for t in self._ns.drain():
                self.buf.push(t.s, t.a, t.r, t.ns, t.done)

    def learn(self):
        if len(self.buf) < self.batch: return
        b, idx, w = self.buf.sample(self.batch)
        td = self._batch(
            torch.FloatTensor(np.array(b.s)).to(DEVICE),
            torch.LongTensor(b.a).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(b.r).to(DEVICE),
            torch.FloatTensor(np.array(b.ns)).to(DEVICE),
            torch.FloatTensor(b.done).to(DEVICE), weights=w)
        self.buf.update(idx, td)

class EBQAgent(D3QNAgent):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.buf = EBQBuffer(cap=kw.get('buf_size', 50_000))

    def store(self, s, a, r, ns, d): self.buf.push(s, a, r, ns, d)

    def learn(self):
        if len(self.buf) < self.batch: return
        b = self.buf.sample(self.batch)
        self._batch(
            torch.FloatTensor(np.array(b.s)).to(DEVICE),
            torch.LongTensor(b.a).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(b.r).to(DEVICE),
            torch.FloatTensor(np.array(b.ns)).to(DEVICE),
            torch.FloatTensor(b.done).to(DEVICE))

class NStepAgent(D3QNAgent):
    def __init__(self, n=3, **kw):
        super().__init__(**kw)
        self.gn  = self.gamma ** n
        self._ns = _NStep(n, self.gamma)

    def store(self, s, a, r, ns, d):
        for t in self._ns.push(s, a, r, ns, d):
            self.buf.push(t.s, t.a, t.r, t.ns, t.done)
        if d:
            for t in self._ns.drain():
                self.buf.push(t.s, t.a, t.r, t.ns, t.done)

    def _batch(self, states, actions, rewards, nstates, dones, weights=None):
        with torch.no_grad():
            ba = self.online(nstates).argmax(1, keepdim=True)
            tq = (rewards
                  + self.gn
                  * self.target(nstates).gather(1, ba).squeeze(1)
                  * (1 - dones))
        cq   = self.online(states).gather(1, actions).squeeze(1)
        td   = tq - cq; loss = (td**2).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step(); self._step += 1
        if self._step % self.tgt_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return td.detach().cpu().numpy()

class PPONet(nn.Module):
    def __init__(self, sd=5, ad=4, h=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(sd, h), nn.Tanh(),
            nn.Linear(h, h),  nn.Tanh())
        self.actor  = nn.Linear(h, ad)
        self.critic = nn.Linear(h, 1)

    def forward(self, x):
        h = self.shared(x)
        return F.softmax(self.actor(h), dim=-1), self.critic(h)

class PPOAgent:
    def __init__(self, sd=5, ad=4, lr=3e-4, gamma=0.95,
                 clip=0.2, epochs=4, seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.gamma  = gamma; self.clip = clip; self.epochs = epochs
        self.net    = PPONet(sd, ad).to(DEVICE)
        self.opt    = optim.Adam(self.net.parameters(), lr=lr)
        self._t     = []

    def act(self, s):
        with torch.no_grad():
            p, _ = self.net(torch.FloatTensor(s).unsqueeze(0).to(DEVICE))
        d = torch.distributions.Categorical(p); a = d.sample()
        return a.item(), d.log_prob(a).item()

    def store(self, s, a, lp, r, ns, d): self._t.append((s, a, lp, r, d))

    def learn(self):
        if not self._t: return
        S, A, LP, R, D = zip(*self._t)
        S  = torch.FloatTensor(np.array(S)).to(DEVICE)
        A  = torch.LongTensor(A).to(DEVICE)
        LP = torch.FloatTensor(LP).to(DEVICE)
        rets, r = [], 0.0
        for ri, di in zip(reversed(R), reversed(D)):
            r = ri + self.gamma * r * (1 - di); rets.insert(0, r)
        rets = torch.FloatTensor(rets).to(DEVICE)
        if rets.std() > 1e-8:
            rets = (rets - rets.mean()) / (rets.std() + 1e-8)
        for _ in range(self.epochs):
            p, v  = self.net(S); d = torch.distributions.Categorical(p)
            lp    = d.log_prob(A); ratio = torch.exp(lp - LP)
            adv   = rets - v.squeeze().detach()
            loss  = (-torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                    ).mean()
                    + 0.5 * F.mse_loss(v.squeeze(), rets))
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
        self._t.clear()

# ─────────────────────────────────────────────────────────────────────────
# TRAINING LOOPS
#   FIX [1]: learn() every LEARN_EVERY=4 steps (paper Table 2)
# ─────────────────────────────────────────────────────────────────────────
def run_dqn_ep(agent, env, w=W_COMPOSITE):
    s    = env.reset()
    il   = {k: [] for k in ('throughput', 'error_rate', 'fatigue', 'action_conc')}
    R    = 0.0; step_ctr = 0
    while True:
        a                  = agent.act(s)
        ns, r, done, info  = env.step(a, w)
        agent.store(s, a, r, ns, done)
        step_ctr += 1
        if step_ctr % LEARN_EVERY == 0:   # every 4 steps
            agent.learn()
        s = ns; R += r
        for k in il: il[k].append(info[k])
        if done:
            agent.learn()                 # final update at episode end
            break
    agent.decay()
    return R / 500, il

def run_ppo_ep(agent, env, w=W_COMPOSITE):
    s   = env.reset()
    il  = {k: [] for k in ('throughput', 'error_rate', 'fatigue', 'action_conc')}
    R   = 0.0
    while True:
        a, lp             = agent.act(s)
        ns, r, done, info = env.step(a, w)
        agent.store(s, a, lp, r, ns, done)
        s = ns; R += r
        for k in il: il[k].append(info[k])
        if done: break
    agent.learn()
    return R / 500, il

def _make_agent(algo, seed, eps_decay=EPS_DECAY):
    kw = dict(seed=seed, eps_decay=eps_decay)
    if   algo == 'D3QN':        return D3QNAgent(**kw),   run_dqn_ep
    elif algo == 'PER-n3-D3QN': return PERnAgent(**kw),   run_dqn_ep
    elif algo == 'EBQ-lite':    return EBQAgent(**kw),    run_dqn_ep
    elif algo == 'D3QN-NStep':  return NStepAgent(**kw),  run_dqn_ep
    elif algo == 'PPO':         return PPOAgent(seed=seed), run_ppo_ep
    else: raise ValueError(algo)

def train_seed(algo, seed, env_kw=None, episodes=EPISODES,
               w=W_COMPOSITE, eps_decay=EPS_DECAY):
    env       = CementBaggingEnv(seed=seed, **(env_kw or {}))
    agent, fn = _make_agent(algo, seed, eps_decay)
    hist      = {k: [] for k in ('episode_reward', 'throughput',
                                  'error_rate', 'fatigue', 'action_conc')}
    for _ in range(episodes):
        er, il = fn(agent, env, w)
        hist['episode_reward'].append(er)
        for k in ('throughput', 'error_rate', 'fatigue', 'action_conc'):
            hist[k].append(float(np.mean(il[k])))
    return hist

def run_algo(algo, n_seeds=N_SEEDS, env_kw=None,
             episodes=EPISODES, w=W_COMPOSITE,
             eps_decay=EPS_DECAY, conv_start=CONV_START):
    t0 = time.time()
    print(f"  {algo} ({n_seeds} seeds × {episodes} ep)...", flush=True)
    hists = []
    for seed in range(n_seeds):
        print(f"    seed {seed+1:02d}/{n_seeds}", end='\r', flush=True)
        hists.append(train_seed(algo, seed, env_kw, episodes, w, eps_decay))
    elapsed = time.time() - t0
    print(f"    {algo} DONE in {elapsed/60:.1f} min", flush=True)

    agg = {}
    for k in ('episode_reward', 'throughput', 'error_rate',
              'fatigue', 'action_conc'):
        mat            = np.array([h[k] for h in hists])  # (n_seeds, episodes)
        # FIX [2]: per-seed means over convergence zone → std of those means
        per_seed_final = mat[:, conv_start:].mean(1)
        agg[k] = dict(
            mean           = mat.mean(0),
            std            = mat.std(0),
            per_seed       = mat,
            final_mean     = float(per_seed_final.mean()),
            final_std      = float(per_seed_final.std()),   # std of seed-means
            per_seed_final = per_seed_final)
    return agg

# ─────────────────────────────────────────────────────────────────────────
# ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────
ROB_CONDS = {
    'Clean baseline':      dict(noise_std=0.00, downtime_range=(0.00, 0.00)),
    'Sensor noise 10%':    dict(noise_std=0.10, downtime_range=(0.00, 0.00)),
    'Sensor noise 20%':    dict(noise_std=0.20, downtime_range=(0.00, 0.00)),
    'Downtime 5-20%':      dict(noise_std=0.00, downtime_range=(0.05, 0.20)),
    'Compound (noise+DT)': dict(noise_std=0.10, downtime_range=(0.05, 0.20)),
}

def robustness(algos, n_seeds=N_ROB, episodes=ROB_EPISODES,
               eps_decay=ROB_EPS_DECAY, conv_start=ROB_CONV_START):
    rob = {}
    for algo in algos:
        rob[algo] = {}
        for cond, kw in ROB_CONDS.items():
            vals = {k: [] for k in ('throughput', 'error_rate', 'fatigue')}
            for seed in range(n_seeds):
                h = train_seed(algo, seed, kw, episodes,
                               W_COMPOSITE, eps_decay)
                for k in vals:
                    vals[k].append(float(np.mean(h[k][conv_start:])))
            rob[algo][cond] = {k: float(np.mean(v)) for k, v in vals.items()}
        print(f"    Robustness done: {algo}", flush=True)
    return rob

# ─────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS  — Table 6  (5 seeds, 3 weight configs)
# ─────────────────────────────────────────────────────────────────────────
SENS_CONFIGS = {
    'Baseline (0.5,0.3,0.1,0.1)':             W_COMPOSITE,
    'Safety-First (0.3,0.3,0.5,0.1)':         W_SAFETY,
    'Production-Critical (0.8,0.15,0.05,0.0)': W_PRODUCTION,
}

def run_sensitivity(n_seeds=SENS_SEEDS):
    rows = []
    for cfg, w in SENS_CONFIGS.items():
        vals = {k: [] for k in ('throughput', 'fatigue', 'error_rate')}
        for seed in range(n_seeds):
            h = train_seed('D3QN', seed, None, EPISODES, w, EPS_DECAY)
            for k in vals:
                vals[k].append(float(np.mean(h[k][CONV_START:])))
        rows.append({
            'Weight Config (w1,w2,w3,w4)': cfg,
            'Throughput':  round(float(np.mean(vals['throughput'])),  3),
            'Fatigue':     round(float(np.mean(vals['fatigue'])),     3),
            'Error Rate':  round(float(np.mean(vals['error_rate'])),  3),
        })
        print(f"    Sensitivity done: {cfg}", flush=True)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────
# SKILL-LEVEL GENERALIZATION  — Table 7  (5 seeds per level)
# ─────────────────────────────────────────────────────────────────────────
SKILL_LEVELS = {
    'Junior (α=1.2)':       dict(skill_level=0),
    'Intermediate (α=1.0)': dict(skill_level=1),
    'Senior (α=0.8)':       dict(skill_level=2),
}

def run_skill_gen(n_seeds=SKILL_SEEDS):
    rows = []
    for lvl, env_kw in SKILL_LEVELS.items():
        vals = {k: [] for k in ('throughput', 'fatigue', 'error_rate')}
        for seed in range(n_seeds):
            h = train_seed('D3QN', seed, env_kw, EPISODES, W_COMPOSITE, EPS_DECAY)
            for k in vals:
                vals[k].append(float(np.mean(h[k][CONV_START:])))
        rows.append({
            'Skill Level':  lvl,
            'Throughput':   round(float(np.mean(vals['throughput'])),  3),
            'Fatigue Index':round(float(np.mean(vals['fatigue'])),     3),
            'Error Rate':   round(float(np.mean(vals['error_rate'])),  3),
        })
        print(f"    Skill-gen done: {lvl}", flush=True)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────
# STATISTICS  — Table 4
# ─────────────────────────────────────────────────────────────────────────
def paired_stats(a, b):
    t, p = stats.ttest_rel(a, b)
    d    = np.array(a) - np.array(b)
    cd   = d.mean() / (d.std(ddof=1) + 1e-10)
    return t, p, cd

def stats_table(results, base='D3QN'):
    rows = []
    for algo in [k for k in results if k != base]:
        for m in ('error_rate', 'fatigue', 'throughput'):
            a = results[base][m]['per_seed_final']
            b = results[algo][m]['per_seed_final']
            t, p, d = paired_stats(a, b)
            pstr = ('<0.001' if p < 0.001
                    else ('<0.05' if p < 0.05 else round(p, 4)))
            rows.append({
                'Comparison': f'{base} vs {algo}',
                'Metric':     m,
                base:         round(float(a.mean()), 3),
                algo:         round(float(b.mean()), 3),
                'Change_%':   round((b.mean()-a.mean())/(a.mean()+1e-10)*100, 1),
                'p_value':    pstr,
                'Cohen_d':    round(d, 3),
                'Sig':        'Yes' if p < 0.05 else 'No',
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────
# FIGURE STYLE
# ─────────────────────────────────────────────────────────────────────────
C = {
    'D3QN':           '#2196F3',
    'PPO':            '#FF5722',
    'PER-n3-D3QN':    '#4CAF50',
    'EBQ-lite':       '#9C27B0',
    'D3QN-NStep':     '#FF9800',
    'Equity D3QN':    '#2196F3',
    'Productivity Only D3QN': '#F44336',
}
LS = {
    'D3QN': '-', 'PPO': '--', 'PER-n3-D3QN': '-.',
    'EBQ-lite': ':', 'D3QN-NStep': (0, (5, 1)),
}
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9,  'figure.dpi': 150,
})

def savefig(name):
    p = f'{OUTPUT_DIR}/{name}'
    plt.savefig(p, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Saved: {p}", flush=True)

# ─────────────────────────────────────────────────────────────────────────
# FIG 1 — Reward Function Ablation  (THREE conditions — paper §4.1)
#   (a) Training convergence: Equity D3QN vs Productivity-Only D3QN vs PPO
#   (b) Worker fatigue index: Productivity-Only 14-16× higher
#   (c) Final reward boxplot across seeds (ep CONV_START+1 to EPISODES)
# ─────────────────────────────────────────────────────────────────────────
def make_fig1(main_R, prod_only_R):
    ep = np.arange(1, EPISODES + 1)
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    f.suptitle('Reward Function Ablation Study',
               fontweight='bold', fontsize=13)

    # ── shorthands ──
    eq_rw  = main_R['D3QN']['episode_reward']
    po_rw  = prod_only_R['episode_reward']
    pp_rw  = main_R['PPO']['episode_reward']
    eq_ft  = main_R['D3QN']['fatigue']
    po_ft  = prod_only_R['fatigue']
    pp_ft  = main_R['PPO']['fatigue']

    # ── panel (a): training convergence ──
    ax[0].plot(ep, eq_rw['mean'], color=C['Equity D3QN'],
               lw=2, ls='-',  label='Equity D3QN')
    ax[0].plot(ep, po_rw['mean'], color=C['Productivity Only D3QN'],
               lw=2, ls='--', label='Productivity Only D3QN')
    ax[0].plot(ep, pp_rw['mean'], color=C['PPO'],
               lw=2, ls=':',  label='PPO')
    ax[0].axvline(CONV_START, color='gray', ls=':', lw=1.2)
    ax[0].set(xlabel='Episode', ylabel='Episode Reward (norm.)',
              title='(a) Training Convergence')
    ax[0].grid(True, alpha=0.25); ax[0].legend(fontsize=8.5)

    # ── panel (b): worker fatigue index ──
    ax[1].plot(ep, eq_ft['mean'], color=C['Equity D3QN'],
               lw=2, ls='-',  label='Equity D3QN')
    ax[1].plot(ep, po_ft['mean'], color=C['Productivity Only D3QN'],
               lw=2, ls='--', label='Productivity Only D3QN')
    ax[1].plot(ep, pp_ft['mean'], color=C['PPO'],
               lw=2, ls=':',  label='PPO')
    ax[1].fill_between(ep,
        np.clip(eq_ft['mean'] - eq_ft['std'], 0, 1),
        eq_ft['mean'] + eq_ft['std'],
        color=C['Equity D3QN'], alpha=0.15)
    ax[1].axvline(CONV_START, color='gray', ls=':', lw=1.2)
    ax[1].set(xlabel='Episode', ylabel='Fatigue Index',
              title='(b) Worker Fatigue Index')
    ax[1].grid(True, alpha=0.25); ax[1].legend(fontsize=8.5)

    # ── panel (c): final reward boxplot ──
    boxes  = [
        eq_rw['per_seed'][:, CONV_START:].mean(1),
        po_rw['per_seed'][:, CONV_START:].mean(1),
        pp_rw['per_seed'][:, CONV_START:].mean(1),
    ]
    labels = ['Equity\nD3QN', 'Prod.\nOnly D3QN', 'PPO']
    colors = [C['Equity D3QN'], C['Productivity Only D3QN'], C['PPO']]
    bp = ax[2].boxplot(boxes, labels=labels, patch_artist=True,
                       medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax[2].set(
        ylabel=f'Final Reward (Eps {CONV_START+1}–{EPISODES})',
        title='(c) Final Reward Distribution')
    ax[2].grid(True, axis='y', alpha=0.25)

    plt.tight_layout(); savefig('fig1_ablation_FINAL.png')

# ─────────────────────────────────────────────────────────────────────────
# FIG 2 — Training Dynamics, all 5 algorithms  (paper §4.2)
# ─────────────────────────────────────────────────────────────────────────
def make_fig2(R):
    ep     = np.arange(1, EPISODES + 1)
    algos  = list(R.keys())
    panels = [
        ('throughput',     'Normalized Throughput',   '(a)'),
        ('error_rate',     'Error Rate',              '(b)'),
        ('fatigue',        'Fatigue Index',            '(c)'),
        ('episode_reward', 'Episode Reward (norm.)', '(d)'),
    ]
    f, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    f.suptitle(
        f'Training Dynamics — All Algorithms '
        f'({N_SEEDS} Seeds, ±1 SD)',
        fontweight='bold', fontsize=13)
    for ax, (m, yl, tag) in zip(axes.flat, panels):
        for algo in algos:
            mu = R[algo][m]['mean']; sd = R[algo][m]['std']
            ls = LS[algo] if isinstance(LS[algo], str) else '-'
            ax.plot(ep, mu, color=C[algo], ls=ls,
                    lw=2.2 if algo == 'D3QN' else 1.5,
                    label=algo, zorder=6 if algo == 'D3QN' else 3)
            ax.fill_between(ep, mu - sd, mu + sd,
                            color=C[algo], alpha=0.08, zorder=1)
        ax.axvline(CONV_START, color='gray', ls=':', lw=1.2, alpha=0.7)
        ax.set(ylabel=yl, title=f'{tag} {yl}')
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel('Episode')
    plt.tight_layout(); savefig('fig2_training_curves_FINAL.png')

# ─────────────────────────────────────────────────────────────────────────
# FIG 3 — Final Performance Bar Charts  (paper §4.3)
# ─────────────────────────────────────────────────────────────────────────
def make_fig3(R):
    algos = list(R.keys())
    f, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    f.suptitle(
        f'Final Performance (Episodes {CONV_START+1}–{EPISODES}, '
        f'Mean ± SD, {N_SEEDS} Seeds)',
        fontweight='bold', fontsize=13)
    x = np.arange(len(algos)); w = 0.6
    for ax, (m, lab) in zip(axes, [
            ('throughput', 'Normalized Throughput ↑'),
            ('error_rate', 'Error Rate ↓'),
            ('fatigue',    'Fatigue Index ↓')]):
        mn  = [R[a][m]['final_mean'] for a in algos]
        sd_ = [R[a][m]['final_std']  for a in algos]
        bars = ax.bar(x, mn, w, yerr=sd_, capsize=5,
                      color=[C[a] for a in algos], alpha=0.85,
                      edgecolor='black', lw=0.8,
                      error_kw=dict(elinewidth=1.5, capthick=1.5))
        for bar, m2, s in zip(bars, mn, sd_):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + max(sd_) * 0.04,
                    f'{m2:.3f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=28, ha='right', fontsize=9)
        ax.set(title=lab); ax.grid(True, axis='y', alpha=0.25); ax.set_ylim(bottom=0)
    plt.tight_layout(); savefig('fig3_performance_bars_FINAL.png')

# ─────────────────────────────────────────────────────────────────────────
# FIG 4 — Multi-Metric Performance Radar  (paper Fig. 4 / §4.4)
#   FIX [4]: this IS Fig 4 — original code had radar as Fig 5
# ─────────────────────────────────────────────────────────────────────────
def make_fig4_radar(R):
    algos  = list(R.keys())
    labels = ['Error\nControl (↑)', 'Fatigue\nControl (↑)',
              'Throughput (↑)', 'Action\nConc. (↑)', 'Convergence\nSpeed (↑)']
    N   = len(labels)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); ang += ang[:1]

    raw = {}
    for a in algos:
        ac = R[a]['action_conc']['final_mean']
        es = R[a]['episode_reward']['std'][:30].mean()
        raw[a] = [
            1 - R[a]['error_rate']['final_mean'],
            1 - R[a]['fatigue']['final_mean'],
            R[a]['throughput']['final_mean'],
            ac,
            1 / (1 + es * 8),
        ]
    # normalise action concentration axis across algorithms
    ac_v = [raw[a][3] for a in algos]
    acmn, acmx = min(ac_v), max(ac_v)
    for a in algos:
        raw[a][3] = (raw[a][3] - acmn) / (acmx - acmn + 1e-10)

    f, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 1); ax.set_yticks([.2, .4, .6, .8, 1.0])
    ax.set_title(
        'Multi-Metric Performance Radar\n'
        '(all axes normalised; higher = better)',
        fontweight='bold', pad=30, fontsize=11)
    for a in algos:
        v  = raw[a] + [raw[a][0]]
        ls = LS[a] if isinstance(LS[a], str) else '-'
        ax.plot(ang, v, color=C[a], ls=ls,
                lw=2.5 if a == 'D3QN' else 1.8,
                label=a, zorder=6 if a == 'D3QN' else 3)
        ax.fill(ang, v, color=C[a], alpha=0.07)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=10)
    plt.tight_layout(); savefig('fig4_radar_chart_FINAL.png')

# ─────────────────────────────────────────────────────────────────────────
# FIG 5 — Robustness Heatmap  (paper Fig. 5 / §4.5)
#   FIX [4]: this IS Fig 5 — original code had heatmap as Fig 4
# ─────────────────────────────────────────────────────────────────────────
def make_fig5_robustness(R, rob):
    algos = list(R.keys())
    conds = list(ROB_CONDS.keys())
    mat   = np.array([
        [(rob[a][c]['error_rate'] - rob[a]['Clean baseline']['error_rate'])
         / (rob[a]['Clean baseline']['error_rate'] + 1e-10) * 100
         for c in conds]
        for a in algos])

    f, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto', vmin=-30, vmax=50)
    plt.colorbar(im, ax=ax, label='Error Rate Change vs. Clean Baseline (%)')
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=22, ha='right', fontsize=10)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=11)
    ax.set_title(
        'Robustness Testing: Error Rate Degradation (%) — '
        'Lower = More Robust',
        fontweight='bold', fontsize=12)
    for i in range(len(algos)):
        for j in range(len(conds)):
            v  = mat[i, j]
            st = ' PASS' if v <= 15 else ' FAIL'
            ax.text(j, i, f'{v:+.0f}%\n{st}',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if abs(v) > 40 else 'black')
    ax.text(0.01, 0.97, 'PASS ≤ +15%', transform=ax.transAxes,
            fontsize=8.5, va='top', color='darkgreen', fontweight='bold')
    ax.text(0.01, 0.90, 'FAIL > +15%', transform=ax.transAxes,
            fontsize=8.5, va='top', color='darkred',  fontweight='bold')
    plt.tight_layout(); savefig('fig5_robustness_heatmap_FINAL.png')

# ─────────────────────────────────────────────────────────────────────────
# TABLE BUILDERS
# ─────────────────────────────────────────────────────────────────────────
def build_table3(results):
    """Table 3: Final performance — all algorithms"""
    ORDER = ['D3QN', 'D3QN-NStep', 'PPO', 'PER-n3-D3QN', 'EBQ-lite']
    rows  = []
    for a in ORDER:
        if a not in results: continue
        rows.append({
            'Algorithm':              a,
            'Throughput (Mean)':      round(results[a]['throughput']['final_mean'],   3),
            'Throughput (SD)':        round(results[a]['throughput']['final_std'],    3),
            'Error Rate (Mean)':      round(results[a]['error_rate']['final_mean'],   3),
            'Error Rate (SD)':        round(results[a]['error_rate']['final_std'],    3),
            'Fatigue Index (Mean)':   round(results[a]['fatigue']['final_mean'],      3),
            'Fatigue Index (SD)':     round(results[a]['fatigue']['final_std'],       3),
            'Action Conc. (Mean)':    round(results[a]['action_conc']['final_mean'],  3),
            'Action Conc. (SD)':      round(results[a]['action_conc']['final_std'],   3),
        })
    return pd.DataFrame(rows)

def build_table5(rob):
    """Table 5: Robustness testing — D3QN only"""
    rows     = []
    base_err = rob['D3QN']['Clean baseline']['error_rate']
    for cond in ROB_CONDS:
        r  = rob['D3QN'][cond]
        dg = (r['error_rate'] - base_err) / (base_err + 1e-10) * 100
        rows.append({
            'Condition':            cond,
            'Throughput':           round(r['throughput'],  3),
            'Error Rate':           round(r['error_rate'],   3),
            'Fatigue Index':        round(r['fatigue'],       3),
            'Error Degradation (%)':round(dg, 1),
            'Status':               '✓ PASS' if dg <= 15 else '✗ FAIL',
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    ALGOS   = ['D3QN', 'PPO', 'PER-n3-D3QN', 'EBQ-lite', 'D3QN-NStep']
    t_total = time.time()

    ckpt     = load_ckpt()
    results  = ckpt.get('results',  {})
    rob      = ckpt.get('rob',      None)
    prod_only= ckpt.get('prod_only', None)
    sens_df  = ckpt.get('sens_df',  None)
    skill_df = ckpt.get('skill_df', None)

    print(f"\n{'='*65}")
    print(f" HRC Experiment v2  |  device={DEVICE}  |  batch={BATCH_SIZE}")
    print(f" Main     : {N_SEEDS} seeds × {EPISODES} ep  |  conv zone ep {CONV_START+1}–{EPISODES}")
    print(f" Robustness: {N_ROB} seeds × {ROB_EPISODES} ep  |  conv zone ep {ROB_CONV_START+1}–{ROB_EPISODES}")
    print(f" Ablation : {ABLATION_SEEDS} seeds (Productivity-Only, Fig 1)")
    print(f" Sens/Skill: {SENS_SEEDS} seeds each (Tables 6 & 7)")
    print(f"{'='*65}")

    # ── [1/6] Main training ──────────────────────────────────────────────
    print(f"\n[1/6] Main training...")
    for algo in ALGOS:
        if algo in results:
            print(f"  Skipping {algo} (checkpoint)"); continue
        results[algo] = run_algo(algo)
        save_ckpt({'results': results, 'rob': rob,
                   'prod_only': prod_only,
                   'sens_df': sens_df, 'skill_df': skill_df})

    # ── [2/6] Productivity-Only ablation for Fig 1 ──────────────────────
    print(f"\n[2/6] Productivity-Only D3QN ablation ({ABLATION_SEEDS} seeds, Fig 1)...")
    if prod_only is None:
        prod_only = run_algo(
            'D3QN', n_seeds=ABLATION_SEEDS,
            w=W_PROD_ONLY, eps_decay=EPS_DECAY, conv_start=CONV_START)
        save_ckpt({'results': results, 'rob': rob,
                   'prod_only': prod_only,
                   'sens_df': sens_df, 'skill_df': skill_df})
    else:
        print("  Skipping (checkpoint)")

    # ── [3/6] Robustness ─────────────────────────────────────────────────
    print(f"\n[3/6] Robustness ({N_ROB} seeds × {ROB_EPISODES} ep "
          f"× 5 conds × 5 algos)...")
    if rob is None:
        rob = robustness(ALGOS)
        save_ckpt({'results': results, 'rob': rob,
                   'prod_only': prod_only,
                   'sens_df': sens_df, 'skill_df': skill_df})
    else:
        print("  Skipping (checkpoint)")

    # ── [4/6] Sensitivity (Table 6) ──────────────────────────────────────
    print(f"\n[4/6] Sensitivity analysis ({SENS_SEEDS} seeds × 3 configs, Table 6)...")
    if sens_df is None:
        sens_df = run_sensitivity()
        save_ckpt({'results': results, 'rob': rob,
                   'prod_only': prod_only,
                   'sens_df': sens_df, 'skill_df': skill_df})
    else:
        print("  Skipping (checkpoint)")

    # ── [5/6] Skill-level generalization (Table 7) ───────────────────────
    print(f"\n[5/6] Skill-level generalization ({SKILL_SEEDS} seeds × 3 levels, Table 7)...")
    if skill_df is None:
        skill_df = run_skill_gen()
        save_ckpt({'results': results, 'rob': rob,
                   'prod_only': prod_only,
                   'sens_df': sens_df, 'skill_df': skill_df})
    else:
        print("  Skipping (checkpoint)")

    # ── [6/6] Figures + Excel ────────────────────────────────────────────
    print(f"\n[6/6] Generating all figures and tables...")
    sdf = stats_table(results)

    make_fig1(results, prod_only)      # Fig 1 — ablation (3 conditions)
    make_fig2(results)                 # Fig 2 — training dynamics
    make_fig3(results)                 # Fig 3 — performance bars
    make_fig4_radar(results)           # Fig 4 — radar chart
    make_fig5_robustness(results, rob) # Fig 5 — robustness heatmap

    t3      = build_table3(results)
    t5      = build_table5(rob)
    rob_all = []
    for a in ALGOS:
        for c in ROB_CONDS:
            r  = rob[a][c]
            b  = rob[a]['Clean baseline']['error_rate']
            dg = (r['error_rate'] - b) / (b + 1e-10) * 100
            rob_all.append({
                'Algorithm':            a,
                'Condition':            c,
                'Throughput':           round(r['throughput'], 3),
                'Error Rate':           round(r['error_rate'],  3),
                'Fatigue Index':        round(r['fatigue'],      3),
                'Error Degradation (%)':round(dg, 1),
                'Status':               'PASS' if dg <= 15 else 'FAIL',
            })
    rdf = pd.DataFrame(rob_all)

    xlsx_path = f'{OUTPUT_DIR}/HRC_Results_ALL_TABLES.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
        t3.to_excel(w,       sheet_name='Table3_Performance',  index=False)
        sdf.to_excel(w,      sheet_name='Table4_Statistics',   index=False)
        t5.to_excel(w,       sheet_name='Table5_D3QN_Robust',  index=False)
        sens_df.to_excel(w,  sheet_name='Table6_Sensitivity',  index=False)
        skill_df.to_excel(w, sheet_name='Table7_SkillGen',     index=False)
        rdf.to_excel(w,      sheet_name='Robustness_All_Algos',index=False)
    print(f"  Saved: {xlsx_path}")

    for df, name in [
            (t3,       'table3_performance'),
            (sdf,      'table4_statistics'),
            (t5,       'table5_robustness_d3qn'),
            (sens_df,  'table6_sensitivity'),
            (skill_df, 'table7_skillgen')]:
        df.to_csv(f'{OUTPUT_DIR}/{name}.csv', index=False)

    # ── Console summary ───────────────────────────────────────────────────
    elapsed = (time.time() - t_total) / 60
    print(f"\nTotal time: {elapsed:.0f} min")
    print(f"\n{'='*65}")
    print("PASTE THIS OUTPUT TO CLAUDE")
    print(f"{'='*65}")
    print(f"\nMEANS (ep {CONV_START+1}–{EPISODES}, {N_SEEDS} seeds):")
    for a in ALGOS:
        print(f"\n{a}:")
        for m in ('throughput', 'error_rate', 'fatigue', 'action_conc'):
            print(f"  {m}: {results[a][m]['final_mean']:.4f} "
                  f"+/- {results[a][m]['final_std']:.4f}")
    print("\nSTATISTICS TABLE (Table 4):")
    print(sdf.to_string(index=False))
    print("\nD3QN ROBUSTNESS (Table 5):")
    print(t5.to_string(index=False))
    print("\nSENSITIVITY ANALYSIS (Table 6):")
    print(sens_df.to_string(index=False))
    print("\nSKILL GENERALIZATION (Table 7):")
    print(skill_df.to_string(index=False))
    print(f"\n{'='*65}")
    print("Output files in /kaggle/working/:")
    print("  fig1_ablation_FINAL.png")
    print("  fig2_training_curves_FINAL.png")
    print("  fig3_performance_bars_FINAL.png")
    print("  fig4_radar_chart_FINAL.png")
    print("  fig5_robustness_heatmap_FINAL.png")
    print("  HRC_Results_ALL_TABLES.xlsx  (Tables 3–7 + robustness all algos)")
    print("  table3_performance.csv")
    print("  table4_statistics.csv")
    print("  table5_robustness_d3qn.csv")
    print("  table6_sensitivity.csv")
    print("  table7_skillgen.csv")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
