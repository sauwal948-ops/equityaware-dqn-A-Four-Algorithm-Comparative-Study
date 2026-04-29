"""
==========================================================================
 HRC Experiment — CHECKPOINT VERSION
 Saves progress after each algorithm so a crash doesn't lose your work.
 If interrupted, just run again — it skips already-completed algorithms.
==========================================================================
"""

import os, random, warnings, pickle
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats

warnings.filterwarnings('ignore')
OUTPUT_DIR  = '/kaggle/working'
CKPT_FILE   = f'{OUTPUT_DIR}/checkpoint.pkl'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def save_checkpoint(data: dict):
    with open(CKPT_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"  [checkpoint saved → {CKPT_FILE}]")

def load_checkpoint() -> dict:
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE, 'rb') as f:
            data = pickle.load(f)
        completed = list(data.get('results', {}).keys())
        print(f"  [checkpoint found — already completed: {completed}]")
        return data
    return {}

# ─────────────────────────────────────────────────────────────
# SECTION 1: ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class CementBaggingEnv:
    ALPHA = {0: 1.2, 1: 1.0, 2: 0.8}
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

    def step(self, action, weights=(0.5, 0.3, 0.1, 0.1)):
        self.step_count += 1
        self.action_hist[action] += 1
        eff_action = action
        if self.downtime_range[1] > 0:
            p_down = self._rng.uniform(*self.downtime_range)
            if self._rng.random() < p_down and action in (1, 2):
                eff_action = 0
        load = self.LOAD[eff_action]
        dt   = 0.1
        self.fatigue = float(np.clip(
            self.fatigue + dt * (self.alpha * load - self.BETA * self.fatigue), 0.0, 1.0))
        if eff_action == 3:
            self.fatigue = max(0.0, self.fatigue - 0.05)
        arrivals     = self._rng.poisson(1.2 * dt)
        service_rate = self.machine_spd * (0.7 if eff_action == 2 else 1.0)
        if eff_action == 3:
            service_rate *= 0.5
        self.queue      = float(np.clip(self.queue + arrivals - service_rate, 0, 50))
        throughput      = service_rate / 1.2
        delta           = 0.02 * (self.fatigue - 0.2) + self._rng.normal(0, 0.005)
        self.error_rate = float(np.clip(self.error_rate + delta * dt, 0.005, 0.50))
        if eff_action == 1:   self.error_rate *= 0.995
        elif eff_action == 2: self.error_rate *= 0.990
        self.machine_spd = float(np.clip(
            self.machine_spd + self._rng.normal(0, 0.01), 0.3, 1.2))
        total  = max(1, self.action_hist.sum())
        probs  = self.action_hist / total
        bias   = float(np.var(probs) * 4.0)
        w1, w2, w3, w4 = weights
        reward = w1 * throughput - w2 * self.error_rate - w3 * self.fatigue - w4 * bias
        done   = (self.step_count >= 500)
        info   = dict(throughput=throughput, error_rate=self.error_rate,
                      fatigue=self.fatigue, bias=bias)
        return self._observe(), reward, done, info

    def _observe(self):
        obs = np.array([self.machine_spd, self.fatigue, self.queue / 50.0,
                        self.error_rate, self.skill_level / 2.0], dtype=np.float32)
        if self.noise_std > 0:
            obs[[1, 3]] += self._rng.normal(0, self.noise_std, size=2).astype(np.float32)
            obs = np.clip(obs, 0.0, 1.0)
        return obs

    @property
    def state_dim(self):  return 5
    @property
    def action_dim(self): return 4


# ─────────────────────────────────────────────────────────────
# SECTION 2: NEURAL NETWORK
# ─────────────────────────────────────────────────────────────

class D3QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=4, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU())
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, action_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x)
        V = self.value_stream(h)
        A = self.adv_stream(h)
        return V + A - A.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────
# SECTION 3: REPLAY BUFFERS
# ─────────────────────────────────────────────────────────────

Transition = namedtuple('Transition', ['s', 'a', 'r', 'ns', 'done'])

class UniformBuffer:
    def __init__(self, capacity=10_000):
        self._buf = deque(maxlen=capacity)
    def push(self, s, a, r, ns, done):
        self._buf.append(Transition(s, a, r, ns, done))
    def sample(self, n):
        return Transition(*zip(*random.sample(self._buf, n)))
    def __len__(self): return len(self._buf)

class _SumTree:
    def __init__(self, capacity):
        self.cap  = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.ptr  = 0
        self.size = 0
    def _propagate(self, idx, delta):
        while idx > 0:
            idx = (idx - 1) >> 1
            self.tree[idx] += delta
    def add(self, priority, datum):
        leaf = self.ptr + self.cap - 1
        self.data[self.ptr] = datum
        self.update(leaf, priority)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
    def update(self, leaf_idx, priority):
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)
    def _retrieve(self, idx, s):
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree): return idx
            if s <= self.tree[left]:   idx = left
            else:                      s -= self.tree[left]; idx = left + 1
    def get(self, s):
        leaf = self._retrieve(0, s)
        return leaf, self.tree[leaf], self.data[leaf - self.cap + 1]
    @property
    def total(self): return self.tree[0]

class PERBuffer:
    _EPS = 1e-6
    def __init__(self, capacity=10_000, alpha=0.6,
                 beta_start=0.4, beta_end=1.0, beta_steps=50_000):
        self._tree      = _SumTree(capacity)
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self._step      = 0
        self._max_p     = 1.0
    @property
    def beta(self):
        return self.beta_start + min(1.0, self._step/self.beta_steps) * (self.beta_end - self.beta_start)
    def push(self, s, a, r, ns, done):
        self._tree.add(self._max_p ** self.alpha, Transition(s, a, r, ns, done))
    def sample(self, n):
        self._step += 1
        indices, priorities, batch = [], [], []
        seg = self._tree.total / n
        for i in range(n):
            s   = random.uniform(seg * i, seg * (i + 1))
            idx, p, datum = self._tree.get(s)
            if datum is None:
                idx, p, datum = self._tree.get(random.uniform(0, self._tree.total))
            indices.append(idx); priorities.append(p); batch.append(datum)
        probs   = np.array(priorities) / (self._tree.total + self._EPS)
        weights = (self._tree.size * probs + self._EPS) ** (-self.beta)
        weights /= weights.max()
        return Transition(*zip(*batch)), indices, torch.FloatTensor(weights)
    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + self._EPS) ** self.alpha
            self._tree.update(idx, p)
            self._max_p = max(self._max_p, p)
    def __len__(self): return self._tree.size

class _NStepAccumulator:
    def __init__(self, n=3, gamma=0.95):
        self.n = n; self.gamma = gamma; self._buf = deque()
    def push(self, s, a, r, ns, done):
        self._buf.append((s, a, r, ns, done))
        if len(self._buf) < self.n and not done: return []
        return self._flush_one(terminal=done)
    def _flush_one(self, terminal):
        results = []
        while self._buf and (len(self._buf) >= self.n or terminal):
            ret, gam = 0.0, 1.0; last_ns = last_done = None
            for _, _, ri, nsi, di in self._buf:
                ret += gam * ri; gam *= self.gamma
                last_ns = nsi; last_done = di
                if di: terminal = True; break
            s0, a0 = self._buf[0][0], self._buf[0][1]
            results.append(Transition(s0, a0, ret, last_ns, last_done))
            self._buf.popleft()
            if not terminal: break
        return results
    def drain(self):
        results = []
        while self._buf: results.extend(self._flush_one(terminal=True))
        return results

class EBQBuffer:
    def __init__(self, capacity=10_000, dup_factor=2,
                 reward_boost_lambda=0.5, positive_threshold=0.0):
        self._buf       = deque(maxlen=capacity)
        self.dup_factor = dup_factor
        self.lam        = reward_boost_lambda
        self.pos_thresh = positive_threshold
        self._ep_buf    = []; self._ep_return = 0.0
    def push_step(self, s, a, r, ns, done):
        self._ep_buf.append(Transition(s, a, r, ns, done))
        self._ep_return += r
        if done: self._commit()
    def _commit(self):
        if not self._ep_buf: return
        t = self._ep_buf[-1]
        r_aug = t.r + self.lam * (self._ep_return / max(1, len(self._ep_buf)))
        self._ep_buf[-1] = Transition(t.s, t.a, r_aug, t.ns, t.done)
        for tr in self._ep_buf:
            self._buf.append(tr)
            if tr.r > self.pos_thresh:
                for _ in range(self.dup_factor - 1): self._buf.append(tr)
        self._ep_buf.clear(); self._ep_return = 0.0
    def sample(self, n):
        return Transition(*zip(*random.sample(self._buf, min(n, len(self._buf)))))
    def __len__(self): return len(self._buf)


# ─────────────────────────────────────────────────────────────
# SECTION 4: AGENTS
# ─────────────────────────────────────────────────────────────

class D3QNAgent:
    def __init__(self, state_dim=5, action_dim=4, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay_ep=60,
                 buf_size=10_000, batch=32, tgt_update=100, seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.action_dim = action_dim; self.gamma = gamma
        self.batch = batch; self.tgt_update = tgt_update
        self.eps = eps_start; self.eps_end = eps_end
        self.eps_delta = (eps_start - eps_end) / eps_decay_ep
        self._step = 0
        self.online = D3QNetwork(state_dim, action_dim)
        self.target = D3QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = UniformBuffer(buf_size)

    def act(self, s, explore=True):
        if explore and random.random() < self.eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.online(torch.FloatTensor(s).unsqueeze(0)).argmax().item()

    def store(self, s, a, r, ns, done): self.buf.push(s, a, r, ns, done)
    def decay_eps(self): self.eps = max(self.eps_end, self.eps - self.eps_delta)

    def _learn_from_batch(self, states, actions, rewards, next_states, dones, weights=None):
        with torch.no_grad():
            best_a = self.online(next_states).argmax(1, keepdim=True)
            tgt_q  = rewards + self.gamma * self.target(next_states).gather(1, best_a).squeeze(1) * (1 - dones)
        curr_q = self.online(states).gather(1, actions).squeeze(1)
        td_err = tgt_q - curr_q
        loss   = ((td_err**2) if weights is None else (weights * td_err**2)).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step(); self._step += 1
        if self._step % self.tgt_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return td_err.detach().numpy(), loss.item()

    def learn(self):
        if len(self.buf) < self.batch: return
        b = self.buf.sample(self.batch)
        self._learn_from_batch(
            torch.FloatTensor(np.array(b.s)), torch.LongTensor(b.a).unsqueeze(1),
            torch.FloatTensor(b.r), torch.FloatTensor(np.array(b.ns)),
            torch.FloatTensor(b.done))


class PERnD3QNAgent(D3QNAgent):
    def __init__(self, n_steps=3, per_alpha=0.6, per_beta_start=0.4, **kwargs):
        super().__init__(**kwargs)
        self.gamma_n = self.gamma ** n_steps
        self.buf     = PERBuffer(capacity=kwargs.get('buf_size', 10_000),
                                 alpha=per_alpha, beta_start=per_beta_start)
        self._nstep  = _NStepAccumulator(n=n_steps, gamma=self.gamma)

    def store(self, s, a, r, ns, done):
        for t in self._nstep.push(s, a, r, ns, done): self.buf.push(t.s, t.a, t.r, t.ns, t.done)
        if done:
            for t in self._nstep.drain(): self.buf.push(t.s, t.a, t.r, t.ns, t.done)

    def learn(self):
        if len(self.buf) < self.batch: return
        b, indices, weights = self.buf.sample(self.batch)
        td_errors, _ = self._learn_from_batch(
            torch.FloatTensor(np.array(b.s)), torch.LongTensor(b.a).unsqueeze(1),
            torch.FloatTensor(b.r), torch.FloatTensor(np.array(b.ns)),
            torch.FloatTensor(b.done), weights=weights)
        self.buf.update_priorities(indices, td_errors)


class EBQLiteAgent(D3QNAgent):
    def __init__(self, dup_factor=2, reward_boost_lambda=0.5, **kwargs):
        super().__init__(**kwargs)
        self.buf = EBQBuffer(capacity=kwargs.get('buf_size', 10_000),
                             dup_factor=dup_factor, reward_boost_lambda=reward_boost_lambda)
    def store(self, s, a, r, ns, done): self.buf.push_step(s, a, r, ns, done)
    def learn(self):
        if len(self.buf) < self.batch: return
        b = self.buf.sample(self.batch)
        self._learn_from_batch(
            torch.FloatTensor(np.array(b.s)), torch.LongTensor(b.a).unsqueeze(1),
            torch.FloatTensor(b.r), torch.FloatTensor(np.array(b.ns)),
            torch.FloatTensor(b.done))


class D3QNNStepAgent(D3QNAgent):
    """n-step returns only — NO prioritized replay. Ablation for PER-n³-D3QN."""
    def __init__(self, n_steps=3, **kwargs):
        super().__init__(**kwargs)
        self.gamma_n = self.gamma ** n_steps
        self._nstep  = _NStepAccumulator(n=n_steps, gamma=self.gamma)

    def store(self, s, a, r, ns, done):
        for t in self._nstep.push(s, a, r, ns, done): self.buf.push(t.s, t.a, t.r, t.ns, t.done)
        if done:
            for t in self._nstep.drain(): self.buf.push(t.s, t.a, t.r, t.ns, t.done)

    def _learn_from_batch(self, states, actions, rewards, next_states, dones, weights=None):
        with torch.no_grad():
            best_a = self.online(next_states).argmax(1, keepdim=True)
            tgt_q  = rewards + self.gamma_n * self.target(next_states).gather(1, best_a).squeeze(1) * (1 - dones)
        curr_q = self.online(states).gather(1, actions).squeeze(1)
        td_err = tgt_q - curr_q
        loss   = (td_err**2).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step(); self._step += 1
        if self._step % self.tgt_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return td_err.detach().numpy(), loss.item()


class PPONet(nn.Module):
    def __init__(self, state_dim=5, action_dim=4, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden),   nn.Tanh())
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return F.softmax(self.actor(h), dim=-1), self.critic(h)


class PPOAgent:
    def __init__(self, state_dim=5, action_dim=4, lr=3e-4, gamma=0.95,
                 clip=0.2, ppo_epochs=4, seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.gamma = gamma; self.clip = clip; self.epochs = ppo_epochs
        self.net   = PPONet(state_dim, action_dim)
        self.opt   = optim.Adam(self.net.parameters(), lr=lr)
        self._traj = []

    def act(self, s):
        with torch.no_grad():
            probs, _ = self.net(torch.FloatTensor(s).unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        a    = dist.sample()
        return a.item(), dist.log_prob(a).item()

    def store(self, s, a, lp, r, ns, done): self._traj.append((s, a, lp, r, done))

    def learn(self):
        if not self._traj: return
        states, actions, old_lps, rewards, dones = zip(*self._traj)
        states  = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_lps = torch.FloatTensor(old_lps)
        returns, R = [], 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d); returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for _ in range(self.epochs):
            probs, values = self.net(states)
            dist  = torch.distributions.Categorical(probs)
            lps   = dist.log_prob(actions)
            ratio = torch.exp(lps - old_lps)
            adv   = returns - values.squeeze().detach()
            loss  = (-torch.min(ratio * adv,
                                torch.clamp(ratio, 1-self.clip, 1+self.clip)*adv).mean()
                     + 0.5 * F.mse_loss(values.squeeze(), returns))
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5); self.opt.step()
        self._traj.clear()


# ─────────────────────────────────────────────────────────────
# SECTION 5: TRAINING LOOPS
# ─────────────────────────────────────────────────────────────

WEIGHTS = (0.5, 0.3, 0.1, 0.1)

def _run_episode_dqn(agent, env):
    s = env.reset()
    ep_info = {k: [] for k in ('throughput','error_rate','fatigue','bias')}
    ep_r = 0.0
    while True:
        a = agent.act(s)
        ns, r, done, info = env.step(a, WEIGHTS)
        agent.store(s, a, r, ns, done); agent.learn()
        s = ns; ep_r += r
        for k in ep_info: ep_info[k].append(info[k])
        if done: break
    agent.decay_eps()
    return ep_r / 500, ep_info

def _run_episode_ppo(agent, env):
    s = env.reset()
    ep_info = {k: [] for k in ('throughput','error_rate','fatigue','bias')}
    ep_r = 0.0
    while True:
        a, lp = agent.act(s)
        ns, r, done, info = env.step(a, WEIGHTS)
        agent.store(s, a, lp, r, ns, done)
        s = ns; ep_r += r
        for k in ep_info: ep_info[k].append(info[k])
        if done: break
    agent.learn()
    return ep_r / 500, ep_info

def train_one_seed(algo_name, seed, env_kwargs=None, episodes=100):
    env_kw = env_kwargs or {}
    env    = CementBaggingEnv(seed=seed, **env_kw)
    if   algo_name == 'D3QN':          agent = D3QNAgent(seed=seed);          run_ep = lambda: _run_episode_dqn(agent, env)
    elif algo_name == 'PER-n³-D3QN':   agent = PERnD3QNAgent(n_steps=3, per_alpha=0.6, per_beta_start=0.4, seed=seed); run_ep = lambda: _run_episode_dqn(agent, env)
    elif algo_name == 'EBQ-lite':       agent = EBQLiteAgent(dup_factor=2, reward_boost_lambda=0.5, seed=seed); run_ep = lambda: _run_episode_dqn(agent, env)
    elif algo_name == 'D3QN-NStep':    agent = D3QNNStepAgent(n_steps=3, seed=seed); run_ep = lambda: _run_episode_dqn(agent, env)
    elif algo_name == 'PPO':            agent = PPOAgent(seed=seed);           run_ep = lambda: _run_episode_ppo(agent, env)
    else: raise ValueError(f"Unknown: {algo_name}")
    hist = {k: [] for k in ('episode_reward','throughput','error_rate','fatigue','bias')}
    for _ in range(episodes):
        ep_r, ep_info = run_ep()
        hist['episode_reward'].append(ep_r)
        for k in ('throughput','error_rate','fatigue','bias'):
            hist[k].append(float(np.mean(ep_info[k])))
    return hist

def run_experiment(algo_name, n_seeds=20, env_kwargs=None, episodes=100):
    print(f"  Training {algo_name} ({n_seeds} seeds × {episodes} ep)...")
    all_hists = []
    for seed in range(n_seeds):
        print(f"    seed {seed+1:02d}/{n_seeds}", end='\r')
        all_hists.append(train_one_seed(algo_name, seed, env_kwargs, episodes))
    print(f"    {algo_name} DONE                    ")
    agg = {}
    for key in ('episode_reward','throughput','error_rate','fatigue','bias'):
        mat = np.array([h[key] for h in all_hists])
        agg[key] = dict(mean=mat.mean(axis=0), std=mat.std(axis=0),
                        per_seed=mat, final_mean=mat[:,70:].mean(),
                        final_std=mat[:,70:].std(),
                        per_seed_final=mat[:,70:].mean(axis=1))
    return agg


# ─────────────────────────────────────────────────────────────
# SECTION 6: ROBUSTNESS TESTING
# ─────────────────────────────────────────────────────────────

ROBUSTNESS_CONDITIONS = {
    'Clean baseline'     : dict(noise_std=0.00, downtime_range=(0.00, 0.00)),
    'Sensor noise 10%'   : dict(noise_std=0.10, downtime_range=(0.00, 0.00)),
    'Sensor noise 20%'   : dict(noise_std=0.20, downtime_range=(0.00, 0.00)),
    'Downtime 5-20%'     : dict(noise_std=0.00, downtime_range=(0.05, 0.20)),
    'Compound (noise+DT)': dict(noise_std=0.10, downtime_range=(0.05, 0.20)),
}

def robustness_test(algos, n_seeds=5, episodes=100):
    rob = {}
    for algo in algos:
        rob[algo] = {}
        for cond, env_kw in ROBUSTNESS_CONDITIONS.items():
            metrics = {k: [] for k in ('throughput','error_rate','fatigue')}
            for seed in range(n_seeds):
                h = train_one_seed(algo, seed, env_kw, episodes)
                for k in metrics: metrics[k].append(np.mean(h[k][70:]))
            rob[algo][cond] = {k: np.mean(v) for k, v in metrics.items()}
        print(f"    Robustness done: {algo}")
    return rob


# ─────────────────────────────────────────────────────────────
# SECTION 7: STATISTICS
# ─────────────────────────────────────────────────────────────

def paired_ttest_cohens_d(a, b):
    t, p = stats.ttest_rel(a, b)
    diff = np.array(a) - np.array(b)
    d    = diff.mean() / (diff.std(ddof=1) + 1e-10)
    return t, p, d

def build_stats_table(results, baseline='D3QN'):
    rows = []
    for algo in [k for k in results if k != baseline]:
        for metric in ('throughput','error_rate','fatigue','bias'):
            a = results[baseline][metric]['per_seed_final']
            b = results[algo][metric]['per_seed_final']
            t, p, d = paired_ttest_cohens_d(a, b)
            change  = (b.mean() - a.mean()) / (a.mean() + 1e-10) * 100
            rows.append({'Comparison': f'{baseline} vs {algo}', 'Metric': metric,
                         f'{baseline}_mean': round(float(a.mean()),4),
                         f'{algo}_mean':     round(float(b.mean()),4),
                         'Change_%': round(change,1), 't_stat': round(t,3),
                         'p_value': '<0.001' if p<0.001 else round(p,4),
                         'Cohen_d': round(d,3),
                         'Significant': 'Yes' if p<0.05 else 'No'})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# SECTION 8: FIGURES
# ─────────────────────────────────────────────────────────────

COLORS = {'D3QN':'#2196F3','PPO':'#FF5722','PER-n³-D3QN':'#4CAF50',
          'EBQ-lite':'#9C27B0','D3QN-NStep':'#FF9800'}
STYLES = {'D3QN':'-','PPO':'--','PER-n³-D3QN':'-.','EBQ-lite':':','D3QN-NStep':(0,(3,1,1,1))}

plt.rcParams.update({'font.family':'serif','font.size':11,'axes.titlesize':12,
                     'axes.labelsize':11,'legend.fontsize':9,'figure.dpi':150})

def fig1_training_curves(results):
    ep = np.arange(1, 101); algos = list(results.keys())
    keys = [('throughput','Normalized Throughput','(a)'),('error_rate','Error Rate','(b)'),
            ('fatigue','Fatigue Index','(c)'),('episode_reward','Episode Reward (norm.)','(d)')]
    fig, axes = plt.subplots(2, 2, figsize=(14,9), sharex=True)
    fig.suptitle('Training Dynamics — All Algorithms (20 Seeds, ±1 SD)', fontweight='bold', fontsize=13)
    for ax, (metric, ylabel, tag) in zip(axes.flat, keys):
        for algo in algos:
            m = results[algo][metric]['mean']; s = results[algo][metric]['std']
            ax.plot(ep, m, color=COLORS[algo], ls=STYLES[algo] if isinstance(STYLES[algo],str) else '-', lw=1.8, label=algo)
            ax.fill_between(ep, m-s, m+s, color=COLORS[algo], alpha=0.10)
        ax.axvline(70, color='gray', ls=':', lw=1, alpha=0.6)
        ax.set_ylabel(ylabel); ax.set_title(f'{tag} {ylabel}')
        ax.grid(True, alpha=0.25); ax.legend(loc='best')
    for ax in axes[1]: ax.set_xlabel('Episode')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/fig1_training_curves.png'
    plt.savefig(path, bbox_inches='tight', dpi=150); plt.close(); print(f"  Saved: {path}")

def fig2_performance_bars(results):
    algos = list(results.keys())
    metrics = [('throughput','Normalized Throughput','↑'),('error_rate','Error Rate','↓'),('fatigue','Fatigue Index','↓')]
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Final Performance (Episodes 71–100, Mean ± SD, 20 Seeds)', fontweight='bold')
    x = np.arange(len(algos)); w = 0.55
    for ax, (metric, label, arrow) in zip(axes, metrics):
        means = [results[a][metric]['final_mean'] for a in algos]
        stds  = [results[a][metric]['final_std']  for a in algos]
        bars  = ax.bar(x, means, w, yerr=stds, capsize=4,
                       color=[COLORS[a] for a in algos], alpha=0.85, edgecolor='black', linewidth=0.7)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(stds)*0.08,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(algos, rotation=25, ha='right')
        ax.set_ylabel(label); ax.set_title(f'{label} {arrow}'); ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/fig2_performance_bars.png'
    plt.savefig(path, bbox_inches='tight', dpi=150); plt.close(); print(f"  Saved: {path}")

def fig3_robustness_heatmap(results, rob):
    algos = list(results.keys()); conds = list(ROBUSTNESS_CONDITIONS.keys())
    baselines = {a: rob[a]['Clean baseline']['error_rate'] for a in algos}
    matrix    = np.zeros((len(algos), len(conds)))
    for i,a in enumerate(algos):
        for j,c in enumerate(conds):
            b = baselines[a]; matrix[i,j] = (rob[a][c]['error_rate']-b)/(b+1e-10)*100
    fig, ax = plt.subplots(figsize=(14,5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=-20, vmax=420)
    cb = plt.colorbar(im, ax=ax); cb.set_label('Error Rate Degradation vs. Clean Baseline (%)', fontsize=10)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=22, ha='right')
    ax.set_yticks(range(len(algos))); ax.set_yticklabels(algos)
    ax.set_title('Robustness: Error Rate Degradation (%) — Lower = More Robust', fontweight='bold')
    for i in range(len(algos)):
        for j in range(len(conds)):
            txt = f'{matrix[i,j]:+.0f}%'; color = 'white' if abs(matrix[i,j])>200 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/fig3_robustness_heatmap.png'
    plt.savefig(path, bbox_inches='tight', dpi=150); plt.close(); print(f"  Saved: {path}")

def fig4_radar_chart(results):
    algos  = list(results.keys())
    labels = ['Throughput\n(↑)','Error\nControl\n(↑)','Fatigue\nControl\n(↑)','Equity\n(bias↓)','Convergence\nSpeed\n(↑)']
    N      = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    def scores(algo):
        return [results[algo]['throughput']['final_mean'],
                1-results[algo]['error_rate']['final_mean'],
                1-results[algo]['fatigue']['final_mean'],
                1-results[algo]['bias']['final_mean'],
                1/(1+results[algo]['episode_reward']['std'][:30].mean()*8)]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0,1)
    ax.set_title('Multi-Metric Performance Radar\n(all axes normalised; higher = better)',
                 fontweight='bold', pad=25)
    for algo in algos:
        vals = scores(algo) + [scores(algo)[0]]
        ax.plot(angles, vals, 'o-', color=COLORS[algo], lw=2, label=algo,
                ls=STYLES[algo] if isinstance(STYLES[algo],str) else '-')
        ax.fill(angles, vals, color=COLORS[algo], alpha=0.07)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4,1.12))
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/fig4_radar_chart.png'
    plt.savefig(path, bbox_inches='tight', dpi=150); plt.close(); print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# SECTION 9: EXPORT
# ─────────────────────────────────────────────────────────────

def export_all(results, rob, stats_df):
    algos = list(results.keys())
    rows  = []
    for algo in algos:
        row = {'Algorithm': algo}
        for m in ('throughput','error_rate','fatigue','bias'):
            row[f'{m}_mean'] = round(results[algo][m]['final_mean'],4)
            row[f'{m}_sd']   = round(results[algo][m]['final_std'],4)
        rows.append(row)
    perf_df = pd.DataFrame(rows)
    rob_rows = []
    for algo in algos:
        for cond in ROBUSTNESS_CONDITIONS:
            r = rob[algo][cond]; base_err = rob[algo]['Clean baseline']['error_rate']
            degrad = (r['error_rate']-base_err)/(base_err+1e-10)*100
            rob_rows.append({'Algorithm':algo,'Condition':cond,
                             'Throughput':round(r['throughput'],4),
                             'Error_Rate':round(r['error_rate'],4),
                             'Fatigue':round(r['fatigue'],4),
                             'Err_Degrad_%':round(degrad,1),
                             'Status':'PASS' if degrad<=15 else 'FAIL'})
    rob_df = pd.DataFrame(rob_rows)
    xlsx_path = f'{OUTPUT_DIR}/extended_results.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        perf_df.to_excel(writer,  sheet_name='Performance_Summary', index=False)
        stats_df.to_excel(writer, sheet_name='Statistical_Tests',   index=False)
        rob_df.to_excel(writer,   sheet_name='Robustness_Testing',  index=False)
    perf_df.to_csv(f'{OUTPUT_DIR}/performance_summary.csv',  index=False)
    stats_df.to_csv(f'{OUTPUT_DIR}/statistical_tests.csv',   index=False)
    rob_df.to_csv(f'{OUTPUT_DIR}/robustness_testing.csv',    index=False)
    print(f"  Saved: {xlsx_path} + 3 x CSV")
    return perf_df, rob_df


# ─────────────────────────────────────────────────────────────
# SECTION 10: MAIN  (with checkpointing)
# ─────────────────────────────────────────────────────────────

def main():
    N_SEEDS_MAIN = 20
    N_SEEDS_ROB  = 5
    EPISODES     = 100
    ALGOS        = ['D3QN', 'PPO', 'PER-n³-D3QN', 'EBQ-lite', 'D3QN-NStep']

    print("=" * 65)
    print(" HRC Experiment — with crash-safe checkpointing")
    print("=" * 65)

    # ── Load any existing progress ───────────────────────────
    ckpt    = load_checkpoint()
    results = ckpt.get('results', {})
    rob     = ckpt.get('rob',     None)

    # ── Main training — skip already-done algorithms ─────────
    print("\n[1/4] Main training runs...")
    any_new = False
    for algo in ALGOS:
        if algo in results:
            print(f"  Skipping {algo} (already in checkpoint)")
            continue
        results[algo] = run_experiment(algo, N_SEEDS_MAIN, episodes=EPISODES)
        any_new = True
        save_checkpoint({'results': results, 'rob': rob})

    # ── Robustness ───────────────────────────────────────────
    print("\n[2/4] Robustness testing...")
    if rob is None:
        rob = robustness_test(ALGOS, N_SEEDS_ROB, EPISODES)
        save_checkpoint({'results': results, 'rob': rob})
    else:
        print("  Skipping robustness (already in checkpoint)")

    # ── Statistics ───────────────────────────────────────────
    print("\n[3/4] Statistical tests...")
    stats_df = build_stats_table(results, baseline='D3QN')

    # ── Figures ──────────────────────────────────────────────
    print("\n[4/4] Generating figures...")
    fig1_training_curves(results)
    fig2_performance_bars(results)
    fig3_robustness_heatmap(results, rob)
    fig4_radar_chart(results)

    # ── Export ───────────────────────────────────────────────
    print("\nExporting results...")
    perf_df, rob_df = export_all(results, rob, stats_df)

    # ══════════════════════════════════════════════════════════
    # MANUSCRIPT DIAGNOSTIC BLOCK
    # Copy everything below this line and paste to Claude
    # ══════════════════════════════════════════════════════════
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("MANUSCRIPT UPDATE — PASTE THIS ENTIRE BLOCK TO CLAUDE")
    print(SEP)

    print("\n--- MEANS +/- SD (Episodes 71-100, 20 seeds) ---")
    for algo in ALGOS:
        print(f"\n{algo}:")
        for m in ('throughput','error_rate','fatigue','bias'):
            mn = results[algo][m]['final_mean']
            sd = results[algo][m]['final_std']
            print(f"  {m:<14}: {mn:.4f} +/- {sd:.4f}")

    print(f"\n--- FULL STATISTICAL TABLE ---")
    print(stats_df.to_string(index=False))

    print(f"\n--- ROBUSTNESS TABLE ---")
    print(rob_df[['Algorithm','Condition','Throughput',
                  'Error_Rate','Fatigue','Err_Degrad_%','Status']].to_string(index=False))

    print(f"\n--- N-STEP ABLATION (D3QN vs D3QN-NStep vs PER-n3-D3QN) ---")
    for m in ('throughput','error_rate','fatigue'):
        d3  = results['D3QN'][m]['final_mean']
        ns  = results['D3QN-NStep'][m]['final_mean']
        per = results['PER-n³-D3QN'][m]['final_mean']
        a   = results['D3QN'][m]['per_seed_final']
        b   = results['D3QN-NStep'][m]['per_seed_final']
        _, p_ns, d_ns = paired_ttest_cohens_d(a, b)
        print(f"  {m:<14}: D3QN={d3:.4f}  NStep={ns:.4f}  PER={per:.4f}"
              f"  | D3QN vs NStep: p={p_ns:.4f} d={d_ns:+.3f}")

    print(f"\n--- EQUITY BIAS DIRECTION CHECK ---")
    d3b  = results['D3QN']['bias']['final_mean']
    ppob = results['PPO']['bias']['final_mean']
    print(f"  D3QN bias : {d3b:.4f}")
    print(f"  PPO  bias : {ppob:.4f}")
    if d3b < ppob:
        print("  >> OK: D3QN lower than PPO — table arrow DOWN is CORRECT")
    else:
        print("  >> WARNING: D3QN has HIGHER bias than PPO — table arrow needs fixing")

    print(f"\n{SEP}")
    print(f"All files saved to {OUTPUT_DIR}/")
    print(SEP)


if __name__ == '__main__':
    main()
