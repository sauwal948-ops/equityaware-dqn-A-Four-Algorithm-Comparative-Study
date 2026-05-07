"""
==========================================================================
 Robustness Re-run FINAL  —  N_ROB=7, ROB_EPISODES=100, both bugs fixed

 Changes from previous rerun:
   ROB_EPISODES   = 100  (was 50  — gives agents time to properly converge)
   ROB_CONV_START = 70   (was 30  — convergence zone ep 71-100)
   N_ROB          = 7    (kept   — stable seed count)

 Bug fixes carried forward:
   Bug 1: obs[[1,3]] clipped only (not full obs vector)
   Bug 2: no double learn() at episode end

 Expected runtime: ~220 min on CPU (~35 min on GPU)
 Touches ONLY: fig5_robustness_heatmap_FINAL.png
               table5_robustness_d3qn.csv
               Table5 sheet in HRC_Results_ALL_TABLES.xlsx

 HOW TO USE IN KAGGLE:
   1. Make sure hrc_checkpoint_v2.pkl is in /kaggle/working/
      (upload it via Settings → Add Input, then copy with shutil)
   2. Paste this entire script into one cell and run
==========================================================================
"""

import os, random, warnings, pickle, time
from collections import deque, namedtuple
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings('ignore')

# ── Device ────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256 if DEVICE.type == 'cuda' else 64
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

OUTPUT_DIR = '/kaggle/working'
CKPT_FILE  = f'{OUTPUT_DIR}/hrc_checkpoint_v2.pkl'

# ── Constants (must match original run exactly) ───────────────
EPS_DECAY   = 100
LEARN_EVERY = 4
W_COMPOSITE = (0.5, 0.3, 0.1, 0.1)

# ── Robustness parameters (these are what changed) ────────────
N_ROB          = 7    # 7 seeds — stable estimates
ROB_EPISODES   = 100  # 100 ep — agents fully converge before measuring
ROB_CONV_START = 70   # measure performance over ep 71-100
ROB_EPS_DECAY  = 50   # scaled: 100/200 × 100 = 50 (decay over first 50 ep)

ROB_CONDS = {
    'Clean baseline':      dict(noise_std=0.00, downtime_range=(0.00, 0.00)),
    'Sensor noise 10%':    dict(noise_std=0.10, downtime_range=(0.00, 0.00)),
    'Sensor noise 20%':    dict(noise_std=0.20, downtime_range=(0.00, 0.00)),
    'Downtime 5-20%':      dict(noise_std=0.00, downtime_range=(0.05, 0.20)),
    'Compound (noise+DT)': dict(noise_std=0.10, downtime_range=(0.05, 0.20)),
}

Transition = namedtuple('Transition', ['s', 'a', 'r', 'ns', 'done'])

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT  —  Bug 1 fixed
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

    def step(self, action, weights=W_COMPOSITE):
        self.step_count += 1
        self.action_hist[action] += 1
        eff = action
        if self.downtime_range[1] > 0:
            if self._rng.random() < self._rng.uniform(*self.downtime_range) \
               and action in (1, 2):
                eff = 0
        load = self.LOAD[eff]; dt = 0.1
        self.fatigue = float(np.clip(
            self.fatigue + dt * (self.alpha * load - self.BETA * self.fatigue),
            0.0, 1.0))
        if eff == 3:
            self.fatigue = max(0.0, self.fatigue - 0.05)
        arr  = self._rng.poisson(1.2 * dt)
        sr   = (self.machine_spd
                * (0.7 if eff == 2 else 1.0)
                * (0.5 if eff == 3 else 1.0))
        self.queue      = float(np.clip(self.queue + arr - sr, 0, 50))
        tput            = sr / 1.2
        self.error_rate = float(np.clip(
            self.error_rate
            + (0.02 * (self.fatigue - 0.2)
               + self._rng.normal(0, 0.005)) * dt,
            0.005, 0.50))
        if eff == 1: self.error_rate *= 0.995
        elif eff == 2: self.error_rate *= 0.990
        self.machine_spd = float(np.clip(
            self.machine_spd + self._rng.normal(0, 0.01), 0.3, 1.2))
        total       = max(1, self.action_hist.sum())
        action_conc = float(np.var(self.action_hist / total) * 4.0)
        w1, w2, w3, w4 = weights
        reward = w1*tput - w2*self.error_rate \
               - w3*self.fatigue - w4*action_conc
        done = self.step_count >= 500
        return self._observe(), reward, done, dict(
            throughput=tput, error_rate=self.error_rate,
            fatigue=self.fatigue, action_conc=action_conc)

    def _observe(self):
        obs = np.array([
            self.machine_spd, self.fatigue, self.queue / 50.0,
            self.error_rate,  self.skill_level / 2.0], dtype=np.float32)
        if self.noise_std > 0:
            obs[[1, 3]] += self._rng.normal(
                0, self.noise_std, 2).astype(np.float32)
            obs[[1, 3]]  = np.clip(obs[[1, 3]], 0.0, 1.0)  # BUG 1 FIX
        return obs

# ─────────────────────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────────────────────
class D3QNetwork(nn.Module):
    def __init__(self, sd=5, ad=4, h=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(sd, h), nn.ReLU(),
            nn.Linear(h, h),  nn.ReLU())
        self.V = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,1))
        self.A = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,ad))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x); V = self.V(h); A = self.A(h)
        return V + A - A.mean(dim=1, keepdim=True)

# ─────────────────────────────────────────────────────────────
# BUFFERS
# ─────────────────────────────────────────────────────────────
class UniformBuffer:
    def __init__(self, cap=50_000): self._b = deque(maxlen=cap)
    def push(self, s,a,r,ns,d): self._b.append(Transition(s,a,r,ns,d))
    def sample(self, n): return Transition(*zip(*random.sample(self._b,n)))
    def __len__(self): return len(self._b)

class _SumTree:
    def __init__(self, cap):
        self.cap=cap; self.tree=np.zeros(2*cap-1,dtype=np.float64)
        self.data=[None]*cap; self.ptr=0; self.size=0
    def _prop(self, i, d):
        while i>0: i=(i-1)>>1; self.tree[i]+=d
    def add(self, p, x):
        l=self.ptr+self.cap-1; self.data[self.ptr]=x
        d=p-self.tree[l]; self.tree[l]=p; self._prop(l,d)
        self.ptr=(self.ptr+1)%self.cap
        self.size=min(self.size+1,self.cap)
    def update(self, l, p):
        d=p-self.tree[l]; self.tree[l]=p; self._prop(l,d)
    def get(self, s):
        i=0
        while True:
            l=2*i+1
            if l>=len(self.tree):
                return i,self.tree[i],self.data[i-self.cap+1]
            if s<=self.tree[l]: i=l
            else: s-=self.tree[l]; i=l+1
    @property
    def total(self): return self.tree[0]

class PERBuffer:
    E=1e-6
    def __init__(self,cap=50_000,alpha=0.6,b0=0.4,b1=1.0,bs=200_000):
        self.T=_SumTree(cap); self.alpha=alpha
        self.b0=b0; self.b1=b1; self.bs=bs; self._s=0; self._mp=1.0
    @property
    def beta(self): return self.b0+min(1.0,self._s/self.bs)*(self.b1-self.b0)
    def push(self,s,a,r,ns,d):
        self.T.add(self._mp**self.alpha,Transition(s,a,r,ns,d))
    def sample(self,n):
        self._s+=1; idx,pri,bat=[],[],[]
        seg=self.T.total/n
        for i in range(n):
            s=random.uniform(seg*i,seg*(i+1))
            l,p,x=self.T.get(s)
            if x is None: l,p,x=self.T.get(random.uniform(0,self.T.total))
            idx.append(l); pri.append(p); bat.append(x)
        pr=np.array(pri)/(self.T.total+self.E)
        w=(self.T.size*pr+self.E)**(-self.beta); w/=w.max()
        return Transition(*zip(*bat)),idx,torch.FloatTensor(w).to(DEVICE)
    def update(self,idx,errs):
        for i,e in zip(idx,errs):
            p=(abs(float(e))+self.E)**self.alpha
            self.T.update(i,p); self._mp=max(self._mp,p)
    def __len__(self): return self.T.size

class _NStep:
    def __init__(self,n=3,g=0.95): self.n=n; self.g=g; self._b=deque()
    def push(self,s,a,r,ns,d):
        self._b.append((s,a,r,ns,d))
        if len(self._b)<self.n and not d: return []
        return self._flush(d)
    def _flush(self,term):
        out=[]
        while self._b and (len(self._b)>=self.n or term):
            R,g=0.0,1.0; ln=None; ld=False
            for k,(_,_,r,ns,d) in enumerate(self._b):
                if k>=self.n: break
                R+=g*r; g*=self.g; ln=ns; ld=d
                if d: break
            s0,a0=self._b[0][0],self._b[0][1]
            out.append(Transition(s0,a0,R,ln,ld))
            self._b.popleft()
            if not term: break
        return out
    def drain(self):
        out=[]
        while self._b: out.extend(self._flush(True))
        return out

class EBQBuffer:
    def __init__(self,cap=50_000,dup=2,lam=0.5,thr=0.0):
        self._b=deque(maxlen=cap); self.dup=dup; self.lam=lam
        self.thr=thr; self._ep=[]; self._ret=0.0
    def push(self,s,a,r,ns,d):
        self._ep.append(Transition(s,a,r,ns,d)); self._ret+=r
        if d: self._commit()
    def _commit(self):
        if not self._ep: return
        t=self._ep[-1]
        boost=self.lam*(self._ret/max(1,len(self._ep)))
        self._ep[-1]=Transition(t.s,t.a,t.r+boost,t.ns,t.done)
        for x in self._ep:
            self._b.append(x)
            if x.r>self.thr:
                for _ in range(self.dup-1): self._b.append(x)
        self._ep.clear(); self._ret=0.0
    def sample(self,n):
        return Transition(*zip(*random.sample(self._b,min(n,len(self._b)))))
    def __len__(self): return len(self._b)

# ─────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────
class D3QNAgent:
    def __init__(self,sd=5,ad=4,lr=1e-3,gamma=0.95,
                 eps_start=1.0,eps_end=0.01,eps_decay=EPS_DECAY,
                 buf_size=50_000,batch=None,tgt_update=200,seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.ad=ad; self.gamma=gamma; self.batch=batch or BATCH_SIZE
        self.tgt_update=tgt_update; self.eps=eps_start; self.eps_end=eps_end
        self.eps_delta=(eps_start-eps_end)/eps_decay; self._step=0
        self.online=D3QNetwork(sd,ad).to(DEVICE)
        self.target=D3QNetwork(sd,ad).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt=optim.Adam(self.online.parameters(),lr=lr)
        self.buf=UniformBuffer(buf_size)

    def act(self,s,explore=True):
        if explore and random.random()<self.eps: return random.randrange(self.ad)
        with torch.no_grad():
            return self.online(
                torch.FloatTensor(s).unsqueeze(0).to(DEVICE)).argmax().item()

    def store(self,s,a,r,ns,d): self.buf.push(s,a,r,ns,d)
    def decay(self): self.eps=max(self.eps_end,self.eps-self.eps_delta)

    def _batch(self,states,actions,rewards,nstates,dones,weights=None):
        with torch.no_grad():
            ba=self.online(nstates).argmax(1,keepdim=True)
            tq=(rewards+self.gamma
                *self.target(nstates).gather(1,ba).squeeze(1)*(1-dones))
        cq=self.online(states).gather(1,actions).squeeze(1); td=tq-cq
        loss=((td**2) if weights is None else (weights*td**2)).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),1.0)
        self.opt.step(); self._step+=1
        if self._step%self.tgt_update==0:
            self.target.load_state_dict(self.online.state_dict())
        return td.detach().cpu().numpy()

    def learn(self):
        if len(self.buf)<self.batch: return
        b=self.buf.sample(self.batch)
        self._batch(
            torch.FloatTensor(np.array(b.s)).to(DEVICE),
            torch.LongTensor(b.a).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(b.r).to(DEVICE),
            torch.FloatTensor(np.array(b.ns)).to(DEVICE),
            torch.FloatTensor(b.done).to(DEVICE))

class PERnAgent(D3QNAgent):
    def __init__(self,n=3,**kw):
        super().__init__(**kw); self.gn=self.gamma**n
        self.buf=PERBuffer(cap=kw.get('buf_size',50_000))
        self._ns=_NStep(n,self.gamma)
    def store(self,s,a,r,ns,d):
        for t in self._ns.push(s,a,r,ns,d):
            self.buf.push(t.s,t.a,t.r,t.ns,t.done)
        if d:
            for t in self._ns.drain():
                self.buf.push(t.s,t.a,t.r,t.ns,t.done)
    def learn(self):
        if len(self.buf)<self.batch: return
        b,idx,w=self.buf.sample(self.batch)
        td=self._batch(
            torch.FloatTensor(np.array(b.s)).to(DEVICE),
            torch.LongTensor(b.a).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(b.r).to(DEVICE),
            torch.FloatTensor(np.array(b.ns)).to(DEVICE),
            torch.FloatTensor(b.done).to(DEVICE),weights=w)
        self.buf.update(idx,td)

class EBQAgent(D3QNAgent):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.buf=EBQBuffer(cap=kw.get('buf_size',50_000))
    def store(self,s,a,r,ns,d): self.buf.push(s,a,r,ns,d)
    def learn(self):
        if len(self.buf)<self.batch: return
        b=self.buf.sample(self.batch)
        self._batch(
            torch.FloatTensor(np.array(b.s)).to(DEVICE),
            torch.LongTensor(b.a).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(b.r).to(DEVICE),
            torch.FloatTensor(np.array(b.ns)).to(DEVICE),
            torch.FloatTensor(b.done).to(DEVICE))

class NStepAgent(D3QNAgent):
    def __init__(self,n=3,**kw):
        super().__init__(**kw); self.gn=self.gamma**n
        self._ns=_NStep(n,self.gamma)
    def store(self,s,a,r,ns,d):
        for t in self._ns.push(s,a,r,ns,d):
            self.buf.push(t.s,t.a,t.r,t.ns,t.done)
        if d:
            for t in self._ns.drain():
                self.buf.push(t.s,t.a,t.r,t.ns,t.done)
    def _batch(self,states,actions,rewards,nstates,dones,weights=None):
        with torch.no_grad():
            ba=self.online(nstates).argmax(1,keepdim=True)
            tq=(rewards+self.gn
                *self.target(nstates).gather(1,ba).squeeze(1)*(1-dones))
        cq=self.online(states).gather(1,actions).squeeze(1)
        td=tq-cq; loss=(td**2).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),1.0)
        self.opt.step(); self._step+=1
        if self._step%self.tgt_update==0:
            self.target.load_state_dict(self.online.state_dict())
        return td.detach().cpu().numpy()

class PPONet(nn.Module):
    def __init__(self,sd=5,ad=4,h=128):
        super().__init__()
        self.shared=nn.Sequential(
            nn.Linear(sd,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.actor=nn.Linear(h,ad); self.critic=nn.Linear(h,1)
    def forward(self,x):
        h=self.shared(x)
        return F.softmax(self.actor(h),dim=-1),self.critic(h)

class PPOAgent:
    def __init__(self,sd=5,ad=4,lr=3e-4,gamma=0.95,
                 clip=0.2,epochs=4,seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.gamma=gamma; self.clip=clip; self.epochs=epochs
        self.net=PPONet(sd,ad).to(DEVICE)
        self.opt=optim.Adam(self.net.parameters(),lr=lr)
        self._t=[]
    def act(self,s):
        with torch.no_grad():
            p,_=self.net(torch.FloatTensor(s).unsqueeze(0).to(DEVICE))
        d=torch.distributions.Categorical(p); a=d.sample()
        return a.item(),d.log_prob(a).item()
    def store(self,s,a,lp,r,ns,d): self._t.append((s,a,lp,r,d))
    def learn(self):
        if not self._t: return
        S,A,LP,R,D=zip(*self._t)
        S=torch.FloatTensor(np.array(S)).to(DEVICE)
        A=torch.LongTensor(A).to(DEVICE)
        LP=torch.FloatTensor(LP).to(DEVICE)
        rets,r=[],0.0
        for ri,di in zip(reversed(R),reversed(D)):
            r=ri+self.gamma*r*(1-di); rets.insert(0,r)
        rets=torch.FloatTensor(rets).to(DEVICE)
        if rets.std()>1e-8: rets=(rets-rets.mean())/(rets.std()+1e-8)
        for _ in range(self.epochs):
            p,v=self.net(S); d=torch.distributions.Categorical(p)
            lp=d.log_prob(A); ratio=torch.exp(lp-LP)
            adv=rets-v.squeeze().detach()
            loss=(-torch.min(ratio*adv,
                             torch.clamp(ratio,1-self.clip,1+self.clip)*adv).mean()
                  +0.5*F.mse_loss(v.squeeze(),rets))
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(),0.5)
            self.opt.step()
        self._t.clear()

# ─────────────────────────────────────────────────────────────
# TRAINING LOOPS  —  Bug 2 fixed
# ─────────────────────────────────────────────────────────────
def run_dqn_ep(agent, env, w=W_COMPOSITE):
    s=env.reset()
    il={k:[] for k in ('throughput','error_rate','fatigue','action_conc')}
    R=0.0; step_ctr=0
    while True:
        a=agent.act(s); ns,r,done,info=env.step(a,w)
        agent.store(s,a,r,ns,done); step_ctr+=1
        if step_ctr % LEARN_EVERY == 0: agent.learn()
        s=ns; R+=r
        for k in il: il[k].append(info[k])
        if done:
            if step_ctr % LEARN_EVERY != 0: agent.learn()  # BUG 2 FIX
            break
    agent.decay(); return R/500, il

def run_ppo_ep(agent, env, w=W_COMPOSITE):
    s=env.reset()
    il={k:[] for k in ('throughput','error_rate','fatigue','action_conc')}
    R=0.0
    while True:
        a,lp=agent.act(s); ns,r,done,info=env.step(a,w)
        agent.store(s,a,lp,r,ns,done); s=ns; R+=r
        for k in il: il[k].append(info[k])
        if done: break
    agent.learn(); return R/500, il

def _make_agent(algo, seed):
    kw=dict(seed=seed, eps_decay=ROB_EPS_DECAY)
    if   algo=='D3QN':        return D3QNAgent(**kw),   run_dqn_ep
    elif algo=='PER-n3-D3QN': return PERnAgent(**kw),   run_dqn_ep
    elif algo=='EBQ-lite':    return EBQAgent(**kw),    run_dqn_ep
    elif algo=='D3QN-NStep':  return NStepAgent(**kw),  run_dqn_ep
    elif algo=='PPO':         return PPOAgent(seed=seed), run_ppo_ep
    else: raise ValueError(algo)

def train_seed_rob(algo, seed, env_kw):
    env=CementBaggingEnv(seed=seed, **(env_kw or {}))
    agent,fn=_make_agent(algo, seed)
    hist={k:[] for k in ('throughput','error_rate','fatigue','action_conc')}
    for _ in range(ROB_EPISODES):
        _,il=fn(agent, env)
        for k in hist: hist[k].append(float(np.mean(il[k])))
    return hist

# ─────────────────────────────────────────────────────────────
# ROBUSTNESS RUNNER  —  with per-run progress + mid-run checkpoint
# ─────────────────────────────────────────────────────────────
def run_robustness(algos, ckpt, partial_rob=None):
    """
    partial_rob: existing dict to resume from (if session was interrupted).
    Saves checkpoint after every algorithm completes.
    """
    rob = partial_rob or {}
    t0  = time.time()
    total = len(algos) * len(ROB_CONDS) * N_ROB
    done  = sum(len(ROB_CONDS)*N_ROB for a in algos if a in rob)

    for algo in algos:
        if algo in rob:
            print(f"  Skipping {algo} (already in partial checkpoint)")
            continue
        rob[algo] = {}
        for cond, kw in ROB_CONDS.items():
            vals={k:[] for k in ('throughput','error_rate','fatigue')}
            for seed in range(N_ROB):
                done += 1
                print(f"  [{done:>3}/{total}] {algo} | {cond} | seed {seed+1}/{N_ROB}",
                      end='\r', flush=True)
                h=train_seed_rob(algo, seed, kw)
                for k in vals:
                    vals[k].append(float(np.mean(h[k][ROB_CONV_START:])))
            rob[algo][cond]={k:float(np.mean(v)) for k,v in vals.items()}

        # Save after each algorithm — safe to interrupt between algos
        ckpt['rob_partial'] = rob
        with open(CKPT_FILE, 'wb') as f: pickle.dump(ckpt, f)
        print(f"  Done: {algo}  ({(time.time()-t0)/60:.1f} min elapsed)",
              flush=True)

    return rob

# ─────────────────────────────────────────────────────────────
# OUTPUTS
# ─────────────────────────────────────────────────────────────
C = {'D3QN':'#2196F3','PPO':'#FF5722','PER-n3-D3QN':'#4CAF50',
     'EBQ-lite':'#9C27B0','D3QN-NStep':'#FF9800'}
plt.rcParams.update({'font.family':'serif','font.size':11,
                     'axes.titlesize':12,'axes.labelsize':11,
                     'legend.fontsize':9,'figure.dpi':150})

def make_fig5(results, rob):
    algos=list(results.keys()); conds=list(ROB_CONDS.keys())
    mat=np.array([
        [(rob[a][c]['error_rate']-rob[a]['Clean baseline']['error_rate'])
         /(rob[a]['Clean baseline']['error_rate']+1e-10)*100
         for c in conds] for a in algos])
    f,ax=plt.subplots(figsize=(14,5.5))
    im=ax.imshow(mat,cmap='RdYlGn_r',aspect='auto',vmin=-20,vmax=70)
    plt.colorbar(im,ax=ax,label='Error Rate Change vs. Clean Baseline (%)')
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds,rotation=22,ha='right',fontsize=10)
    ax.set_yticks(range(len(algos))); ax.set_yticklabels(algos,fontsize=11)
    ax.set_title(
        f'Robustness Testing: Error Rate Degradation (%) — Lower = More Robust\n'
        f'({N_ROB} seeds × {ROB_EPISODES} ep per condition, '
        f'convergence zone ep {ROB_CONV_START+1}–{ROB_EPISODES})',
        fontweight='bold', fontsize=12)
    for i in range(len(algos)):
        for j in range(len(conds)):
            v=mat[i,j]; st='PASS' if v<=15 else 'FAIL'
            ax.text(j,i,f'{v:+.0f}%\n{st}',ha='center',va='center',
                    fontsize=9,fontweight='bold',
                    color='white' if abs(v)>55 else 'black')
    ax.text(0.01,0.97,'PASS ≤ +15%',transform=ax.transAxes,
            fontsize=8.5,va='top',color='darkgreen',fontweight='bold')
    ax.text(0.01,0.90,'FAIL > +15%',transform=ax.transAxes,
            fontsize=8.5,va='top',color='darkred',fontweight='bold')
    plt.tight_layout()
    p=f'{OUTPUT_DIR}/fig5_robustness_heatmap_FINAL.png'
    plt.savefig(p,bbox_inches='tight',dpi=150); plt.close()
    print(f"  Saved: {p}")

def build_table5(rob):
    rows=[]; base_err=rob['D3QN']['Clean baseline']['error_rate']
    for cond in ROB_CONDS:
        r=rob['D3QN'][cond]
        dg=(r['error_rate']-base_err)/(base_err+1e-10)*100
        rows.append({
            'Condition':             cond,
            'Throughput':            round(r['throughput'],3),
            'Error Rate':            round(r['error_rate'], 3),
            'Fatigue Index':         round(r['fatigue'],    3),
            'Error Degradation (%)': round(dg, 1),
            'Status': '✓ PASS' if dg<=15 else '✗ FAIL'})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*62}")
    print(f" Robustness Re-run FINAL")
    print(f" N_ROB={N_ROB} seeds × ROB_EPISODES={ROB_EPISODES} ep")
    print(f" Convergence zone: ep {ROB_CONV_START+1}–{ROB_EPISODES}")
    print(f" ROB_EPS_DECAY={ROB_EPS_DECAY}  |  Both bugs fixed")
    print(f"{'='*62}")

    print(f"\nLoading checkpoint from {CKPT_FILE} ...")
    with open(CKPT_FILE,'rb') as f: ckpt=pickle.load(f)
    results = ckpt['results']
    ALGOS   = list(results.keys())
    print(f"  Loaded algorithms: {ALGOS}")

    # Resume partial rob if session was interrupted mid-run
    partial = ckpt.get('rob_partial', None)
    if partial:
        done_algos = list(partial.keys())
        print(f"  Resuming from partial checkpoint — done: {done_algos}")
    else:
        print("  No partial checkpoint — starting fresh")

    total_runs = N_ROB * len(ROB_CONDS) * len(ALGOS)
    print(f"\nRunning {total_runs} total training runs ...")
    print(f"Estimated time: ~{total_runs*1.25/60:.0f} min on CPU "
          f"/ ~{total_runs*0.20/60:.0f} min on GPU\n")

    t0 = time.time()
    rob_fixed = run_robustness(ALGOS, ckpt, partial_rob=partial)
    elapsed   = (time.time()-t0)/60
    print(f"\nRobustness complete in {elapsed:.1f} min")

    # Commit to checkpoint (replace rob, remove rob_partial)
    ckpt['rob'] = rob_fixed
    ckpt.pop('rob_partial', None)
    with open(CKPT_FILE,'wb') as f: pickle.dump(ckpt,f)
    print("Checkpoint updated (rob_partial cleared)")

    # Build and save outputs
    print("\nGenerating Table 5 and Fig 5 ...")
    t5 = build_table5(rob_fixed)
    make_fig5(results, rob_fixed)
    t5.to_csv(f'{OUTPUT_DIR}/table5_robustness_d3qn.csv', index=False)
    print(f"  Saved: table5_robustness_d3qn.csv")

    # Update Excel — replace only Table5 sheet
    xlsx = f'{OUTPUT_DIR}/HRC_Results_ALL_TABLES.xlsx'
    if os.path.exists(xlsx):
        with pd.ExcelWriter(xlsx, engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as w:
            t5.to_excel(w, sheet_name='Table5_D3QN_Robust', index=False)
        print(f"  Updated: {xlsx} (Table5 sheet replaced)")

    # Console summary
    print(f"\n{'='*62}")
    print("FIXED TABLE 5 — paste to Claude:")
    print(f"{'='*62}")
    print(t5.to_string(index=False))
    print(f"\nAll other results (Tables 3,4,6,7 + Figs 1–4) unchanged.")
    print(f"{'='*62}")

if __name__=='__main__':
    main()
