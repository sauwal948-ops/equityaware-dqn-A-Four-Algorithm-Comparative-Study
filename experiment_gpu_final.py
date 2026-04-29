"""
==========================================================================
 HRC Experiment — GPU VERSION, persistent checkpoint
 
 BEFORE RUNNING:
   1. In your Kaggle notebook: Settings → Accelerator → GPU T4 x2
   2. Make sure "Internet" is ON in Settings
   3. Paste this entire script into ONE cell and click Run
 
 Estimated runtime:
   CPU  : 8-12 hours  (too slow — use GPU)
   GPU  : 60-90 minutes
 
 Checkpoint survives session restart IF you save output dataset.
 To do this: after first successful algo completes, click
 "Save Version" in Kaggle → this commits /kaggle/working to storage.
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

# ── GPU setup ────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    BATCH_SIZE = 256   # larger batch on GPU
else:
    print("WARNING: No GPU found. This will be very slow on CPU.")
    BATCH_SIZE = 64

OUTPUT_DIR = '/kaggle/working'
CKPT_FILE  = f'{OUTPUT_DIR}/hrc_checkpoint.pkl'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPISODES   = 200
CONV_START = 140    # convergence zone: ep 141-200
N_SEEDS    = 20
N_ROB      = 5

# ─────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────
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
    print("  [no checkpoint found — starting fresh]")
    return {}

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
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
        eff = action
        if self.downtime_range[1] > 0:
            if self._rng.random() < self._rng.uniform(*self.downtime_range) and action in (1,2):
                eff = 0
        load = self.LOAD[eff]; dt = 0.1
        self.fatigue = float(np.clip(
            self.fatigue + dt*(self.alpha*load - self.BETA*self.fatigue), 0.0, 1.0))
        if eff == 3: self.fatigue = max(0.0, self.fatigue - 0.05)
        arr  = self._rng.poisson(1.2*dt)
        sr   = self.machine_spd * (0.7 if eff==2 else 1.0) * (0.5 if eff==3 else 1.0)
        self.queue = float(np.clip(self.queue + arr - sr, 0, 50))
        tput = sr / 1.2
        self.error_rate = float(np.clip(
            self.error_rate + (0.02*(self.fatigue-0.2) + self._rng.normal(0,0.005))*dt,
            0.005, 0.50))
        if eff==1: self.error_rate *= 0.995
        elif eff==2: self.error_rate *= 0.990
        self.machine_spd = float(np.clip(
            self.machine_spd + self._rng.normal(0,0.01), 0.3, 1.2))
        total = max(1, self.action_hist.sum())
        bias  = float(np.var(self.action_hist/total)*4.0)
        w1,w2,w3,w4 = weights
        reward = w1*tput - w2*self.error_rate - w3*self.fatigue - w4*bias
        done   = self.step_count >= 500
        return self._observe(), reward, done, \
               dict(throughput=tput, error_rate=self.error_rate,
                    fatigue=self.fatigue, bias=bias)

    def _observe(self):
        obs = np.array([self.machine_spd, self.fatigue, self.queue/50.0,
                        self.error_rate, self.skill_level/2.0], dtype=np.float32)
        if self.noise_std > 0:
            obs[[1,3]] += self._rng.normal(0, self.noise_std, 2).astype(np.float32)
            obs = np.clip(obs, 0.0, 1.0)
        return obs

# ─────────────────────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────────────────────
class D3QNetwork(nn.Module):
    def __init__(self, sd=5, ad=4, h=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(sd,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU())
        self.V = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,1))
        self.A = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,ad))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        h=self.shared(x); V=self.V(h); A=self.A(h)
        return V + A - A.mean(dim=1, keepdim=True)

# ─────────────────────────────────────────────────────────────
# BUFFERS
# ─────────────────────────────────────────────────────────────
Transition = namedtuple('Transition', ['s','a','r','ns','done'])

class UniformBuffer:
    def __init__(self, cap=50_000): self._b = deque(maxlen=cap)
    def push(self,s,a,r,ns,d): self._b.append(Transition(s,a,r,ns,d))
    def sample(self,n): return Transition(*zip(*random.sample(self._b,n)))
    def __len__(self): return len(self._b)

class _SumTree:
    def __init__(self,cap):
        self.cap=cap; self.tree=np.zeros(2*cap-1,dtype=np.float64)
        self.data=[None]*cap; self.ptr=0; self.size=0
    def _prop(self,i,d):
        while i>0: i=(i-1)>>1; self.tree[i]+=d
    def add(self,p,x):
        l=self.ptr+self.cap-1; self.data[self.ptr]=x
        d=p-self.tree[l]; self.tree[l]=p; self._prop(l,d)
        self.ptr=(self.ptr+1)%self.cap; self.size=min(self.size+1,self.cap)
    def update(self,l,p):
        d=p-self.tree[l]; self.tree[l]=p; self._prop(l,d)
    def get(self,s):
        i=0
        while True:
            l=2*i+1
            if l>=len(self.tree): return i,self.tree[i],self.data[i-self.cap+1]
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
    def push(self,s,a,r,ns,d): self.T.add(self._mp**self.alpha,Transition(s,a,r,ns,d))
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
            R,g,ln,ld=0.0,1.0,None,False
            for _,_,r,ns,d in self._b:
                R+=g*r; g*=self.g; ln=ns; ld=d
                if d: term=True; break
            s0,a0=self._b[0][0],self._b[0][1]
            out.append(Transition(s0,a0,R,ln,ld)); self._b.popleft()
            if not term: break
        return out
    def drain(self):
        o=[]
        while self._b: o.extend(self._flush(True))
        return o

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
        self._ep[-1]=Transition(t.s,t.a,t.r+self.lam*(self._ret/max(1,len(self._ep))),t.ns,t.done)
        for x in self._ep:
            self._b.append(x)
            if x.r>self.thr:
                for _ in range(self.dup-1): self._b.append(x)
        self._ep.clear(); self._ret=0.0
    def sample(self,n): return Transition(*zip(*random.sample(self._b,min(n,len(self._b)))))
    def __len__(self): return len(self._b)

# ─────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────
class D3QNAgent:
    def __init__(self,sd=5,ad=4,lr=1e-3,gamma=0.95,
                 eps_start=1.0,eps_end=0.01,eps_decay=100,
                 buf_size=50_000,batch=None,tgt_update=200,seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.ad=ad; self.gamma=gamma
        self.batch=batch or BATCH_SIZE
        self.tgt_update=tgt_update
        self.eps=eps_start; self.eps_end=eps_end
        self.eps_delta=(eps_start-eps_end)/eps_decay
        self._step=0
        self.online=D3QNetwork(sd,ad).to(DEVICE)
        self.target=D3QNetwork(sd,ad).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt=optim.Adam(self.online.parameters(),lr=lr)
        self.buf=UniformBuffer(buf_size)

    def act(self,s,explore=True):
        if explore and random.random()<self.eps: return random.randrange(self.ad)
        with torch.no_grad():
            return self.online(torch.FloatTensor(s).unsqueeze(0).to(DEVICE)).argmax().item()

    def store(self,s,a,r,ns,d): self.buf.push(s,a,r,ns,d)
    def decay(self): self.eps=max(self.eps_end,self.eps-self.eps_delta)

    def _batch(self,states,actions,rewards,nstates,dones,weights=None):
        with torch.no_grad():
            ba=self.online(nstates).argmax(1,keepdim=True)
            tq=rewards+self.gamma*self.target(nstates).gather(1,ba).squeeze(1)*(1-dones)
        cq=self.online(states).gather(1,actions).squeeze(1)
        td=tq-cq
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
        super().__init__(**kw)
        self.gn=self.gamma**n
        self.buf=PERBuffer(cap=kw.get('buf_size',50_000))
        self._ns=_NStep(n,self.gamma)
    def store(self,s,a,r,ns,d):
        for t in self._ns.push(s,a,r,ns,d): self.buf.push(t.s,t.a,t.r,t.ns,t.done)
        if d:
            for t in self._ns.drain(): self.buf.push(t.s,t.a,t.r,t.ns,t.done)
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
        super().__init__(**kw); self.gn=self.gamma**n; self._ns=_NStep(n,self.gamma)
    def store(self,s,a,r,ns,d):
        for t in self._ns.push(s,a,r,ns,d): self.buf.push(t.s,t.a,t.r,t.ns,t.done)
        if d:
            for t in self._ns.drain(): self.buf.push(t.s,t.a,t.r,t.ns,t.done)
    def _batch(self,states,actions,rewards,nstates,dones,weights=None):
        with torch.no_grad():
            ba=self.online(nstates).argmax(1,keepdim=True)
            tq=rewards+self.gn*self.target(nstates).gather(1,ba).squeeze(1)*(1-dones)
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
        self.shared=nn.Sequential(nn.Linear(sd,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.actor=nn.Linear(h,ad); self.critic=nn.Linear(h,1)
    def forward(self,x):
        h=self.shared(x); return F.softmax(self.actor(h),dim=-1),self.critic(h)

class PPOAgent:
    def __init__(self,sd=5,ad=4,lr=3e-4,gamma=0.95,clip=0.2,epochs=4,seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.gamma=gamma; self.clip=clip; self.epochs=epochs
        self.net=PPONet(sd,ad).to(DEVICE)
        self.opt=optim.Adam(self.net.parameters(),lr=lr); self._t=[]
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
        A=torch.LongTensor(A).to(DEVICE); LP=torch.FloatTensor(LP).to(DEVICE)
        rets,r=[], 0.0
        for ri,di in zip(reversed(R),reversed(D)):
            r=ri+self.gamma*r*(1-di); rets.insert(0,r)
        rets=torch.FloatTensor(rets).to(DEVICE)
        if rets.std()>1e-8: rets=(rets-rets.mean())/(rets.std()+1e-8)
        for _ in range(self.epochs):
            p,v=self.net(S); d=torch.distributions.Categorical(p)
            lp=d.log_prob(A); ratio=torch.exp(lp-LP)
            adv=rets-v.squeeze().detach()
            loss=(-torch.min(ratio*adv,torch.clamp(ratio,1-self.clip,1+self.clip)*adv).mean()
                  +0.5*F.mse_loss(v.squeeze(),rets))
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(),0.5); self.opt.step()
        self._t.clear()

# ─────────────────────────────────────────────────────────────
# TRAINING LOOPS
# ─────────────────────────────────────────────────────────────
W=(0.5,0.3,0.1,0.1)

def run_dqn_ep(agent,env):
    s=env.reset(); info_lists={k:[] for k in ('throughput','error_rate','fatigue','bias')}; R=0.0
    while True:
        a=agent.act(s); ns,r,done,info=env.step(a,W)
        agent.store(s,a,r,ns,done); agent.learn(); s=ns; R+=r
        for k in info_lists: info_lists[k].append(info[k])
        if done: break
    agent.decay(); return R/500, info_lists

def run_ppo_ep(agent,env):
    s=env.reset(); info_lists={k:[] for k in ('throughput','error_rate','fatigue','bias')}; R=0.0
    while True:
        a,lp=agent.act(s); ns,r,done,info=env.step(a,W)
        agent.store(s,a,lp,r,ns,done); s=ns; R+=r
        for k in info_lists: info_lists[k].append(info[k])
        if done: break
    agent.learn(); return R/500, info_lists

def train_seed(algo,seed,env_kw=None,episodes=EPISODES):
    env=CementBaggingEnv(seed=seed,**(env_kw or {}))
    if   algo=='D3QN':        ag=D3QNAgent(seed=seed,eps_decay=100);             ep_fn=lambda:run_dqn_ep(ag,env)
    elif algo=='PER-n3-D3QN': ag=PERnAgent(seed=seed,eps_decay=100);             ep_fn=lambda:run_dqn_ep(ag,env)
    elif algo=='EBQ-lite':    ag=EBQAgent(seed=seed,eps_decay=100);              ep_fn=lambda:run_dqn_ep(ag,env)
    elif algo=='D3QN-NStep':  ag=NStepAgent(seed=seed,eps_decay=100);            ep_fn=lambda:run_dqn_ep(ag,env)
    elif algo=='PPO':         ag=PPOAgent(seed=seed);                            ep_fn=lambda:run_ppo_ep(ag,env)
    else: raise ValueError(algo)
    hist={k:[] for k in ('episode_reward','throughput','error_rate','fatigue','bias')}
    for _ in range(episodes):
        er,il=ep_fn()
        hist['episode_reward'].append(er)
        for k in ('throughput','error_rate','fatigue','bias'):
            hist[k].append(float(np.mean(il[k])))
    return hist

def run_algo(algo,n_seeds=N_SEEDS,env_kw=None,episodes=EPISODES):
    t0=time.time()
    print(f"  {algo} ({n_seeds} seeds × {episodes} ep)...")
    hists=[]
    for seed in range(n_seeds):
        print(f"    seed {seed+1:02d}/{n_seeds}",end='\r')
        hists.append(train_seed(algo,seed,env_kw,episodes))
    elapsed=time.time()-t0
    print(f"    {algo} DONE in {elapsed/60:.1f} min           ")
    agg={}
    for k in ('episode_reward','throughput','error_rate','fatigue','bias'):
        mat=np.array([h[k] for h in hists])
        agg[k]=dict(mean=mat.mean(0),std=mat.std(0),per_seed=mat,
                    final_mean=mat[:,CONV_START:].mean(),
                    final_std=mat[:,CONV_START:].std(),
                    per_seed_final=mat[:,CONV_START:].mean(1))
    return agg

# ─────────────────────────────────────────────────────────────
# ROBUSTNESS
# ─────────────────────────────────────────────────────────────
ROB_CONDS={
    'Clean baseline':      dict(noise_std=0.00,downtime_range=(0.00,0.00)),
    'Sensor noise 10%':    dict(noise_std=0.10,downtime_range=(0.00,0.00)),
    'Sensor noise 20%':    dict(noise_std=0.20,downtime_range=(0.00,0.00)),
    'Downtime 5-20%':      dict(noise_std=0.00,downtime_range=(0.05,0.20)),
    'Compound (noise+DT)': dict(noise_std=0.10,downtime_range=(0.05,0.20)),
}

def robustness(algos,n_seeds=N_ROB,episodes=EPISODES):
    rob={}
    for algo in algos:
        rob[algo]={}
        for cond,kw in ROB_CONDS.items():
            vals={k:[] for k in ('throughput','error_rate','fatigue')}
            for seed in range(n_seeds):
                h=train_seed(algo,seed,kw,episodes)
                for k in vals: vals[k].append(np.mean(h[k][CONV_START:]))
            rob[algo][cond]={k:np.mean(v) for k,v in vals.items()}
        print(f"    Robustness done: {algo}")
    return rob

# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────
def paired_stats(a,b):
    t,p=stats.ttest_rel(a,b); d=(np.array(a)-np.array(b))
    return t,p,d.mean()/(d.std(ddof=1)+1e-10)

def stats_table(results,base='D3QN'):
    rows=[]
    for algo in [k for k in results if k!=base]:
        for m in ('throughput','error_rate','fatigue','bias'):
            a=results[base][m]['per_seed_final']
            b=results[algo][m]['per_seed_final']
            t,p,d=paired_stats(a,b)
            rows.append({'Comparison':f'{base} vs {algo}','Metric':m,
                         f'{base}':round(float(a.mean()),4),
                         f'{algo}':round(float(b.mean()),4),
                         'Change_%':round((b.mean()-a.mean())/(a.mean()+1e-10)*100,1),
                         'p_value':'<0.001' if p<0.001 else round(p,4),
                         'Cohen_d':round(d,3),'Sig':'Yes' if p<0.05 else 'No'})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# FIGURES  (from real training data)
# ─────────────────────────────────────────────────────────────
C={'D3QN':'#2196F3','PPO':'#FF5722','PER-n3-D3QN':'#4CAF50',
   'EBQ-lite':'#9C27B0','D3QN-NStep':'#FF9800'}
LS={'D3QN':'-','PPO':'--','PER-n3-D3QN':'-.','EBQ-lite':':','D3QN-NStep':(0,(5,1))}
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.titlesize':12,
                     'axes.labelsize':11,'legend.fontsize':9,'figure.dpi':150})

def savefig(name):
    p=f'{OUTPUT_DIR}/{name}'
    plt.savefig(p,bbox_inches='tight',dpi=150); plt.close()
    print(f"  Saved: {p}")

def make_fig1(R):
    ep=np.arange(1,EPISODES+1)
    f,ax=plt.subplots(1,3,figsize=(15,5))
    f.suptitle('Fig. 1  Reward Function Ablation Study',fontweight='bold',fontsize=13)
    for algo,ls,lab in [('D3QN','-','Equity D3QN'),('PPO','--','PPO')]:
        ax[0].plot(ep,R[algo]['episode_reward']['mean'],color=C[algo],ls=ls,lw=2,label=lab)
        ax[1].plot(ep,R[algo]['fatigue']['mean'],color=C[algo],ls=ls,lw=2,label=lab)
    for a in ax[:2]:
        a.axvline(CONV_START,color='gray',ls=':',lw=1.2); a.grid(True,alpha=0.25); a.legend(fontsize=8.5)
    ax[1].fill_between(ep,
        np.clip(R['D3QN']['fatigue']['mean']-R['D3QN']['fatigue']['std'],0,1),
        R['D3QN']['fatigue']['mean']+R['D3QN']['fatigue']['std'],
        color=C['D3QN'],alpha=0.15)
    ax[0].set(xlabel='Episode',ylabel='Episode Reward (norm.)',title='(a) Training Convergence')
    ax[1].set(xlabel='Episode',ylabel='Fatigue Index',title='(b) Worker Fatigue Index')
    for algo,lab in [('D3QN','Equity D3QN'),('PPO','PPO')]:
        box=R[algo]['episode_reward']['per_seed'][:,CONV_START:].mean(1)
        ax[2].boxplot([box],positions=[['D3QN','PPO'].index(algo)],
                      tick_labels=[lab],patch_artist=True,
                      medianprops=dict(color='black',lw=2),
                      boxprops=dict(facecolor=C[algo],alpha=0.75))
    ax[2].set(ylabel=f'Final Reward (Eps {CONV_START+1}-{EPISODES}, 20 seeds)',
              title='(c) Final Reward Distribution'); ax[2].grid(True,axis='y',alpha=0.25)
    plt.tight_layout(); savefig('fig1_ablation_FINAL.png')

def make_fig2(R):
    ep=np.arange(1,EPISODES+1); algos=list(R.keys())
    panels=[('throughput','Normalized Throughput','(a)'),
            ('error_rate','Error Rate','(b)'),
            ('fatigue','Fatigue Index','(c)'),
            ('episode_reward','Episode Reward (norm.)','(d)')]
    f,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    f.suptitle(f'Fig. 2  Training Dynamics — All Algorithms (20 Seeds, ±1 SD)',
               fontweight='bold',fontsize=13)
    for ax,(m,yl,tag) in zip(axes.flat,panels):
        for algo in algos:
            mu=R[algo][m]['mean']; sd=R[algo][m]['std']
            ls=LS[algo] if isinstance(LS[algo],str) else '-'
            ax.plot(ep,mu,color=C[algo],ls=ls,lw=2.2 if algo=='D3QN' else 1.5,
                    label=algo,zorder=6 if algo=='D3QN' else 3)
            ax.fill_between(ep,mu-sd,mu+sd,color=C[algo],alpha=0.08,zorder=1)
        ax.axvline(CONV_START,color='gray',ls=':',lw=1.2,alpha=0.7)
        ax.set(ylabel=yl,title=f'{tag} {yl}'); ax.grid(True,alpha=0.25); ax.legend(fontsize=8.5)
    for ax in axes[1]: ax.set_xlabel('Episode')
    plt.tight_layout(); savefig('fig2_training_curves_FINAL.png')

def make_fig3(R):
    algos=list(R.keys())
    f,axes=plt.subplots(1,3,figsize=(16,5.5))
    f.suptitle(f'Fig. 3  Final Performance (Episodes {CONV_START+1}-{EPISODES}, Mean ± SD, 20 Seeds)',
               fontweight='bold',fontsize=13)
    x=np.arange(len(algos)); w=0.6
    for ax,(m,lab) in zip(axes,[('throughput','Normalized Throughput'),
                                  ('error_rate','Error Rate'),('fatigue','Fatigue Index')]):
        mn=[R[a][m]['final_mean'] for a in algos]
        sd=[R[a][m]['final_std']  for a in algos]
        bars=ax.bar(x,mn,w,yerr=sd,capsize=5,color=[C[a] for a in algos],
                    alpha=0.85,edgecolor='black',lw=0.8,
                    error_kw=dict(elinewidth=1.5,capthick=1.5))
        for bar,m2,s in zip(bars,mn,sd):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+s+max(sd)*0.04,
                    f'{m2:.3f}',ha='center',va='bottom',fontsize=8.5,fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(algos,rotation=28,ha='right',fontsize=9)
        ax.set(title=lab); ax.grid(True,axis='y',alpha=0.25); ax.set_ylim(bottom=0)
    plt.tight_layout(); savefig('fig3_performance_bars_FINAL.png')

def make_fig4(R,rob):
    algos=list(R.keys()); conds=list(ROB_CONDS.keys())
    mat=np.array([[(rob[a][c]['error_rate']-rob[a]['Clean baseline']['error_rate'])
                   /(rob[a]['Clean baseline']['error_rate']+1e-10)*100
                   for c in conds] for a in algos])
    f,ax=plt.subplots(figsize=(14,5.5))
    im=ax.imshow(mat,cmap='RdYlGn_r',aspect='auto',vmin=-30,vmax=50)
    plt.colorbar(im,ax=ax,label='Error Rate Change vs. Clean Baseline (%)')
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds,rotation=22,ha='right',fontsize=10)
    ax.set_yticks(range(len(algos))); ax.set_yticklabels(algos,fontsize=11)
    ax.set_title('Fig. 4  Robustness Testing: Error Rate Degradation (%) — Lower = More Robust',
                 fontweight='bold',fontsize=12)
    for i in range(len(algos)):
        for j in range(len(conds)):
            v=mat[i,j]; st=' PASS' if v<=15 else ' FAIL'
            ax.text(j,i,f'{v:+.0f}%\n{st}',ha='center',va='center',
                    fontsize=9,fontweight='bold',color='white' if abs(v)>40 else 'black')
    ax.text(0.01,0.97,'PASS = <=+15%',transform=ax.transAxes,fontsize=8.5,
            va='top',color='darkgreen',fontweight='bold')
    ax.text(0.01,0.90,'FAIL = >+15%',transform=ax.transAxes,fontsize=8.5,
            va='top',color='darkred',fontweight='bold')
    plt.tight_layout(); savefig('fig4_robustness_heatmap_FINAL.png')

def make_fig5(R):
    algos=list(R.keys())
    labels=['Error\nControl (↑)','Fatigue\nControl (↑)',
            'Throughput (↑)','Action\nConc. (↑)','Convergence\nSpeed (↑)']
    N=len(labels); ang=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); ang+=ang[:1]
    raw={}
    for a in algos:
        ac=R[a]['bias']['final_mean']
        es=R[a]['episode_reward']['std'][:30].mean()
        raw[a]=[1-R[a]['error_rate']['final_mean'],
                1-R[a]['fatigue']['final_mean'],
                R[a]['throughput']['final_mean'],ac,1/(1+es*8)]
    ac_v=[raw[a][3] for a in algos]; acmn,acmx=min(ac_v),max(ac_v)
    for a in algos: raw[a][3]=(raw[a][3]-acmn)/(acmx-acmn+1e-10)
    f,ax=plt.subplots(figsize=(9,9),subplot_kw=dict(polar=True))
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels,fontsize=10.5)
    ax.set_ylim(0,1); ax.set_yticks([.2,.4,.6,.8,1.0])
    ax.set_title('Fig. 5  Multi-Metric Performance Radar\n(all axes normalised; higher = better)',
                 fontweight='bold',pad=30,fontsize=11)
    for a in algos:
        v=raw[a]+[raw[a][0]]; ls=LS[a] if isinstance(LS[a],str) else '-'
        ax.plot(ang,v,color=C[a],lw=2.5 if a=='D3QN' else 1.8,label=a,ls=ls,
                zorder=6 if a=='D3QN' else 3)
        ax.fill(ang,v,color=C[a],alpha=0.07)
    ax.legend(loc='upper right',bbox_to_anchor=(1.45,1.15),fontsize=10)
    plt.tight_layout(); savefig('fig5_radar_chart_FINAL.png')

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    ALGOS=['D3QN','PPO','PER-n3-D3QN','EBQ-lite','D3QN-NStep']
    t_total=time.time()

    ckpt=load_ckpt(); results=ckpt.get('results',{}); rob=ckpt.get('rob',None)

    print(f"\n{'='*65}")
    print(f" HRC Experiment  |  device={DEVICE}  |  batch={BATCH_SIZE}")
    print(f" {N_SEEDS} seeds x {EPISODES} episodes  |  conv zone ep {CONV_START+1}-{EPISODES}")
    print(f"{'='*65}")

    print(f"\n[1/4] Main training...")
    for algo in ALGOS:
        if algo in results: print(f"  Skipping {algo} (checkpoint)"); continue
        results[algo]=run_algo(algo)
        save_ckpt({'results':results,'rob':rob})

    print(f"\n[2/4] Robustness testing...")
    if rob is None:
        rob=robustness(ALGOS); save_ckpt({'results':results,'rob':rob})
    else:
        print("  Skipping (checkpoint)")

    print(f"\n[3/4] Statistics...")
    sdf=stats_table(results)

    print(f"\n[4/4] Figures...")
    make_fig1(results); make_fig2(results); make_fig3(results)
    make_fig4(results,rob); make_fig5(results)

    print("\nExporting Excel + CSV...")
    algos=list(results.keys())
    perf=pd.DataFrame([{'Algorithm':a,
        **{f'{m}_mean':round(results[a][m]['final_mean'],4) for m in ('throughput','error_rate','fatigue','bias')},
        **{f'{m}_sd':  round(results[a][m]['final_std'],4)  for m in ('throughput','error_rate','fatigue','bias')}}
        for a in algos])
    rob_rows=[]
    for a in algos:
        for c in ROB_CONDS:
            r=rob[a][c]; b=rob[a]['Clean baseline']['error_rate']
            dg=(r['error_rate']-b)/(b+1e-10)*100
            rob_rows.append({'Algorithm':a,'Condition':c,
                'Throughput':round(r['throughput'],4),'Error_Rate':round(r['error_rate'],4),
                'Fatigue':round(r['fatigue'],4),'Degrad_%':round(dg,1),
                'Status':'PASS' if dg<=15 else 'FAIL'})
    rdf=pd.DataFrame(rob_rows)
    with pd.ExcelWriter(f'{OUTPUT_DIR}/results_200ep.xlsx',engine='openpyxl') as w:
        perf.to_excel(w,sheet_name='Performance',index=False)
        sdf.to_excel(w,sheet_name='Statistics',index=False)
        rdf.to_excel(w,sheet_name='Robustness',index=False)
    perf.to_csv(f'{OUTPUT_DIR}/performance_200ep.csv',index=False)
    sdf.to_csv(f'{OUTPUT_DIR}/statistics_200ep.csv',index=False)
    rdf.to_csv(f'{OUTPUT_DIR}/robustness_200ep.csv',index=False)

    print(f"\nTotal time: {(time.time()-t_total)/60:.0f} min")
    print(f"\n{'='*65}")
    print("PASTE THIS TO CLAUDE")
    print(f"{'='*65}")
    print(f"\nMEANS (ep {CONV_START+1}-{EPISODES}, 20 seeds):")
    for a in ALGOS:
        print(f"\n{a}:")
        for m in ('throughput','error_rate','fatigue','bias'):
            print(f"  {m}: {results[a][m]['final_mean']:.4f} +/- {results[a][m]['final_std']:.4f}")
    print("\nSTATS TABLE:")
    print(sdf.to_string(index=False))
    print("\nROBUSTNESS:")
    print(rdf[['Algorithm','Condition','Error_Rate','Degrad_%','Status']].to_string(index=False))

if __name__=='__main__':
    main()
