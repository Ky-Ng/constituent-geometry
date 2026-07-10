import sys, os, random, importlib, time
from collections import Counter
sys.path.insert(0, os.path.abspath("experiments/22_rasp_l_hi_hf"))
sys.path.insert(0, os.path.abspath("src"))
import grammar.v2.generate_with_frames as G
from grammar.v2.generate_with_frames import sample_pair_from_frame
from grammar_oracle import oracle_features
MODNAMES=["_s2_closer","_s2_interval","_s2_gaplca","_s2_twolevel"]
MODS={m:getattr(importlib.import_module(m),"hi_to_hf") for m in MODNAMES}
DPc=G.FrameDPCommon; DPp=G.FrameDPProper
Bar=G.FrameNPsBar; Adj=G.FrameNPsAdj; Rel=G.FrameNPsRel
CPrel=G.FrameCPrel; SubjGap=G.FrameSsubjGap; ObjGap=G.FrameSobjGap; VObjGap=G.FrameVPobjGap
Vin=G.FrameVPIntrans; VinA=G.FrameVPIntransAdv
Vdp=G.FrameVPdp; VdpA=G.FrameVPdpAdv; Vcp=G.FrameVPcp; VcpA=G.FrameVPcpAdv
CPsent=G.FrameCPsent; S=G.FrameS
def g_nps(r,b):
    opts=[lambda:Bar(), lambda:Adj()]
    if b>=1:
        core=r.choice([Bar,Adj])()
        opts.append(lambda core=core:Rel(core, g_cprel(r,b)))
    return r.choice(opts)()
def g_dp(r,b):
    if r.random()<0.35: return DPp()
    return DPc(g_nps(r,b))
def g_cprel(r,b):
    if r.random()<0.5: return CPrel(SubjGap(g_vp(r,b-1)))
    return CPrel(ObjGap(g_dp(r,b-1), VObjGap()))
def g_cpsent(r,b): return CPsent(g_s(r,b-1))
def g_vp(r,b):
    opts=[lambda:Vin(), lambda:VinA(), lambda:Vdp(g_dp(r,b)), lambda:VdpA(g_dp(r,b))]
    if b>=1:
        opts.append(lambda:Vcp(g_cpsent(r,b))); opts.append(lambda:VcpA(g_cpsent(r,b)))
    return r.choice(opts)()
def g_s(r,b): return S(g_dp(r,b), g_vp(r,b))
N=int(sys.argv[1]); seed0=int(sys.argv[2])
r=random.Random(seed0)
stats={m:[0,0] for m in MODNAMES}; fails={m:[] for m in MODNAMES}
bydepth=Counter(); oracle_bad=0; lenmax=0
t0=time.time()
for i in range(N):
    f=g_s(r,2); d=f.depth(); bydepth[d]+=1
    p=sample_pair_from_frame(f, random.Random(r.random()*1e9))
    lenmax=max(lenmax,len(p.hi_tokens))
    hf=p.hf_tokens; of=oracle_features(p.hi_tokens)
    if of["hf_tokens"]!=hf: oracle_bad+=1
    for m in MODNAMES:
        try: got=MODS[m](p.hi_tokens)
        except Exception as e: got=("EXC",repr(e))
        stats[m][1]+=1
        if got==hf: stats[m][0]+=1
        else: fails[m].append((d,p.hi_tokens,hf,got))
print(f"N={N} seed={seed0} secs={time.time()-t0:.1f} depths={dict(bydepth)} maxlen={lenmax} oracle_bad={oracle_bad}")
for m in MODNAMES:
    c,t=stats[m]; print(f"{m}: {c}/{t} = {c/t:.6f}  fails={t-c}")
    fails[m].sort(key=lambda x: len(x[1]))
    for (d,hi,hf,got) in fails[m][:3]:
        print("   FAIL d",d,"len",len(hi),"| HI:"," ".join(hi))
        print("        exp:"," ".join(hf)); print("        got:"," ".join(got) if isinstance(got,list) else got)
