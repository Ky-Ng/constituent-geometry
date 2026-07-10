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
LMAX=int(sys.argv[1]); K=int(sys.argv[2]); ONLY_D2=int(sys.argv[3])
def E_nps(b):
    res=[(Bar(),1),(Adj(),2)]
    if b>=1:
        for cpf,cc in E_cprel(b):
            for coref,cs in [(Bar,1),(Adj,2)]:
                if cs+cc<=LMAX: res.append((Rel(coref(),cpf),cs+cc))
    return res
def E_dp(b):
    res=[(DPp(),1)]
    for nps,ns in E_nps(b):
        if 1+ns<=LMAX: res.append((DPc(nps),1+ns))
    return res
def E_cprel(b):
    res=[]
    if b<1: return res
    for vp,vs in E_vp(b-1):
        if 1+vs<=LMAX: res.append((CPrel(SubjGap(vp)),1+vs))
    for dp,ds in E_dp(b-1):
        if 2+ds<=LMAX: res.append((CPrel(ObjGap(dp,VObjGap())),2+ds))
    return res
def E_cpsent(b):
    res=[]
    if b<1: return res
    for s,ss in E_s(b-1):
        if 1+ss<=LMAX: res.append((CPsent(s),1+ss))
    return res
def E_vp(b):
    res=[(Vin(),1),(VinA(),2)]
    for dp,ds in E_dp(b):
        if 1+ds<=LMAX: res.append((Vdp(dp),1+ds))
        if 2+ds<=LMAX: res.append((VdpA(dp),2+ds))
    if b>=1:
        for cp,cs in E_cpsent(b):
            if 1+cs<=LMAX: res.append((Vcp(cp),1+cs))
            if 2+cs<=LMAX: res.append((VcpA(cp),2+cs))
    return res
def E_s(b):
    res=[]; dps=E_dp(b); vps=E_vp(b)
    for dp,ds in dps:
        for vp,vs in vps:
            if ds+vs<=LMAX: res.append((S(dp,vp),ds+vs))
    return res

allS=E_s(2)
frames=[f for f,sz in allS if (f.depth()==2 or not ONLY_D2)]
print(f"LMAX={LMAX} K={K} only_d2={ONLY_D2}: {len(frames)} frames to test")
stats={m:[0,0] for m in MODNAMES}; fails={m:[] for m in MODNAMES}
oracle_bad=0; tested=0; lenmax=0; t0=time.time()
for fi,f in enumerate(frames):
    seen=set()
    for s in range(K):
        p=sample_pair_from_frame(f, random.Random(9000+fi*17+s))
        if p.hi in seen: continue
        seen.add(p.hi)
        hf=p.hf_tokens; lenmax=max(lenmax,len(p.hi_tokens))
        of=oracle_features(p.hi_tokens)
        if of["hf_tokens"]!=hf: oracle_bad+=1
        tested+=1
        for m in MODNAMES:
            try: got=MODS[m](p.hi_tokens)
            except Exception as e: got=("EXC",repr(e))
            stats[m][1]+=1
            if got==hf: stats[m][0]+=1
            else:
                if len(fails[m])<20: fails[m].append((f.depth(),p.hi_tokens,hf,got))
print(f"tested={tested} secs={time.time()-t0:.1f} oracle_bad={oracle_bad} maxlen={lenmax}")
for m in MODNAMES:
    c,t=stats[m]; print(f"{m}: {c}/{t} = {c/t:.6f}  fails={t-c}")
    fails[m].sort(key=lambda x: len(x[1]))
    for (d,hi,hf,got) in fails[m][:3]:
        print("   FAIL d",d,"len",len(hi),"| HI:"," ".join(hi))
        print("        exp:"," ".join(hf)); print("        got:"," ".join(got) if isinstance(got,list) else got)
