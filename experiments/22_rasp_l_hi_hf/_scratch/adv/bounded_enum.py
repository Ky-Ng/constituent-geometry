import sys, os
from collections import Counter
sys.path.insert(0, os.path.abspath("experiments/22_rasp_l_hi_hf"))
sys.path.insert(0, os.path.abspath("src"))
import grammar.v2.generate_with_frames as G
DPc=G.FrameDPCommon; DPp=G.FrameDPProper
Bar=G.FrameNPsBar; Adj=G.FrameNPsAdj; Rel=G.FrameNPsRel
CPrel=G.FrameCPrel; SubjGap=G.FrameSsubjGap; ObjGap=G.FrameSobjGap; VObjGap=G.FrameVPobjGap
Vin=G.FrameVPIntrans; VinA=G.FrameVPIntransAdv
Vdp=G.FrameVPdp; VdpA=G.FrameVPdpAdv; Vcp=G.FrameVPcp; VcpA=G.FrameVPcpAdv
CPsent=G.FrameCPsent; S=G.FrameS
LMAX=int(sys.argv[1]) if len(sys.argv)>1 else 14

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

if __name__=="__main__" and os.environ.get("COUNT_ONLY"):
    allS=E_s(2)
    bysize_d2=Counter()
    for f,sz in allS:
        if f.depth()==2: bysize_d2[sz]+=1
    print("LMAX",LMAX,"total frames<=LMAX:",len(allS))
    print("depth2 by size:", dict(sorted(bysize_d2.items())))
    c=0
    for s in sorted(bysize_d2):
        c+=bysize_d2[s]; print(f"   size<= {s}: {c}")
