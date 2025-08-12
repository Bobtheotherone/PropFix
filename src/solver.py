from __future__ import annotations
import math
EPS=1e-6

def compute_scales(head_w, sh_w, hp_w, sh_per_head, hp_per_sh, policy):
    tgt_sh=sh_per_head*head_w
    tgt_hp=hp_per_sh*tgt_sh
    s_sh=float(tgt_sh/max(EPS,sh_w)); s_hp=float(tgt_hp/max(EPS,hp_w))
    if policy=="verdict": s_sh=max(1.0,s_sh); s_hp=min(1.0,s_hp)
    return tgt_sh, tgt_hp, s_sh, s_hp

def per_pass_factors(s_total, shrink_floor=0.85, widen_ceiling=1.10):
    if s_total>=1.0:
        n=max(1,int(math.ceil(math.log(s_total+1e-9)/math.log(widen_ceiling+1e-9))))
        return n, s_total**(1.0/n)
    else:
        n=max(1,int(math.ceil(math.log(max(s_total,1e-9))/math.log(max(shrink_floor,1e-9)))))
        return n, s_total**(1.0/n)
