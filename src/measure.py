from __future__ import annotations
import numpy as np
from typing import List, Tuple

def row_width(mask01: np.ndarray, y: float, thr: float=0.5):
    H,W=mask01.shape; yy=int(np.clip(round(y),0,H-1)); xs=np.where(mask01[yy,:]>=thr)[0]
    if xs.size==0: return 0.0, 0, W-1
    return float(xs[-1]-xs[0]), int(xs[0]), int(xs[-1])

def band_profile(mask01: np.ndarray, y_center: float, band_px: int, thr: float=0.5):
    H,_=mask01.shape; y0=int(np.clip(round(y_center),0,H-1))
    ys=np.arange(max(0,y0-band_px), min(H,y0+band_px+1), dtype=int)
    out=[]
    for yy in ys:
        w,xl,xr=row_width(mask01,yy,thr)
        out.append((yy,w,xl,xr))
    return out

def choose_width(profile: List[Tuple[int,float,int,int]], strategy="median"):
    if not profile: return None
    arr=np.array([w for _,w,_,_ in profile], np.float32)
    order=np.argsort(arr)
    if strategy=="median": idx=int(order[len(arr)//2])
    elif strategy=="p25": idx=int(order[int(0.25*(len(arr)-1))])
    elif strategy=="p20": idx=int(order[int(0.20*(len(arr)-1))])
    elif strategy=="min": idx=int(np.argmin(arr))
    elif strategy=="robust_min":
        a=arr.copy()
        if len(a)>=3:
            a2=a.copy()
            for i in range(1,len(a)-1): a2[i]=np.median(a[i-1:i+2])
            idx=int(np.argmin(a2))
        else: idx=int(np.argmin(a))
    else: idx=int(order[len(arr)//2])
    return profile[idx]
