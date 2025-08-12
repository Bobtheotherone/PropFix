from __future__ import annotations
import numpy as np, cv2

def band_edges(mask01):
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    er=cv2.erode(mask01.astype(np.uint8),k,1)
    return (mask01.astype(np.uint8)-er)>0

def ssim(a,b, mask=None):
    # simple SSIM (luminance only) with Gaussian blur; mask selects region to score
    a=a.astype(np.float32); b=b.astype(np.float32)
    if a.ndim==3: a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim==3: b=cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    C1=6.5025; C2=58.5225
    mu1=cv2.GaussianBlur(a,(11,11),1.5)
    mu2=cv2.GaussianBlur(b,(11,11),1.5)
    mu1_sq=mu1*mu1; mu2_sq=mu2*mu2; mu12=mu1*mu2
    sigma1_sq=cv2.GaussianBlur(a*a,(11,11),1.5)-mu1_sq
    sigma2_sq=cv2.GaussianBlur(b*b,(11,11),1.5)-mu2_sq
    sigma12=cv2.GaussianBlur(a*b,(11,11),1.5)-mu12
    ssim_map=((2*mu12+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if mask is not None:
        m=mask.astype(np.float32); m/=max(1.0, m.sum())
        return float((ssim_map*m).sum())
    return float(ssim_map.mean())

def ssim_outside_roi(img0,img1, roi_mask01):
    inv=(roi_mask01<0.5).astype(np.float32)
    return ssim(img0,img1, mask=inv)

def seam_energy(img, roi_mask01, band_px=20):
    # L2 of Laplacian in narrow band around ROI boundary
    edges=band_edges(roi_mask01)
    dist=cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    band=(dist<=band_px).astype(np.uint8)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    lap=cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.sqrt((lap*band).var()))
