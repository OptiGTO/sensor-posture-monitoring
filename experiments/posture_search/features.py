import numpy as np
from scipy.ndimage import median_filter
from src.window_pipeline import extract_window_features
from src.posture_core import _extract_base_features

def spatial(w):
    r=w/np.maximum(w.sum(2,keepdims=True),1e-6)
    m=r.mean(1);grid=m.reshape(-1,4,4)
    xx,yy=np.meshgrid(np.linspace(-1,1,4),np.linspace(-1,1,4))
    return np.concatenate([m,r.std(1),grid.sum(1),grid.sum(2),
        (m*xx.ravel()).sum(1)[:,None],(m*yy.ravel()).sum(1)[:,None],
        (m*xx.ravel()**2).sum(1)[:,None],(m*yy.ravel()**2).sum(1)[:,None],
        (m*xx.ravel()*yy.ravel()).sum(1)[:,None],
        -(m*np.log(m+1e-9)).sum(1)[:,None]],axis=1).astype(np.float32)

def transform(data,mode='baseline'):
    w=data['seq'];f=data.get('features')
    if mode=='raw':return w
    if mode=='baseline':return f if f is not None else extract_window_features(w)
    if mode=='relative':return transform(data)[:,298:]
    if mode=='spatial':return spatial(w)
    if mode=='personal':return np.concatenate([transform(data),data['personal']],axis=1)
    if mode=='personal_spatial':return np.concatenate([spatial(w),data['personal']],axis=1)
    if mode=='multiscale':return np.concatenate([transform(data),spatial(w[:,-12:]),spatial(w[:,-24:])],axis=1)
    if mode=='log':return extract_window_features(np.log1p(np.maximum(w,0)))
    if mode=='proportion':return np.stack([_extract_base_features(x) for x in w/np.maximum(w.sum(2,keepdims=True),1e-6)])
    if mode=='denoise':return extract_window_features(median_filter(w,size=(1,3,1),mode='nearest'))
    raise ValueError(mode)
