"""Measure full feature-to-stateful-decision CPU cost on validation windows only."""
import argparse
import gc
import json
import os
import platform
from pathlib import Path
import resource
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .inference import StreamingPredictor
from .prepare import OUT, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    manifest=json.loads((a.output/'dataset.json').read_text());data=dict(np.load(a.output/'data.npz'))
    mask=np.isin(data['groups'],manifest['partitions'][0]['validation'])
    seq=data['seq'][mask];records=data['record_ids'][mask];starts=data['start_s'][mask]
    t=time.perf_counter();model=joblib.load(a.output/a.id/'primary/checkpoint.joblib');load_seconds=time.perf_counter()-t
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:pass
    predictor=StreamingPredictor(model)
    with threadpool_limits(limits=2):
        for i in range(10):predictor.predict_proba(seq[i],start_s=float(starts[i]),record_id=records[i])
        predictor.reset();times=[];probs=[]
        for i in range(len(seq)):
            t=time.perf_counter();probs.append(predictor.predict_proba(seq[i],start_s=float(starts[i]),record_id=records[i]))
            times.append((time.perf_counter()-t)*1000)
    expected=np.load(a.output/a.id/'primary/validation_predictions.npz')['proba']
    max_error=float(np.max(np.abs(expected-np.stack(probs))))
    assert np.allclose(expected,probs,atol=2e-6),max_error
    obj={'id':a.id,'scope':'36x16 resampled raw window -> 596 feature extraction -> all classifiers -> causal temporal state -> decision probabilities',
         'window_collection_seconds':3,'resampling_excluded':True,'input_windows':len(seq),
         'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
         'p99_ms':float(np.percentile(times,99)),'load_seconds':load_seconds,
         'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
         'memory_scope':'separate Python process including cached dataset, imported sklearn and torch; not incremental model RAM',
         'stream_batch_max_probability_error':max_error,'argmax_agreement':float(np.mean(np.argmax(probs,1)==expected.argmax(1))),
         'threads':2,'platform':platform.platform(),'cpu_count':os.cpu_count(),
         'physical_realtime_validation':False}
    write_json(a.output/f'runtime_{a.id}.json',obj);print(json.dumps(obj,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
