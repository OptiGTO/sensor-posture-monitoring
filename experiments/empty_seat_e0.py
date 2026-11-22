"""Read-only E0 baseline characterization; outputs go to a new isolated run."""
from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def protect(root):
    paths = [p for p in (root / 'data').rglob('*') if p.is_file() and not p.is_symlink()]
    paths += [root / 'research/models' / n for n in ('active_model.json', 'release_selection.json')]
    return {str(p.relative_to(root)): {'sha256': digest(p), 'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(paths) if p.exists()}


def load_frozen(snapshot):
    spec = importlib.util.spec_from_file_location('e0_posture_core', snapshot / 'src/posture_core.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    names = {'StandardProfile', 'OccupancyConfig', 'OccupancyDetector', 'build_standard_profile'}
    tree = ast.parse((snapshot / 'src/app_services.py').read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in selected} == names
    scope = dict(np=np, Path=Path, dataclass=dataclass, field=field, deque=deque,
                 Iterable=Iterable, hashlib=hashlib, read_sensor_csv=core.read_sensor_csv,
                 SENSOR_COLUMNS=core.SENSOR_COLUMNS, _digest=digest, __name__=__name__)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(snapshot / 'src/app_services.py'), 'exec'), scope)
    return core, scope


def score(x, profile):
    delta = np.abs(x - profile.center)
    z = delta / profile.deviation
    return np.median(z, axis=1) + np.mean(z >= 3, axis=1) + delta.sum(axis=1) / profile.deviation.sum()


def summarize(root, out):
    started = now()
    snapshot = out / 'source_snapshot'
    core, runtime = load_frozen(snapshot)
    cols = core.SENSOR_COLUMNS
    paths = sorted((root / 'data/standard').glob('*.csv'))
    assert len(paths) >= 3
    protected = protect(root)
    save(out / 'protected_before.json', protected)
    frames = [core.read_sensor_csv(p) for p in paths]
    values = [f[cols].to_numpy(float) for f in frames]
    times = [f.elapsed_s.to_numpy(float) for f in frames]
    centers = np.array([np.median(x, axis=0) for x in values])
    mads = np.array([1.4826 * np.median(abs(x-b), axis=0) for x, b in zip(values, centers)])
    assert all(set(f.label) == {'standard'} for f in frames)
    assert len({digest(p) for p in paths}) == len(paths)
    assert len({core.sensor_identity(f) for f in frames}) == len(paths)
    ids = [p.stem.rsplit('_', 1)[-1] for p in paths]
    manifest = [{'file': str(p.relative_to(root)), 'sha256': digest(p),
                 'sensor_sha256': core.sensor_identity(f), 'rows': len(f),
                 'measurement_date': None, 'session_id': None, 'role': 'empty_seat'}
                for p, f in zip(paths, frames)]
    save(out / 'source_manifest.json', manifest)
    config = runtime['OccupancyConfig']()
    protocol = dict(experiment='E0', seed=None, stochastic_operations=False,
                    fit='Median of recording medians; original frozen build_standard_profile',
                    split='Leave one complete file out; no tuning; no occupied recordings',
                    smoothing_frames=config.smoothing_frames, enter_score=config.enter_score,
                    exit_score=config.exit_score, enter_hold_seconds=config.enter_hold_seconds,
                    exit_hold_seconds=config.exit_hold_seconds,
                    temporal_weights='State at t[i] applies to [t[i],t[i+1]); no tail extension',
                    same_value_fraction='Fraction of consecutive sample pairs equal, per sensor',
                    units='Unconfirmed original CSV units', confidence_intervals=False,
                    source_extraction='Exact AST definitions from frozen app_services; no UI/module initialization',
                    plan_sha256=digest(snapshot / 'plan.md'))
    save(out / 'protocol.json', protocol)
    global_profile = runtime['build_standard_profile'](root / 'data/standard')
    file_stats, sensor_stats, corr_stats, profiles, folds, timeline = [], [], [], [], [], []
    corr_matrices, smoothed_scores = [], []
    for i, (p, x, t) in enumerate(zip(paths, values, times)):
        dt = np.diff(t)
        assert np.all((dt > 0) & (dt <= 1)), 'Gaps require a separate reset protocol'
        corr = np.corrcoef(x.T)
        assert np.isfinite(corr).all()
        corr_matrices.append(corr)
        upper = corr[np.triu_indices(16, 1)]
        common = np.median(x - centers[i], axis=1)
        fs = dict(file=ids[i], rows=len(x), start_s=float(t[0]), end_s=float(t[-1]),
                  duration_s=float(t[-1]-t[0]), min=float(x.min()), max=float(x.max()),
                  dt_min=float(dt.min()), dt_median=float(np.median(dt)), dt_max=float(dt.max()),
                  dt_q05=float(np.quantile(dt,.05)), dt_q95=float(np.quantile(dt,.95)),
                  dt_mean=float(dt.mean()), zero_mad_channels=int((mads[i]==0).sum()),
                  median_pairwise_correlation=float(np.median(upper)),
                  common_mode_std=float(np.std(common)),
                  negative_all_channels_rows=int(np.all(x-centers[i] < 0,axis=1).sum()))
        file_stats.append(fs)
        for j,c in enumerate(cols):
            q = np.quantile(x[:,j], [.01,.05,.95,.99])
            sensor_stats.append(dict(file=ids[i], sensor=c, median=float(centers[i,j]),
                scaled_mad=float(mads[i,j]), std=float(x[:,j].std()), min=float(x[:,j].min()),
                max=float(x[:,j].max()), q01=float(q[0]),q05=float(q[1]),q95=float(q[2]),q99=float(q[3]),
                consecutive_equal_fraction=float(np.mean(np.diff(x[:,j])==0)),
                unique_values=int(len(np.unique(x[:,j]))),
                first_5s_median=float(np.median(x[t<t[0]+5,j])),
                last_5s_median=float(np.median(x[t>=t[-1]-5,j])),
                last_minus_first_5s=float(np.median(x[t>=t[-1]-5,j])-np.median(x[t<t[0]+5,j]))))
            for k in range(j+1,16):
                corr_stats.append(dict(file=ids[i],sensor_a=c,sensor_b=cols[k],correlation=float(corr[j,k])))
        with tempfile.TemporaryDirectory(prefix='e0-profile-') as directory:
            for k, source in enumerate(paths):
                if k != i: (Path(directory)/source.name).symlink_to(source)
            profile = runtime['build_standard_profile'](Path(directory))
        training = [k for k in range(len(paths)) if k != i]
        b = np.median(centers[training], axis=0)
        between = 1.4826*np.median(abs(centers[training]-b),axis=0)
        within = np.median(mads[training],axis=0)
        floor = np.maximum(np.maximum(abs(b),np.median(abs(b)))*.01,np.finfo(float).eps)
        np.testing.assert_array_equal(profile.center,b)
        np.testing.assert_allclose(profile.deviation,np.maximum.reduce([between,within,floor]),rtol=0,atol=1e-15)
        smooth = np.array([x[max(0,k-config.smoothing_frames+1):k+1].mean(axis=0) for k in range(len(x))])
        scores = score(smooth,profile)
        raw_scores = score(x,profile)
        smoothed_scores.append(scores)
        detector = runtime['OccupancyDetector'](profile)
        states, entries = [], 0
        for timestamp,row in zip(t,x):
            changed = detector.update(float(timestamp),row)
            entries += int(changed is True)
            states.append(detector.occupied)
        weights = np.r_[dt,0.]
        fs_fold = dict(held_out_file=ids[i], training_files=[ids[k] for k in training],
            rows=len(x), duration_s=float(weights.sum()), score_max=float(scores.max()),
            score_q99=float(np.quantile(scores,.99)),score_median=float(np.median(scores)),
            raw_score_max=float(raw_scores.max()), above_enter_rows=int((scores>=config.enter_score).sum()),
            above_exit_rows=int((scores>=config.exit_score).sum()),
            above_enter_seconds=float(weights[scores>=config.enter_score].sum()),
            occupied_seconds=float(np.dot(states,weights)),false_entries=entries,
            max_center_change=float(np.max(abs(b-global_profile.center))),
            max_heldout_median_deviation_ratio=float(np.max(abs(centers[i]-b)/profile.deviation)),
            floor_dominant_channels=int(np.sum((floor>=between)&(floor>=within))),
            max_raw_residual=float(abs(x-b).max()),
            full_fit_score_max=float(score(smooth,global_profile).max()))
        folds.append(fs_fold)
        for j,c in enumerate(cols):
            profiles.append(dict(held_out_file=ids[i],sensor=c,center=float(b[j]),deviation=float(profile.deviation[j]),
                                 between_mad=float(between[j]),within_mad=float(within[j]),relative_floor=float(floor[j]),
                                 heldout_median_residual=float(centers[i,j]-b[j])))
        for k in range(len(x)):
            timeline.append(dict(file=ids[i],elapsed_s=float(t[k]),raw_score=float(raw_scores[k]),
                                 smoothed_score=float(scores[k]),occupied=states[k],
                                 time_weight_s=float(weights[k]),common_mode_residual=float(common[k])))
    for name,rows in [('file_stats',file_stats),('sensor_stats',sensor_stats),('correlations',corr_stats),
                      ('leave_one_file_out_profiles',profiles),('leave_one_file_out_metrics',folds),('timeline',timeline)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    np.savez_compressed(out/'correlation_matrices.npz',files=np.array(ids),sensors=np.array(cols),correlations=corr_matrices)
    overall = []
    for j,c in enumerate(cols):
        overall.append(dict(sensor=c,center=float(global_profile.center[j]),deviation=float(global_profile.deviation[j]),
                            recording_median_range=float(np.ptp(centers[:,j])),
                            median_within_scaled_mad=float(np.median(mads[:,j])),
                            max_record_median_deviation_ratio=float(np.max(abs(centers[:,j]-global_profile.center[j]))/global_profile.deviation[j])))
    pd.DataFrame(overall).to_csv(out/'global_profile.csv',index=False)
    # Hand-solvable checks catch score normalization, floor and state-machine mistakes.
    dummy = runtime['StandardProfile'](np.zeros(16),np.ones(16),'synthetic',1)
    np.testing.assert_allclose(score(np.array([np.zeros(16),np.ones(16)*3]),dummy),[0,7])
    det = runtime['OccupancyDetector'](dummy)
    for timestamp in (0.,.5): assert det.update(timestamp,np.ones(16)*3) is None
    assert det.update(.75,np.ones(16)*3) is True
    os.environ['MPLBACKEND']='Agg'
    os.environ['MPLCONFIGDIR']=str(out/'plot_cache')
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,1,figsize=(13,6),layout='constrained')
    for ax,matrix,title in zip(axs,[centers,mads],['Empty-seat medians (original units)','Within-recording scaled MAD (original units)']):
        im=ax.imshow(matrix,aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,shrink=.85)
        ax.set(xticks=range(16),xticklabels=cols,yticks=range(len(ids)),yticklabels=ids,title=title,ylabel='File')
    fig.savefig(out/'baseline_heatmaps.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(len(ids),2,figsize=(13,12),layout='constrained')
    for i,(x,t,s) in enumerate(zip(values,times,smoothed_scores)):
        rel=t-t[0];ax=axs[i,0]
        ax.fill_between(rel,x.min(axis=1),x.max(axis=1),alpha=.25,label='16-channel min/max')
        ax.plot(rel,np.median(x,axis=1),lw=1,label='Channel median')
        ax.set(title=f'File {ids[i]}: full raw timeline',ylim=(1.134,1.164),ylabel='Original units')
        ax=axs[i,1];ax.plot(rel,s,lw=1);ax.axhline(4,color='#b91c1c',ls='--',label='Enter = 4');ax.axhline(2,color='#b45309',ls=':',label='Exit = 2')
        ax.set(title=f'File {ids[i]}: held-out occupancy score',ylim=(0,4.3),ylabel='Score')
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Seconds from first sample')
    fig.savefig(out/'raw_and_heldout_scores.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,corr,label in zip(axs.flat,corr_matrices,ids):
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set(title=f'File {label}',xticks=[0,5,10,15],xticklabels=[cols[j] for j in [0,5,10,15]],yticks=[0,5,10,15],yticklabels=[cols[j] for j in [0,5,10,15]])
    axs.flat[-1].axis('off');fig.colorbar(im,ax=list(axs.flat),shrink=.75,label='Pearson correlation (descriptive)')
    fig.savefig(out/'channel_correlations.png',dpi=160);plt.close(fig)
    anomalies=[]
    for i in range(len(paths)):
        k,j=np.unravel_index(np.argmin(values[i]),values[i].shape)
        anomalies.append(dict(file=ids[i],csv_line=int(k+2),elapsed_s=float(times[i][k]),sensor=cols[j],
                              value=float(values[i][k,j]),channels_below_file_median=int((values[i][k]<centers[i]).sum()),
                              global_mean_at_event=float(values[i][k].mean())))
    save(out/'minimum_events.json',anomalies)
    fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    i=ids.index('004');x,t=values[i],times[i];k=np.unravel_index(np.argmin(x),x.shape)[0];mask=abs(t-t[k])<=1
    axs[0].plot(t[mask],x[mask],alpha=.6,lw=.9);axs[0].axvline(t[k],color='black',ls=':')
    axs[0].set(title=f'004: all 16 channels around minimum at {t[k]:.4f} s',xlabel='Recorded elapsed_s',ylabel='Original units')
    i=ids.index('005');x,t=values[i],times[i]
    axs[1].plot(t-t[0],x-global_profile.center,alpha=.45,lw=.7)
    axs[1].plot(t-t[0],np.median(x-global_profile.center,axis=1),color='black',lw=2,label='Median residual')
    axs[1].axhline(0,color='black',ls=':');axs[1].legend()
    axs[1].set(title='005: residual relative to all-file reference (descriptive fit)',xlabel='Seconds from first sample',ylabel='Residual (original units)')
    fig.savefig(out/'notable_recordings.png',dpi=180);plt.close(fig)
    protected_after=protect(root)
    save(out/'protected_after.json',protected_after)
    unchanged=protected==protected_after
    summary=dict(started_at=started,finished_at=now(),files=len(paths),rows=sum(map(len,values)),
                 duration_s=sum(f['duration_s'] for f in file_stats),folds=folds,global_profile=overall,
                 file_stats=file_stats,minimum_events=anomalies,
                 protected_files=len(protected),protected_unchanged=unchanged,
                 validation={'profile_independent_recalculation':'passed','known_score_and_hold_cases':'passed',
                             'original_and_sensor_hash_duplicates':False,'all_sensor_and_time_values_finite':True,
                             'timestamps_strict_and_no_gaps_above_1s':True},
                 environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
                 scope='Offline empty-seat E0 only; no training, occupied sensitivity, hardware or GUI validation')
    save(out/'summary.json',summary)
    assert unchanged,'Protected data or selection files changed during execution'
    print(json.dumps(summary,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen',action='store_true')
    args=parser.parse_args();root=args.project_root.resolve();out=args.output.resolve()
    if args.frozen:
        summarize(root,out);return
    out.mkdir(parents=True,exist_ok=False)
    snap=out/'source_snapshot';(snap/'src').mkdir(parents=True)
    sources=[Path(__file__).resolve(),root/'src/app_services.py',root/'src/posture_core.py',root/'docs/08_experiments_and_results/2026-09-23_standard_empty_seat_experiment_plan.md']
    targets=[snap/'empty_seat_e0.py',snap/'src/app_services.py',snap/'src/posture_core.py',snap/'plan.md']
    for source,target in zip(sources,targets):shutil.copy2(source,target)
    command=[sys.executable,str(targets[0]),'--project-root',str(root),'--output',str(out),'--frozen']
    save(out/'execution.json',dict(created_at=now(),command=command,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True),
        source_hashes={str(source.relative_to(root)):digest(target) for source,target in zip(sources,targets)}))
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    subprocess.run(command,env=env,check=True)


if __name__=='__main__':main()
