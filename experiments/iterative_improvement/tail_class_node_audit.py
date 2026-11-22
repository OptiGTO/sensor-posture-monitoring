"""TRAIN-only class tails and bounded saved-forest node geometry diagnostic.

No estimator is fitted and no prediction/score method is called. decision_path
is used only to recover actual TRAIN membership in the first 20 stored trees.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

FIRST_TREES = 20
QUANTILES = [.01, .99]


def summarize_nodes(rows):
    count = len(rows)
    membership = sum(row['training_rows'] for row in rows)
    variable = [row['scaled']['central_full_ratio'] for row in rows if row['scaled']['central_full_ratio'] is not None]
    outside = [row for row in rows if row['threshold_outside_local_central']]
    return {'internal_nodes': count, 'node_memberships': membership,
        'threshold_outside_local_central_nodes': len(outside),
        'threshold_outside_local_central_fraction': None if not count else len(outside) / count,
        'membership_weighted_outside_fraction': None if not membership else sum(row['training_rows'] for row in outside) / membership,
        'constant_selected_features': count - len(variable),
        'zero_central_variable_nodes': sum(row['scaled']['zero_central_variable'] for row in rows),
        'median_central_full_ratio': None if not variable else float(np.median(variable)),
        'ratio_below_0_1_nodes': sum(value < .1 for value in variable),
        'ratio_below_0_25_nodes': sum(value < .25 for value in variable),
        'raw_inverse_threshold_membership_disagreements': sum(row['raw_inverse_threshold_membership_disagreements'] for row in rows),
        'raw_vs_scaled_central_outside_disagreements': sum(row['raw_central_outside'] != row['threshold_outside_local_central'] for row in rows)}


def interval(values):
    values = values.astype(np.float64)
    low, high = np.quantile(values, QUANTILES, method='linear')
    minimum, maximum = float(values.min()), float(values.max())
    full, central = maximum - minimum, float(high - low)
    return {'minimum': minimum, 'maximum': maximum, 'q01': float(low), 'q99': float(high),
        'full_range': full, 'central_range': central, 'central_full_ratio': None if full == 0 else central / full,
        'zero_central_variable': bool(full > 0 and central == 0)}


def audit(root, run, output):
    from .feature_tail_audit import file_hash, file_evidence, read_json, json_hash, array_hash, require, selected_rows, feature_groups
    started = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    require(not output.exists(), 'Use a fresh independent diagnostic output directory')
    protocol = read_json(run / 'protocol.json')
    protocol_hash = file_hash(run / 'protocol.json')
    require(protocol_hash == read_json(run / 'protocol_lock.json')['sha256'], 'Protocol seal mismatch')
    # This fixed-baseline audit needs its eight immutable algorithm sources;
    # unrelated extensions of the mutable recipe dispatcher are not imported.
    legacy_sources = ('src/iterative_pipeline.py', 'src/posture_core.py', 'src/window_pipeline.py',
        'src/pressure_contrast_features.py', 'src/domain_feature_selection.py',
        'src/group_probability_bias.py', 'src/extended_models.py', 'src/sagittal_hierarchy.py')
    source_paths = [*legacy_sources, 'src/__init__.py', 'src/forest_alternatives_pipeline_v1.py', 'src/tail_clipping_v1.py',
        'src/binary_forest_v1.py', 'experiments/iterative_improvement/feature_tail_audit.py',
        'experiments/iterative_improvement/record_training.py', 'experiments/iterative_improvement/supplemental.py']
    sources = {}
    for relative in source_paths:
        expected = protocol['source_hashes'][relative]
        require(file_hash(root / relative) == expected == file_hash(run / 'source_snapshot' / relative), 'Frozen source mismatch: ' + relative)
        sources[relative] = file_evidence(root / relative)
    sources[str(Path(__file__).resolve().relative_to(root))] = file_evidence(Path(__file__))
    import joblib
    import sklearn
    from src.posture_core import CLASS_NAMES, feature_names, normalize_text
    from src.forest_alternatives_pipeline_v1 import _forest_digest, _scaler_digest
    from .record_training import append_record_training
    names = feature_names()
    blocks = feature_groups(names)
    feature_block = {index: name for name, indices in blocks.items() for index in indices}
    selected_files = [run / name for name in ('protocol.json', 'protocol_lock.json', 'data.npz', 'dataset.json',
        'supplemental_training.npz', 'supplemental_manifest.json', 'protected_before.json', 'audits/alternatives_fitted_identity.json')]
    selected_files.extend((root / 'research/models/checkpoints/research_baseline.json', root / 'src/validated_checkpoints.json'))
    inputs = {str(path.relative_to(root)): file_evidence(path) for path in selected_files}
    baseline_pointer = read_json(root / 'research/models/checkpoints/research_baseline.json')['current_research_baseline']
    require(baseline_pointer['pipeline'] == protocol['baseline'], 'Stored run baseline differs from current adopted recipe')
    require(file_hash(root / baseline_pointer['model']['path']) == baseline_pointer['model']['sha256'], 'Adopted model hash mismatch')
    require(file_hash(run / 'data.npz') == protocol['cache_sha256']
        and file_hash(run / 'dataset.json') == protocol['dataset_sha256'], 'Base cache identity mismatch')
    supplement_manifest = read_json(run / 'supplemental_manifest.json')
    require(file_hash(run / 'supplemental_manifest.json') == protocol['supplemental_training']['manifest_sha256']
        and file_hash(run / 'supplemental_training.npz') == protocol['supplemental_training']['cache_sha256']
        == supplement_manifest['cache_sha256'] and supplement_manifest['personal_reference_used'] is False,
        'Frozen ordinary restoration identity mismatch')
    protected = read_json(run / 'protected_before.json')
    require({relative: file_evidence(root / relative) for relative in protected} == protected, 'Protected originals changed since016 freeze')
    fitted_audit = read_json(run / 'audits/alternatives_fitted_identity.json')
    require(fitted_audit['passed'] and fitted_audit['protocol_sha256'] == protocol_hash, 'Actual TRAIN fitted-state proof missing')
    controls = {row['fold']: row for row in fitted_audit['results'] if row['candidate'] == 'TAIL_identity'}
    caches = {}
    for kind, filename in (('base', 'data.npz'), ('ordinary', 'supplemental_training.npz')):
        with np.load(run / filename, allow_pickle=False) as archive:
            groups, records = archive['groups'], archive['record_ids']
        caches[kind] = {'path': run / filename, 'groups': np.asarray([normalize_text(str(value)) for value in groups]),
            'record_ids': np.asarray([normalize_text(str(value)) for value in records])}
    require(not set(caches['base']['record_ids']) & set(caches['ordinary']['record_ids']), 'Restoration duplicates base recording')
    folds, window_sets = [], []
    for part in protocol['partitions']:
        subjects = sorted(normalize_text(value) for value in part['train'])
        require(not set(subjects) & {normalize_text(value) for key in ('validation', 'test') for value in part[key]}, 'TRAIN overlap')
        selected = {}
        for kind, cache in caches.items():
            take = np.flatnonzero(np.isin(cache['groups'], subjects))
            selected[kind] = {'X': selected_rows(cache['path'], 'X', take, len(cache['groups'])),
                'y': selected_rows(cache['path'], 'y', take, len(cache['groups'])),
                'groups': cache['groups'][take], 'record_ids': cache['record_ids'][take], 'indices': take}
            require(set(selected[kind]['groups']) == set(subjects), 'Incomplete TRAIN subjects')
        base, extra = selected['base'], selected['ordinary']
        raw, labels, groups, records, restoration = append_record_training(base, extra, np.arange(len(base['X'])))
        control = controls[part['id']]
        train_hashes = {'training_input_sha256': array_hash(raw), 'training_labels_sha256': array_hash(labels),
            'training_groups_sha256': json_hash(groups.tolist())}
        require(all(control[key] == value for key, value in train_hashes.items()), 'Actual TRAIN differs from016 fitted-state proof')
        directory = run / 'development/baseline' / part['id']
        model_path, metric_path, seal_path = directory / 'model.joblib', directory / 'metrics.json', directory / 'metrics_seal.json'
        metadata = read_json(metric_path)
        require(file_hash(metric_path) == read_json(seal_path)['sha256'] and file_hash(model_path) == metadata['model_sha256'], 'Baseline artifact seal mismatch')
        require(metadata['pipeline'] == protocol['baseline'] and metadata['protocol_sha256'] == protocol_hash
            and metadata['supplemental_training_audit'] == restoration, 'Baseline recipe/restore linkage mismatch')
        for path in (model_path, metric_path, seal_path):
            inputs[str(path.relative_to(root))] = file_evidence(path)
        model = joblib.load(model_path)
        require(model.spec_ == protocol['baseline'] and model.effective_seed_ == protocol['seeds'][0], 'Fitted baseline recipe/seed mismatch')
        require(model.training_details_ == metadata['training_details'], 'Baseline fitted audit mismatch')
        selector, scaler, forest = [model.estimator_.named_steps[key] for key in ('features', 'scale', 'model')]
        require(selector.mode == 'all' and np.array_equal(selector.indices_, np.arange(596))
            and selector.get_feature_names_out().tolist() == names, 'Baseline feature coordinate schema changed')
        require(_forest_digest(forest) == control['estimator_state_sha256']
            and _scaler_digest(scaler) == control['scaler']['state_sha256'], 'Baseline learned state differs from audited identical TRAIN control')
        scaled = scaler.transform(raw)
        effective_mean = scaler.mean_.astype(raw.dtype)
        effective_scale = scaler.scale_.astype(raw.dtype)
        independent = raw.copy()
        independent -= effective_mean
        independent /= effective_scale
        require(scaled.dtype == np.float32 and scaled.tobytes() == independent.tobytes(), 'Raw-to-scaled float32 arithmetic mismatch')
        require(array_hash(scaled) == control['transformed_state_sha256'], 'Scaled TRAIN bytes differ from audited fitted control')
        require(np.array_equal(scaler.mean_, raw.mean(axis=0, dtype=np.float64))
            and np.allclose(scaler.var_, raw.var(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-12), 'Scaler not bound to actual TRAIN statistics')
        quantiles = np.quantile(raw.astype(np.float64), QUANTILES, axis=0, method='linear')
        tails = (raw < quantiles[0]) | (raw > quantiles[1])
        classes = []
        for class_id, label in enumerate(CLASS_NAMES):
            take = labels == class_id
            class_cells = tails[take]
            classes.append({'class_id': class_id, 'class_name': label, 'training_windows': int(take.sum()),
                'training_records': len(np.unique(records[take])), 'tail_cells': int(class_cells.sum()),
                'total_cells': int(class_cells.size), 'tail_cell_rate': float(class_cells.mean()),
                'groups': {group: {'features': len(indices), 'tail_cells': int(class_cells[:, indices].sum()),
                    'total_cells': int(take.sum()) * len(indices), 'tail_cell_rate': float(class_cells[:, indices].mean())}
                    for group, indices in blocks.items()}})
        node_rows, tree_rows = [], []
        for tree_index, tree in enumerate(forest.estimators_[:FIRST_TREES]):
            path = tree.decision_path(scaled, check_input=False).tocsc()
            learned = tree.tree_
            require(np.array_equal(np.diff(path.indptr), learned.n_node_samples), 'Decision-path TRAIN membership differs from stored node counts')
            depths = np.zeros(learned.node_count, dtype=np.int64)
            tree_nodes = []
            for node in range(learned.node_count):
                left, right = int(learned.children_left[node]), int(learned.children_right[node])
                if left < 0:
                    continue
                depths[left] = depths[right] = depths[node] + 1
                feature = int(learned.feature[node])
                threshold = float(learned.threshold[node])
                members = path.indices[path.indptr[node]:path.indptr[node + 1]]
                x = scaled[members, feature].astype(np.float64)
                raw_x = raw[members, feature].astype(np.float64)
                left_rows = path.indices[path.indptr[left]:path.indptr[left + 1]]
                right_rows = path.indices[path.indptr[right]:path.indptr[right + 1]]
                require(np.array_equal(np.sort(left_rows), np.sort(members[x <= threshold]))
                    and np.array_equal(np.sort(right_rows), np.sort(members[x > threshold])), 'Actual child routing differs from stored threshold')
                scaled_interval, raw_interval = interval(x), interval(raw_x)
                inverse_threshold = threshold * float(effective_scale[feature]) + float(effective_mean[feature])
                row = {'tree_index': tree_index, 'node_id': node, 'depth': int(depths[node]),
                    'feature_index': feature, 'feature_name': names[feature], 'feature_group': feature_block[feature],
                    'training_rows': len(members), 'left_rows': len(left_rows), 'right_rows': len(right_rows),
                    'membership_sha256': array_hash(members.astype(np.int64)),
                    'scaled': scaled_interval, 'raw': raw_interval, 'selected_threshold_scaled': threshold,
                    'selected_threshold_raw_inverse': inverse_threshold,
                    'threshold_outside_local_central': bool(threshold < scaled_interval['q01'] or threshold > scaled_interval['q99']),
                    'raw_central_outside': bool(inverse_threshold < raw_interval['q01'] or inverse_threshold > raw_interval['q99']),
                    'raw_inverse_threshold_membership_disagreements': int(np.count_nonzero((raw_x <= inverse_threshold) != (x <= threshold))),
                    'membership_and_children_verified': True}
                node_rows.append(row)
                tree_nodes.append(row)
            tree_rows.append({'tree_index': tree_index, 'tree_seed': int(tree.random_state),
                'all_nodes': int(learned.node_count), 'all_node_train_counts_verified': True,
                'decision_path_indptr_sha256': array_hash(path.indptr), 'decision_path_indices_sha256': array_hash(path.indices),
                'summary': summarize_nodes(tree_nodes)})
        require(len(tree_rows) == FIRST_TREES, 'Insufficient stored trees for preregistered diagnostic')
        folds.append({'partition': part['id'], 'training_windows': len(raw), 'training_records': len(np.unique(records)),
            'training_subjects': len(set(groups)), 'class_window_counts': np.bincount(labels, minlength=7).tolist(),
            'training_hashes': train_hashes, 'base_train_indices': base['indices'].tolist(), 'ordinary_train_indices': extra['indices'].tolist(),
            'restoration': restoration, 'model_sha256': metadata['model_sha256'], 'scaler_state_sha256': _scaler_digest(scaler),
            'forest_state_sha256': _forest_digest(forest), 'coordinates': {'raw_dtype': str(raw.dtype), 'scaled_dtype': str(scaled.dtype),
                'raw_scaled_arithmetic_bitwise_verified': True, 'scaled_train_sha256': array_hash(scaled),
                'arithmetic': 'sklearn1.8: cast fitted mean/scale to input float32; float32 subtraction then division',
                'effective_mean_sha256': array_hash(effective_mean), 'effective_scale_sha256': array_hash(effective_scale),
                'scaler_matches_audited_no_clipping_control': True},
            'global_train_quantile_bounds_sha256': array_hash(quantiles), 'class_tails': classes,
            'node_summary': summarize_nodes(node_rows),
            'node_summary_by_depth': {str(depth): summarize_nodes([row for row in node_rows if row['depth'] == depth]) for depth in sorted({row['depth'] for row in node_rows})},
            'node_summary_by_feature_group': {group: summarize_nodes([row for row in node_rows if row['feature_group'] == group]) for group in blocks},
            'trees': tree_rows, 'nodes': node_rows})
        window_sets.append({f'{kind}:{index}' for kind, data in selected.items() for index in data['indices']})
    require({rel: file_evidence(root / rel) for rel in protected} == protected, 'Protected originals changed during diagnostic')
    require({rel: file_evidence(root / rel) for rel in inputs} == inputs, 'Inputs changed during diagnostic')
    require({rel: file_evidence(root / rel) for rel in sources} == sources, 'Source changed during diagnostic')
    result = {'schema_version': 1, 'version': 'tail_class_node_audit_v1', 'started_at': started,
        'finished_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), 'timezone': 'Asia/Seoul',
        'run': str(run.relative_to(root)), 'adopted_checkpoint_id': baseline_pointer['checkpoint_id'],
        'protocol_sha256': protocol_hash, 'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'dirty_state': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        'command': [sys.executable, '-B', '-m', 'experiments.iterative_improvement.tail_class_node_audit', '--input-run', str(run), '--output', str(output)],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'installed_scaler_transform': {'source': inspect.getsource(type(scaler).transform),
            'source_sha256': hashlib.sha256(inspect.getsource(type(scaler).transform).encode('utf-8')).hexdigest()},
        'inputs': inputs, 'sources': sources, 'script_source': Path(__file__).read_text(encoding='utf-8'),
        'protected_originals': {'count': len(protected), 'before_after_equal': True, 'manifest_sha256': json_hash(protected)},
        'design': {'classes': 7, 'feature_groups': 50, 'tail_quantiles': QUANTILES, 'quantile_method': 'float64 linear',
            'tail_cells': 'strict raw X<wholeTRAINq01 or X>wholeTRAINq99, denominators class_windows*group_features',
            'tree_selection': 'first20 trees in stored estimator order; all internal nodes; fixed before diagnostic',
            'node_feature': 'only the actual selected split feature for each internal node',
            'node_quantiles': 'q01/q99 recomputed within actual TRAIN membership of that node, on float32 standardized coordinate values promoted to float64',
            'node_outside': 'actual selected threshold<localq01 or >localq99; equal boundary is inside',
            'node_fraction': 'number outside / all analyzed internal nodes',
            'membership_weighted_fraction': 'sum node TRAIN counts outside / sum all internal-node TRAIN counts; repeated visits are not independent samples',
            'raw_coordinates': 'original feature coordinates, not validated physical units; threshold inverse uses the actual float32-cast fitted mean/scale in float64 arithmetic; subtraction/division rounding disagreements are counted',
            'not_observed': 'rejected candidate thresholds, search efficiency, causal information loss or performance effects'},
        'overlap': {'split_window_observations': sum(map(len, window_sets)), 'unique_source_windows': len(set.union(*window_sets)),
            'partitions_and_tree_nodes_are_not_independent': True},
        'partitions': folds, 'model_fits': 0, 'model_prediction_calls': 0, 'performance_evaluations': 0,
        'heldout_feature_or_label_arrays_decoded': False, 'raw_data_or_model_mutated': False,
        'limitations': ['Class-tail association is descriptive and does not establish useful or harmful posture information.',
            'Observed stored split thresholds do not reveal rejected candidate thresholds or prove clipping destroyed separation.',
            'Only the fixed first20 of200 stored trees are analyzed; no extrapolation of uncertainty to all trees or people.',
            'TRAIN partitions, trees, ancestor/descendant nodes and repeated windows are dependent. No inferential confidence interval or evaluation metric is calculated.']}
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'tail_class_node_audit.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    destination.chmod(0o444)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = audit(root, args.input_run.resolve(), args.output.resolve())
    from .feature_tail_audit import file_hash
    print(json.dumps({'path': str(result), 'sha256': file_hash(result)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
