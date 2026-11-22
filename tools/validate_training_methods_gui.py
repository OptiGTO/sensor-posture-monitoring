#!/usr/bin/env python3
"""Validate the shared training-method wizard in real Tk and isolated state.

The fixture registry tests evidence gating without publishing synthetic winners.
Use --registry actual to exercise the currently registered experiment winners.
By default the real GTO application runs with isolated state. --product harness
selects the lightweight shared-screen check. No bundle is built and no
measurement device or training process is started.
"""
import argparse
import csv
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import ttk
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from software.experiment_wizard import ExperimentWizardMixin
from src.app_services import TrainingWizardState
from src.model_catalog import MODEL_NAMES, missing_model_packages
from src.training_methods import available_training_methods


class WizardHarness(ExperimentWizardMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('학습 방식 검증 · 격리된 공유 화면')
        self.geometry('960x900')
        self.training_wizard_state = TrainingWizardState(models=('rbf_svm',))
        self.personal_retrospective = tk.BooleanVar(value=False)
        self.selected_records = []
        self.show_model_select = lambda: None
        self.start_training = lambda: None

    def clear(self):
        for child in self.winfo_children():
            child.destroy()
        frame = ttk.Frame(self)
        frame.pack(fill='both', expand=True)
        return frame

    def heading(self, frame, title, subtitle):
        ttk.Label(frame, text=title).pack()


def check_wizard(registry_kind, require_improvement=False, product='harness'):
    with tempfile.TemporaryDirectory(prefix='posture-method-gui-') as temporary:
        registry = Path(temporary) / 'registry.json'
        registry.write_text(json.dumps({'schema_version': 1, 'methods': [
            {'id': 'selected', 'label': '핵심 특징 96개', 'families': ['rbf_svm'],
             'window_seconds': [3], 'evidence': {'improved': True}},
            {'id': 'relative', 'families': ['rbf_svm'], 'window_seconds': [3],
             'evidence': {'improved': False}},
        ]}), encoding='utf-8')
        context = mock.patch('src.training_methods.REGISTRY_PATH', registry) if registry_kind == 'fixture' else nullcontext()
        with context:
            methods = available_training_methods()
            if require_improvement and len(methods) == 1:
                raise AssertionError('등록된 개선 방식이 없습니다.')
            identity = None
            if product == 'harness':
                app = WizardHarness()
            else:
                state_root = (Path(temporary) / product / 'state').resolve()
                os.environ['POSTURE_APP_STATE_ROOT'] = str(state_root.resolve())
                os.environ['POSTURE_APP_VARIANT'] = product
                from software.posture_app import PostureApplication as ApplicationClass
                from software import posture_app_base as application
                assert application.PRODUCT.variant == product
                assert application.STATE_ROOT == state_root
                app = ApplicationClass()
                assert app.title() == application.PRODUCT.display_name
                assert all(path.resolve().is_relative_to(state_root) for path in app.dataset_service.data_directories)
                identity = {'variant': product, 'title': app.title(),
                            'bundle_identifier': application.PRODUCT.bundle_identifier,
                            'entrypoint': application.PRODUCT.entrypoint,
                            'isolated_state': True}
                from src.app_services import CSV_COLUMNS
                fixture = application.DATA_DIRECTORY / 'qa-participant' / 'sample.csv'
                fixture.parent.mkdir(parents=True, exist_ok=True)
                rows = 12 * max(window for method in methods for window in method['window_seconds']) + 1
                with fixture.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(CSV_COLUMNS)
                    writer.writerows([index / 12, '정자세', *[10 + channel + (index % 9) / 10 for channel in range(16)]]
                                     for index in range(rows))
            checks = []
            try:
                app.show_training_wizard_setup()
                app.update()
                assert list(app.wizard_training_method_ids.values()) == [method['id'] for method in methods]
                if registry_kind == 'fixture':
                    assert list(app.wizard_training_method_ids.values()) == ['standard', 'selected']
                for method in methods:
                    families = [family for family in method['families'] if not missing_model_packages(family)]
                    if not families:
                        raise AssertionError(f"{method['id']}: 실행 가능한 모델 의존성이 없습니다.")
                    window = 3 if method['id'] == 'standard' else method['window_seconds'][0]
                    app.training_wizard_state = TrainingWizardState(models=(families[0],), windows=(window,))
                    app.show_training_wizard_setup()
                    app.update()
                    app.wizard_training_method.set(method['label'])
                    app._refresh_training_method_hint()
                    config = app._remember_experiment_setup()
                    assert config['training_method'] == method['id']
                    with mock.patch('software.experiment_wizard.messagebox.showinfo') as details:
                        app._show_training_method_details()
                    assert details.called
                    if product == 'harness':
                        app.show_training_wizard_review()
                        app.update()
                    else:
                        app._wizard_to_dataset()
                        app.update()
                        assert app.training_wizard_state.current_step == 2, app.wizard_error.get()
                        assert len(app.visible_records) == 1
                        app.select_visible_datasets()
                        app._wizard_to_review()
                        app.update()
                        assert app.training_wizard_state.current_step == 3, app.wizard_error.get()
                        app.show_training_wizard_dataset()
                        app.update()
                        assert len(app.selected_records) == 1
                    app.show_training_wizard_setup()
                    app.update()
                    assert app._selected_training_method()['id'] == method['id']
                    app.wizard_kind.set('personal')
                    app._change_experiment_kind()
                    app.update()
                    assert app._remember_experiment_setup()['training_method'] == 'standard'
                    app.wizard_kind.set('individual')
                    app._change_experiment_kind()
                    app.update()
                    assert app._remember_experiment_setup()['training_method'] == method['id']
                    unsupported = next((family for family in MODEL_NAMES if family not in method['families']), None)
                    rejected = None
                    if method['id'] != 'standard' and unsupported is not None:
                        app.wizard_model_vars[unsupported].set(True)
                        try:
                            app._remember_experiment_setup()
                        except ValueError:
                            rejected = True
                        else:
                            raise AssertionError('지원하지 않는 모델 조합이 허용되었습니다.')
                    checks.append({'method': method['id'], 'family': families[0], 'window_seconds': window,
                                   'review_return': True, 'details': True, 'personal_draft_preserved': True,
                                   'unsupported_family_rejected': rejected,
                                   'real_dataset_navigation': product != 'harness'})
            finally:
                app.destroy()
            return {'real_tk': True, 'visible_methods': [method['id'] for method in methods],
                    'checks': checks, 'canonical_measurements_accessed': False, 'training_started': False,
                    'product_identity': identity,
                    'desktop_products_launched': [] if product == 'harness' else [product], 'bundles_built': []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registry', choices=('fixture', 'actual'), default='fixture')
    parser.add_argument('--require-improvement', action='store_true')
    parser.add_argument('--product', choices=('gto', 'harness'), default='gto')
    args = parser.parse_args(argv)
    result = {'started_at': datetime.now().astimezone().isoformat(), 'registry': args.registry,
              'scope': 'shared wizard Tk harness' if args.product == 'harness' else 'real desktop product classes with isolated state',
              'product': args.product, 'passed': False}
    try:
        result.update(check_wizard(args.registry, args.require_improvement, args.product))
        result['passed'] = True
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        result['completed_at'] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
