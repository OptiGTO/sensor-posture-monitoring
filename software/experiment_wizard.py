"""Shared configuration/data/review shell with configuration-specific editors."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from src.experiment_workflow import CONFIGURATIONS, SELECTION_METRIC_LABELS, build_configuration, apply_configuration, validate_registration
from src.model_catalog import MODEL_NAMES, missing_model_packages, model_display_name
from src.training_windows import merge_window_selection
from src.training_methods import available_training_methods
from src.checkpoint_registry import USER_ADOPTION_STATUS


class ExperimentWizardMixin:
    def _experiment_heading(self, frame, step, title):
        self.training_wizard_state.current_step = step
        self.heading(frame, '모델 실험 · ' + title, '구성 → 데이터 → 검토 → 실행 → 결과')

    def show_training_wizard_setup(self):
        frame = self.clear()
        state = self.training_wizard_state
        self._experiment_heading(frame, 1, '1/3 구성')
        self.wizard_kind = tk.StringVar(value=state.configuration_type)
        self.wizard_name = tk.StringVar(value=state.experiment_name)
        self.wizard_seed = tk.StringVar(value=str(state.seed))
        types = ttk.LabelFrame(frame, text='모델 구성', padding=12); types.pack(fill='x')
        for definition in CONFIGURATIONS.values():
            ttk.Radiobutton(types, text=definition.label, variable=self.wizard_kind, value=definition.id,
                command=self._change_experiment_kind).pack(side='left', padx=(0, 24))
        name = ttk.Frame(frame); name.pack(fill='x', pady=12)
        ttk.Label(name, text='실험 이름').pack(side='left', padx=(0, 12))
        ttk.Entry(name, textvariable=self.wizard_name).pack(side='left', fill='x', expand=True)
        definition = CONFIGURATIONS[state.configuration_type]
        ttk.Label(frame, text=definition.description).pack(anchor='w', pady=(0, 8))
        self.wizard_model_vars = None
        self.wizard_voter_vars = {}
        editors = {'models': self._experiment_model_editor, 'personal': self._experiment_personal_editor}
        editors[definition.editor](frame)
        self.wizard_selection_metric = tk.StringVar(value=SELECTION_METRIC_LABELS.get(state.selection_metric, SELECTION_METRIC_LABELS['loso']))
        self.wizard_selection_metric_ids = {label: key for key, label in SELECTION_METRIC_LABELS.items()}
        if definition.editor == 'models':
            # The default metric stays visible; only the seed is behind the advanced toggle.
            metric = ttk.LabelFrame(frame, text='모델 선택 기준', padding=12); metric.pack(fill='x', pady=(10, 0))
            ttk.Combobox(metric, state='readonly', textvariable=self.wizard_selection_metric,
                         values=list(SELECTION_METRIC_LABELS.values()), width=36).pack(side='left')
            ttk.Label(metric, text='LOSO: 최종 평가 외 모든 피험자를 한 명씩 빼고 평가해 모델·분석 창을 고릅니다. 모델마다 피험자 수만큼 추가 학습하므로 시간이 더 걸립니다.',
                      wraplength=520).pack(side='left', padx=12)
        advanced = ttk.Frame(frame); advanced.pack(fill='x', pady=10)
        advanced_body = ttk.Frame(advanced)
        self.wizard_advanced = tk.BooleanVar(value=False)
        def toggle_advanced():
            advanced_body.pack(fill='x', pady=8) if self.wizard_advanced.get() else advanced_body.pack_forget()
        ttk.Checkbutton(advanced, text='평가 설정', variable=self.wizard_advanced, command=toggle_advanced).pack(anchor='w')
        ttk.Label(advanced_body, text='분할 시드').pack(side='left')
        ttk.Entry(advanced_body, textvariable=self.wizard_seed, width=16).pack(side='left', padx=10)
        ttk.Label(advanced_body, text='피험자 분리 · 선택 기준으로 구성 확정 · 최종 평가').pack(side='left')
        self.wizard_error = tk.StringVar()
        ttk.Label(frame, textvariable=self.wizard_error, foreground='#a32c32', wraplength=760).pack(fill='x')
        navigation = ttk.Frame(frame); navigation.pack(fill='x', pady=16)
        ttk.Button(navigation, text='취소', command=self.show_model_select).pack(side='left')
        ttk.Button(navigation, text='다음: 데이터', command=self._wizard_to_dataset).pack(side='right')

    def _experiment_model_editor(self, frame):
        state = self.training_wizard_state
        self._experiment_training_method_editor(frame)
        models = ttk.LabelFrame(frame, text='학습할 모델', padding=12); models.pack(fill='x')
        self.wizard_model_vars = {key: tk.BooleanVar(value=key in state.models) for key in MODEL_NAMES}
        for i, (key, variable) in enumerate(self.wizard_model_vars.items()):
            missing = missing_model_packages(key)
            button = ttk.Checkbutton(models, text=MODEL_NAMES[key] + (' (설치 필요)' if missing else ''),
                variable=variable, command=self._refresh_experiment_voters)
            button.grid(row=i//3, column=i%3, sticky='w', padx=6, pady=4)
            if missing:
                variable.set(False); button.state(['disabled'])
        windows = ttk.LabelFrame(frame, text='분석 창', padding=12); windows.pack(fill='x', pady=10)
        quick = ttk.Frame(windows); quick.pack(fill='x')
        self.wizard_window_vars = {seconds: tk.BooleanVar(value=seconds in state.windows) for seconds in state.QUICK_WINDOWS}
        for i, (seconds, variable) in enumerate(self.wizard_window_vars.items()):
            ttk.Checkbutton(quick, text=f'{seconds}초', variable=variable, command=self._refresh_experiment_voters).grid(row=i//4, column=i%4, sticky='w', padx=12, pady=3)
        custom = ttk.Frame(windows); custom.pack(fill='x', pady=(8, 0))
        ttk.Label(custom, text='직접 입력 (1~60초)').pack(side='left')
        self.wizard_custom_windows = tk.StringVar(value=', '.join(str(n) for n in state.windows if n not in state.QUICK_WINDOWS))
        entry = ttk.Entry(custom, textvariable=self.wizard_custom_windows, width=24); entry.pack(side='left', padx=10)
        entry.bind('<KeyRelease>', self._refresh_experiment_voters)
        ttk.Label(custom, text='여러 값은 쉼표 또는 공백으로 구분').pack(side='left')
        self.wizard_voters_frame = None
        if CONFIGURATIONS[state.configuration_type].ensemble:
            self.wizard_voters_frame = ttk.LabelFrame(frame, text='앙상블에 포함할 모델 · 동일 가중 Soft voting', padding=12)
            self.wizard_voters_frame.pack(fill='x', pady=8)
            self._refresh_experiment_voters()
        self._apply_checkpoint_selection()

    def _experiment_personal_editor(self, frame):
        panel = ttk.LabelFrame(frame, text='개인 기준 설정', padding=14); panel.pack(fill='x', pady=8)
        ttk.Label(panel, text='학습 방식: 기본 · 개인 기준 고정 구성').pack(anchor='w', pady=(0, 8))
        ttk.Label(panel, text='분석 창 3초 · 기존 개인 기준 2단계 구성').pack(anchor='w', pady=(0, 8))
        ttk.Radiobutton(panel, text='정자세 등록 기록 사용', variable=self.personal_retrospective, value=False).pack(anchor='w', pady=4)
        self.personal_retrospective.set(False)
        ttk.Label(panel, text='실제 등록 정보가 연결된 데이터를 사용합니다. 기존 정자세 CSV를 등록용으로 배정하지 않습니다.', wraplength=760).pack(anchor='w', pady=10)

    def _experiment_training_method_editor(self, frame):
        panel = ttk.Frame(frame); panel.pack(fill='x', pady=(0, 10))
        ttk.Label(panel, text='학습 방식').pack(side='left', padx=(0, 12))
        self.wizard_training_methods = available_training_methods()
        self.wizard_training_method_ids = {
            method['label']: method['id'] for method in self.wizard_training_methods
        }
        selected = self.training_wizard_state.training_method
        label = next((method['label'] for method in self.wizard_training_methods
                      if method['id'] == selected), '사용 불가: ' + selected)
        self.wizard_training_method = tk.StringVar(value=label)
        choice = ttk.Combobox(panel, state='readonly', textvariable=self.wizard_training_method,
                              values=list(self.wizard_training_method_ids), width=28)
        choice.pack(side='left')
        choice.bind('<<ComboboxSelected>>', self._select_training_method)
        ttk.Button(panel, text='자세히 보기', command=self._show_training_method_details).pack(side='left', padx=8)
        self.wizard_training_method_hint = tk.StringVar()
        ttk.Label(frame, textvariable=self.wizard_training_method_hint, wraplength=760).pack(anchor='w', pady=(0, 8))
        self._refresh_training_method_hint()

    def _selected_training_method(self):
        selected_id = self.wizard_training_method_ids.get(self.wizard_training_method.get())
        return next((method for method in self.wizard_training_methods
                     if method['id'] == selected_id), None)

    def _select_training_method(self, _event=None):
        self._apply_checkpoint_selection()
        self._refresh_training_method_hint()

    def _apply_checkpoint_selection(self):
        method = self._selected_training_method()
        checkpoint = (method or {}).get('checkpoint')
        if checkpoint is None or self.wizard_model_vars is None:
            if self.wizard_model_vars is not None:
                for family in tuple(self.wizard_model_vars):
                    if family not in MODEL_NAMES:
                        del self.wizard_model_vars[family]
            return
        pipeline = checkpoint['pipeline']
        if pipeline['family'] not in self.wizard_model_vars:
            self.wizard_model_vars[pipeline['family']] = tk.BooleanVar(value=True)
        for family, variable in self.wizard_model_vars.items():
            variable.set(family == pipeline['family'])
        seconds = pipeline['window_seconds']
        for window, variable in self.wizard_window_vars.items():
            variable.set(window == seconds)
        self.wizard_custom_windows.set('' if seconds in self.wizard_window_vars else str(seconds))
        self.wizard_seed.set(str(pipeline['training']['seed']))
        self._refresh_experiment_voters()

    def _refresh_training_method_hint(self, _event=None):
        method = self._selected_training_method()
        if method is None:
            hint = '이전 학습 방식은 현재 사용할 수 없습니다. 학습 방식을 다시 선택하세요.'
        elif method['id'] == 'standard':
            hint = '기존 모델별 학습 설정'
        else:
            families = ', '.join(model_display_name(family) for family in method['families'])
            windows = ', '.join(f'{seconds}초' for seconds in method['window_seconds'])
            hint = f'지원: {families} · {windows}'
            if method.get('checkpoint'):
                status = method['checkpoint']['status']
                hint += (' · 사용자 채택 연구 기준선' if status == USER_ADOPTION_STATUS else ' · 후향적 검증 체크포인트')
                hint += ' · 저장된 전체 파이프라인으로 재학습'
        self.wizard_training_method_hint.set(hint)

    def _show_training_method_details(self):
        method = self._selected_training_method()
        if method is None:
            messagebox.showinfo('학습 방식', self.wizard_training_method_hint.get()); return
        if method['id'] == 'standard':
            text = '기존 모델별 후보 설정을 사용합니다. 피험자를 나누어 학습·검증·최종 평가합니다.'
        else:
            text = self.wizard_training_method_hint.get() + '\n\n'
            checkpoint = method.get('checkpoint')
            if checkpoint and checkpoint['status'] == USER_ADOPTION_STATUS:
                text += ('사용자 지정으로 채택한 연구 기준선입니다. '
                         '유의미한 개선 기준 전체의 통과를 뜻하지 않으며, 평가 결과와 미충족 기준은 보고서에 기록되어 있습니다. ')
            elif checkpoint:
                text += '기존 자료의 피험자 분리 평가에서 개선·재현 기준을 통과한 체크포인트입니다. '
            else:
                text += '기존 데이터의 피험자 분리 재실험에서 개선을 확인한 방식입니다. '
            text += ('선택한 데이터에서도 같은 향상을 보장하지 않으며, 신규 피험자 검증이 필요합니다. '
                     '학습 결과는 별도 검증·승인을 거쳐 측정 모델로 사용할 수 있습니다.')
            evidence = method.get('evidence') or {}
            if evidence.get('summary'):
                text += '\n\n' + str(evidence['summary'])
            if checkpoint:
                text += ('\n\n재학습에는 사용자가 선택한 적격 기록을 사용합니다. '
                         '저장된 전체 파이프라인을 적용하고, 전처리와 모델은 학습 피험자의 기록으로 다시 학습합니다.')
                if checkpoint['pipeline']['training'].get('data_policy') == 'include_ordinary_reserved_training_records_v1':
                    text += ('\n일반 정자세 기록 복원 정책은 기존 연구 모델의 학습 기록 출처를 나타냅니다. '
                             '새 학습에 과거 연구의 추가 기록 캐시를 자동으로 넣지 않습니다.')
                text += '\n\n기존 연구 자료로 학습한 저장 모델: ' + checkpoint['files']['model']['path']
                text += '\n실험 보고서: ' + checkpoint['files']['report']['path']
                text += '\n\n정확한 파이프라인:\n' + json.dumps(checkpoint['pipeline'], ensure_ascii=False, indent=2)
        messagebox.showinfo(method['label'], text)

    def _refresh_experiment_voters(self, _event=None):
        host = getattr(self, 'wizard_voters_frame', None)
        if host is None or not host.winfo_exists(): return
        state = self.training_wizard_state
        if self.wizard_voter_vars:
            state.soft_voting_members = tuple(key for key, value in self.wizard_voter_vars.items() if value.get())
        for child in host.winfo_children(): child.destroy()
        self.wizard_voter_vars = {}
        try:
            windows = merge_window_selection([n for n,v in self.wizard_window_vars.items() if v.get()], self.wizard_custom_windows.get())
        except ValueError as error:
            ttk.Label(host, text=str(error), wraplength=760).pack(anchor='w'); return
        choices = [f'{family}:{seconds}' for family, value in self.wizard_model_vars.items() if value.get() for seconds in windows]
        chosen = state.soft_voting_members if state.voter_selection_initialized else choices
        if choices: state.voter_selection_initialized = True
        for i, key in enumerate(choices):
            family, seconds = key.split(':')
            variable = self.wizard_voter_vars[key] = tk.BooleanVar(value=key in chosen)
            ttk.Checkbutton(host, text=f'{model_display_name(family)} · {seconds}초', variable=variable).grid(row=i//2, column=i%2, sticky='w', padx=8, pady=3)
        ttk.Label(host, text='2개 이상 선택 · 같은 기록·종료 시점에서 확률 평균과 개별 모델을 비교', wraplength=760).grid(row=(len(choices)+1)//2, column=0, columnspan=2, sticky='w', pady=8)

    def _remember_experiment_setup(self):
        state = self.training_wizard_state
        state.experiment_name = self.wizard_name.get()
        state.seed = int(self.wizard_seed.get())
        choice = getattr(self, 'wizard_selection_metric', None)
        if choice is not None:
            state.selection_metric = self.wizard_selection_metric_ids.get(choice.get(), 'loso')
        definition = CONFIGURATIONS[state.configuration_type]
        state.soft_voting_enabled = definition.ensemble
        if self.wizard_model_vars is not None:
            method = self._selected_training_method()
            if method is None:
                raise ValueError('사용 가능한 학습 방식을 선택하세요.')
            state.training_method = method['id']
            windows = merge_window_selection([n for n,v in self.wizard_window_vars.items() if v.get()], self.wizard_custom_windows.get())
            state.configure(windows, state.seed, [key for key,value in self.wizard_model_vars.items() if value.get()])
            if definition.ensemble:
                # Rebuild after direct-entry edits, while preserving checked members.
                self._refresh_experiment_voters()
                state.soft_voting_members = tuple(key for key,value in self.wizard_voter_vars.items() if value.get())
        return build_configuration(state, allow_retrospective=self.personal_retrospective.get())

    def _change_experiment_kind(self):
        try: self._remember_experiment_setup()
        except ValueError: pass  # Switching away is allowed; valid draft values remain.
        self.training_wizard_state.configuration_type = self.wizard_kind.get()
        self.training_wizard_state.soft_voting_enabled = CONFIGURATIONS[self.wizard_kind.get()].ensemble
        self.show_training_wizard_setup()

    def _wizard_to_dataset(self):
        try: self._remember_experiment_setup()
        except ValueError as error:
            self.wizard_error.set(str(error)); return
        self.show_training_wizard_dataset()

    def show_training_wizard_dataset(self):
        frame = self.clear()
        self._experiment_heading(frame, 2, '2/3 데이터')
        configuration = build_configuration(self.training_wizard_state, allow_retrospective=self.personal_retrospective.get())
        ttk.Label(frame, text=configuration['label'] + (' · 등록 기록과 학습·평가 기록을 분리합니다.' if configuration['personalized'] else ' · 품질 통과 라벨 데이터를 선택하세요.')).pack(anchor='w', pady=(0, 8))
        self._build_dataset_selector(frame, allow_status_changes=False)
        self.wizard_error = tk.StringVar()
        ttk.Label(frame, textvariable=self.wizard_error, foreground='#a32c32', wraplength=760).pack(fill='x', pady=8)
        navigation = ttk.Frame(frame); navigation.pack(fill='x', pady=14)
        ttk.Button(navigation, text='이전: 구성', command=self.show_training_wizard_setup).pack(side='left')
        ttk.Button(navigation, text='다음: 검토', command=self._wizard_to_review).pack(side='right')
        self.refresh_dataset_table()

    def _wizard_to_review(self):
        try:
            if not self.selected_records: raise ValueError('학습에 사용할 데이터를 하나 이상 선택하세요.')
            configuration = build_configuration(self.training_wizard_state, allow_retrospective=self.personal_retrospective.get())
            validate_registration(self.selected_records, configuration)
        except (OSError, ValueError) as error:
            self.wizard_error.set(str(error)); return
        self.training_wizard_state.remember_records(self.selected_records)
        self.show_training_wizard_review()

    def show_training_wizard_review(self):
        frame = self.clear()
        self._experiment_heading(frame, 3, '3/3 검토')
        config = build_configuration(self.training_wizard_state, allow_retrospective=self.personal_retrospective.get())
        panel = ttk.LabelFrame(frame, text=config['name'], padding=16); panel.pack(fill='x')
        lines = ['모델 구성: ' + config['label'], '학습 방식: ' + config['training_method_label'],
            '분석 창: ' + ', '.join(f'{n}초' for n in config['windows']),
            f"선택 데이터: {len(self.selected_records)}개 · 분할 시드: {config['seed']}"]
        if config['personalized']:
            lines += ['등록 방식: ' + ('과거 기록 실험 · 빈 좌석 0 가정' if config['allow_retrospective'] else '연결된 정자세 등록 기록'),
                '학습 설정: 기존 개인 기준 2단계의 고정 후보군', '비교군: 기존 단일 모델 · 일반 2단계 · 개인 표준화 2단계']
        else:
            lines += ['학습 모델: ' + ', '.join(model_display_name(f) for f in config['models']),
                '모델 선택 기준: ' + config['selection_metric_label']]
            if config['soft_voting_members']:
                count = len(config['soft_voting_members'])
                lines += [f"학습 후보 {len(config['models'])*len(config['windows'])}개 · 앙상블 참여 {count}개 · 각각 1/{count} 가중치",
                    '구성: ' + ' + '.join(model_display_name(m.split(':')[0]) + ' ' + m.split(':')[1] + '초' for m in config['soft_voting_members']),
                    '평가: 같은 기록·종료 시점의 구성 모델과 비교 · 공통 시점은 데이터 준비 중 확인']
        for line in lines: ttk.Label(panel, text=line, wraplength=760).pack(anchor='w', pady=4)
        ttk.Label(frame, text=('피험자 분리 → 검증으로 매개변수 선택 → LOSO로 모델·창 확정 → 최종 평가 → 결과 저장'
            if config.get('selection_metric') == 'loso' else '피험자 분리 → 검증 후 구성 확정 → 최종 평가 → 결과 저장'), wraplength=760).pack(anchor='w', pady=12)
        ttk.Label(frame, text='앙상블·개인 표준화는 실험용입니다. 실행만으로 현재 측정 모델을 바꾸지 않습니다.', wraplength=760).pack(anchor='w')
        navigation = ttk.Frame(frame); navigation.pack(fill='x', pady=18)
        ttk.Button(navigation, text='이전: 데이터', command=self.show_training_wizard_dataset).pack(side='left')
        ttk.Button(navigation, text='구성 수정', command=self.show_training_wizard_setup).pack(side='left', padx=8)
        ttk.Button(navigation, text='실험 시작', command=self.start_training).pack(side='right')

    def clone_experiment(self, report_path):
        path = Path(report_path)
        run = path.parent.parent if path.name == 'model_comparison.json' else path.parent
        try:
            config = json.loads((run / 'experiment.json').read_text())
            apply_configuration(self.training_wizard_state, config)
            self.training_wizard_state.source_experiment = str(run.resolve())
            self.personal_retrospective.set(False)
            paths = {Path(p).resolve() for p in config.get('selected_records', [])}
            self.selected_records = [r for r in self.dataset_service.records() if r.csv_path.resolve() in paths]
            self.training_wizard_state.remember_records(self.selected_records)
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror('설정 복제 오류', str(error)); return
        self.show_training_wizard_setup()
